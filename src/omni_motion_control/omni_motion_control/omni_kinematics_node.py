# pyright: reportMissingImports=false
# This node translates "where the target is" into robot wheel commands.
import math
from typing import List

import rclpy
from geometry_msgs.msg import Twist, Vector3
from rclpy.node import Node
from std_msgs.msg import Int16MultiArray , Float32MultiArray


class OmniKinematicsNode(Node):
    """Converts distance/angle commands into per-wheel PWM commands."""

    def __init__(self) -> None:
        super().__init__('omni_kinematics_node')

        # Topic names and control settings are parameters so they can be tuned
        # from launch files without editing this source code.
        self.declare_parameter('target_topic', '/bean_bag_trajectory')
        self.declare_parameter('wheel_cmd_topic', '/motion/wheel_cmd')
        self.declare_parameter('robot_cmd_topic', '/motion/robot_cmd')
        self.declare_parameter('control_rate_hz', 50.0)
        self.declare_parameter('command_timeout_sec', 0.3)

        self.declare_parameter('distance_to_speed_gain', 0.020)
        self.declare_parameter('angle_to_omega_gain', 1.20)

        self.declare_parameter('max_vx_mps', 1.2)
        self.declare_parameter('max_vy_mps', 1.2)
        self.declare_parameter('max_omega_radps', 2.5)
        self.declare_parameter('max_wheel_cmd', 255)

        self.declare_parameter('board_half_length_m', 0.6096)
        self.declare_parameter('board_half_width_m', 0.3048)
        self.declare_parameter('wheel_radius_m', 0.0485)

        # Resolve configured topic names once at startup.
        target_topic = self.get_parameter('target_topic').value
        wheel_cmd_topic = self.get_parameter('wheel_cmd_topic').value
        robot_cmd_topic = self.get_parameter('robot_cmd_topic').value

        # Input: target distance/angle.

        self.target_sub = self.create_subscription(
            Float32MultiArray, target_topic, self.target_callback, 1
        )

        # Outputs: wheel-level command and body-level command for debugging/tools.
        self.wheel_pub = self.create_publisher(Int16MultiArray, wheel_cmd_topic, 1)
        self.robot_pub = self.create_publisher(Twist, robot_cmd_topic, 1)

        # Latest target command remembered between timer ticks.
        self.latest_distance_in = 0.0
        self.latest_angle_rad = 0.0
        self.last_target_time = self.get_clock().now()

        # Main control loop timer (for example, 50 Hz -> every 20 ms).
        rate_hz = float(self.get_parameter('control_rate_hz').value)
        timer_period = 1.0 / max(rate_hz, 1.0)
        self.control_timer = self.create_timer(timer_period, self.control_loop)

        self.get_logger().info(
            f'Omni kinematics ready. in={target_topic} out={wheel_cmd_topic}'
        )

    def target_callback(self, msg: Vector3) -> None:
        # Store the newest target and timestamp for timeout safety handling.
        self.latest_distance_in = float(msg.data[0])
        self.latest_angle_rad = float(msg.data[1])
        self.last_target_time = self.get_clock().now()

    def control_loop(self) -> None:
        now = self.get_clock().now()
        timeout_sec = float(self.get_parameter('command_timeout_sec').value)
        age_sec = (now - self.last_target_time).nanoseconds / 1e9

        # If the target is stale, command zero motion to avoid runaway behavior.
        if age_sec > timeout_sec:
            distance_in = 0.0
            angle_rad = 0.0
        else:
            distance_in = self.latest_distance_in
            angle_rad = self.latest_angle_rad

        # Convert target error into a desired robot motion command.
        vx, vy, omega = self.to_body_command(distance_in, angle_rad)
        self.publish_body_command(vx, vy, omega)
        # Convert body motion into four wheel commands.
        wheel_cmd = self.inverse_kinematics_to_pwm(vx, vy, omega)
        self.wheel_pub.publish(Int16MultiArray(data=wheel_cmd))

    def to_body_command(self, distance_in: float, angle_rad: float) -> tuple:
        # These two gains are simple proportional control terms.
        speed_gain = float(self.get_parameter('distance_to_speed_gain').value)
        omega_gain = float(self.get_parameter('angle_to_omega_gain').value)

        # Resolve distance+angle into x/y velocity and yaw rate.
        vx = speed_gain * distance_in * math.cos(angle_rad)
        vy = speed_gain * distance_in * math.sin(angle_rad)
        omega = omega_gain * angle_rad

        # Clamp to configured safety limits.
        max_vx = float(self.get_parameter('max_vx_mps').value)
        max_vy = float(self.get_parameter('max_vy_mps').value)
        max_omega = float(self.get_parameter('max_omega_radps').value)

        vx = max(min(vx, max_vx), -max_vx)
        vy = max(min(vy, max_vy), -max_vy)
        omega = max(min(omega, max_omega), -max_omega)
        return (vx, vy, omega)

    def publish_body_command(self, vx: float, vy: float, omega: float) -> None:
        # Publish robot-frame velocity command for visibility and debugging.
        msg = Twist()
        msg.linear.x = vx
        msg.linear.y = vy
        msg.angular.z = omega
        self.robot_pub.publish(msg)

    def inverse_kinematics_to_pwm(
        self, vx: float, vy: float, omega: float
    ) -> List[int]:
        # Geometry of robot footprint and wheel size.
        half_length = float(self.get_parameter('board_half_length_m').value)
        half_width = float(self.get_parameter('board_half_width_m').value)
        wheel_radius = float(self.get_parameter('wheel_radius_m').value)
        max_cmd = int(self.get_parameter('max_wheel_cmd').value)

        # Classic 4-wheel omni inverse kinematics assumption.
        k_geom = half_length + half_width
        fl = (vx - vy - k_geom * omega) / wheel_radius
        fr = (vx + vy + k_geom * omega) / wheel_radius
        rl = (vx + vy - k_geom * omega) / wheel_radius
        rr = (vx - vy + k_geom * omega) / wheel_radius
        wheel_vals = [fl, fr, rl, rr]

        # If any wheel would exceed max command, scale all equally.
        max_abs = max(abs(v) for v in wheel_vals) if wheel_vals else 0.0
        scale = 1.0
        if max_abs > float(max_cmd):
            scale = float(max_cmd) / max_abs

        return [int(round(v * scale)) for v in wheel_vals]


def main(args=None) -> None:
    # Standard ROS 2 node startup/shutdown pattern.
    rclpy.init(args=args)
    node = OmniKinematicsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
