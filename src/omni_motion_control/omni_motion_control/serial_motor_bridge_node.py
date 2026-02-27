# pyright: reportMissingImports=false
# This node is the serial "translator" between ROS topics and Arduino commands.
from typing import List, Optional

import rclpy
import serial
from geometry_msgs.msg import Vector3
from rclpy.node import Node
from std_msgs.msg import Bool, Int16MultiArray, String


class SerialMotorBridgeNode(Node):
    """Bridges ROS motion commands to Arduino serial protocol."""

    def __init__(self) -> None:
        super().__init__('serial_motor_bridge_node')

        # Parameterized topic names make integration with other nodes easier.
        self.declare_parameter('wheel_cmd_topic', '/motion/wheel_cmd')
        self.declare_parameter('target_topic', '/motion/target_da')
        self.declare_parameter('safety_stop_topic', '/motion/safety_stop')
        self.declare_parameter('serial_rx_topic', '/motion/serial_rx')
        self.declare_parameter('serial_status_topic', '/motion/serial_status')

        # Serial connection and timing settings.
        self.declare_parameter('serial_port', '/dev/ttyACM0')
        self.declare_parameter('baud_rate', 115200)
        self.declare_parameter('command_mode', 'da')
        self.declare_parameter('send_rate_hz', 50.0)
        self.declare_parameter('command_timeout_sec', 0.3)
        self.declare_parameter('reconnect_interval_sec', 2.0)

        wheel_cmd_topic = self.get_parameter('wheel_cmd_topic').value
        target_topic = self.get_parameter('target_topic').value
        safety_topic = self.get_parameter('safety_stop_topic').value
        serial_rx_topic = self.get_parameter('serial_rx_topic').value
        serial_status_topic = self.get_parameter('serial_status_topic').value

        # Inputs from kinematics/safety.
        self.create_subscription(
            Int16MultiArray, wheel_cmd_topic, self.wheel_cmd_callback, 1
        )
        self.create_subscription(Vector3, target_topic, self.target_callback, 1)
        self.create_subscription(Bool, safety_topic, self.safety_callback, 1)

        # Outputs from serial device back to ROS.
        self.serial_rx_pub = self.create_publisher(String, serial_rx_topic, 10)
        self.serial_status_pub = self.create_publisher(String, serial_status_topic, 10)

        # Last known command state used by periodic transmit loop.
        self.last_wheel_cmd: List[int] = [0, 0, 0, 0]
        self.last_distance_in = 0.0
        self.last_angle_rad = 0.0
        self.last_cmd_time = self.get_clock().now()
        self.safety_stop_active = False

        self._serial: Optional[serial.Serial] = None
        self._last_connect_attempt_ns = 0
        self._last_tx_line = ''

        # Main I/O heartbeat: reconnect, read firmware lines, write next command.
        send_rate_hz = float(self.get_parameter('send_rate_hz').value)
        timer_period = 1.0 / max(send_rate_hz, 1.0)
        self.io_timer = self.create_timer(timer_period, self.io_loop)

        self.get_logger().info('Serial motor bridge ready')

    def wheel_cmd_callback(self, msg: Int16MultiArray) -> None:
        # Expect [FL, FR, RL, RR]. Ignore malformed packets.
        if len(msg.data) < 4:
            return
        self.last_wheel_cmd = [self._clamp_pwm(int(v)) for v in msg.data[:4]]
        self.last_cmd_time = self.get_clock().now()

    def target_callback(self, msg: Vector3) -> None:
        # Distance/angle mode input (used when command_mode == "da").
        self.last_distance_in = float(msg.x)
        self.last_angle_rad = float(msg.y)
        self.last_cmd_time = self.get_clock().now()

    def safety_callback(self, msg: Bool) -> None:
        # Safety takes priority over all other command sources.
        self.safety_stop_active = bool(msg.data)
        if self.safety_stop_active:
            self.get_logger().warn('Safety stop active')

    def io_loop(self) -> None:
        # Keep loop short and deterministic: connect/read/write every tick.
        self.ensure_serial()
        self.read_serial_lines()
        self.write_command_line()

    def ensure_serial(self) -> None:
        # If already connected, do nothing.
        if self._serial is not None and self._serial.is_open:
            return

        now_ns = self.get_clock().now().nanoseconds
        reconnect_sec = float(self.get_parameter('reconnect_interval_sec').value)
        # Throttle reconnection attempts.
        if (now_ns - self._last_connect_attempt_ns) < int(reconnect_sec * 1e9):
            return

        self._last_connect_attempt_ns = now_ns
        port = str(self.get_parameter('serial_port').value)
        baud = int(self.get_parameter('baud_rate').value)

        try:
            # Non-blocking serial read via timeout=0.
            self._serial = serial.Serial(port=port, baudrate=baud, timeout=0)
            self.publish_status(f'SERIAL_CONNECTED {port} {baud}')
            self.get_logger().info(f'Connected to firmware at {port} {baud}')
        except serial.SerialException as exc:
            self._serial = None
            self.publish_status(f'SERIAL_CONNECT_ERROR {exc}')

    def read_serial_lines(self) -> None:
        if self._serial is None or not self._serial.is_open:
            return

        try:
            # Limit lines per tick so this function does not starve writes.
            for _ in range(20):
                if self._serial.in_waiting <= 0:
                    break
                raw = self._serial.readline()
                line = raw.decode('utf-8', errors='replace').strip()
                if line:
                    self.serial_rx_pub.publish(String(data=line))
        except serial.SerialException as exc:
            self.publish_status(f'SERIAL_READ_ERROR {exc}')
            self.close_serial()

    def write_command_line(self) -> None:
        if self._serial is None or not self._serial.is_open:
            return

        mode = str(self.get_parameter('command_mode').value).strip().lower()
        timeout_sec = float(self.get_parameter('command_timeout_sec').value)
        age_sec = (self.get_clock().now() - self.last_cmd_time).nanoseconds / 1e9

        # Priority order: safety stop -> timeout stop -> active command mode.
        if self.safety_stop_active:
            line = 'STOP'
        elif age_sec > timeout_sec:
            line = 'm 0 0 0 0' if mode == 'raw_wheels' else 'STOP'
        elif mode == 'da':
            line = f'DA {self.last_distance_in:.4f} {self.last_angle_rad:.4f}'
        else:
            fl, fr, rl, rr = self.last_wheel_cmd
            line = f'm {fl} {fr} {rl} {rr}'

        # Avoid spamming duplicate STOP lines.
        if line == self._last_tx_line and line == 'STOP':
            return

        self.write_line(line)
        self._last_tx_line = line

    def write_line(self, line: str) -> None:
        # Firmware expects one command per line.
        if self._serial is None or not self._serial.is_open:
            return
        try:
            self._serial.write((line + '\n').encode('utf-8'))
        except serial.SerialException as exc:
            self.publish_status(f'SERIAL_WRITE_ERROR {exc}')
            self.close_serial()

    def close_serial(self) -> None:
        # Safe close helper used on errors and shutdown.
        if self._serial is None:
            return
        try:
            if self._serial.is_open:
                self._serial.close()
        finally:
            self._serial = None

    def publish_status(self, text: str) -> None:
        self.serial_status_pub.publish(String(data=text))

    @staticmethod
    def _clamp_pwm(value: int) -> int:
        # Enforce expected PWM command range.
        return max(min(value, 255), -255)


def main(args=None) -> None:
    # Standard ROS 2 node startup/shutdown pattern.
    rclpy.init(args=args)
    node = SerialMotorBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close_serial()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
