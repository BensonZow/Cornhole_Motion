# pyright: reportMissingImports=false
# This node turns raw firmware text lines into structured ROS topics.
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Int32MultiArray, String


class MotorTelemetryNode(Node):
    """Parses firmware serial lines into structured telemetry topics."""

    def __init__(self) -> None:
        super().__init__('motor_telemetry_node')

        # Topic names are configurable so this parser can fit different systems.
        self.declare_parameter('serial_rx_topic', '/motion/serial_rx')
        self.declare_parameter('motor_feedback_topic', '/motion/motor_feedback')
        self.declare_parameter('firmware_event_topic', '/motion/firmware_event')
        self.declare_parameter('safety_state_topic', '/motion/safety_state')

        serial_rx_topic = self.get_parameter('serial_rx_topic').value
        motor_feedback_topic = self.get_parameter('motor_feedback_topic').value
        firmware_event_topic = self.get_parameter('firmware_event_topic').value
        safety_state_topic = self.get_parameter('safety_state_topic').value

        # Input: raw serial lines from bridge node.
        self.create_subscription(String, serial_rx_topic, self.serial_rx_callback, 10)
        # Outputs: parsed encoder counts, generic firmware lines, and safety state.
        self.motor_feedback_pub = self.create_publisher(
            Int32MultiArray, motor_feedback_topic, 10
        )
        self.firmware_event_pub = self.create_publisher(
            String, firmware_event_topic, 10
        )
        self.safety_state_pub = self.create_publisher(Bool, safety_state_topic, 10)

        self.get_logger().info('Motor telemetry parser ready')

    def serial_rx_callback(self, msg: String) -> None:
        # Normalize whitespace and skip empty serial lines.
        line = msg.data.strip()
        if not line:
            return

        # PG line format carries 4 wheel feedback values.
        if line.startswith('PG '):
            self.publish_pg_counts(line)
            return

        # Map known safety message into a dedicated boolean topic.
        if line.startswith('WATCHDOG STOP'):
            self.safety_state_pub.publish(Bool(data=True))

        # Keep a generic event stream for logs and future parser extensions.
        self.firmware_event_pub.publish(String(data=line))

    def publish_pg_counts(self, line: str) -> None:
        # Expected format: "PG <fl> <fr> <rl> <rr>"
        parts = line.split()
        if len(parts) != 5:
            self.firmware_event_pub.publish(String(data=f'PARSE_ERR {line}'))
            return
        try:
            fl = int(parts[1])
            fr = int(parts[2])
            rl = int(parts[3])
            rr = int(parts[4])
        except ValueError:
            self.firmware_event_pub.publish(String(data=f'PARSE_ERR {line}'))
            return

        # Publish parsed wheel values in [FL, FR, RL, RR] order.
        self.motor_feedback_pub.publish(Int32MultiArray(data=[fl, fr, rl, rr]))


def main(args=None) -> None:
    # Standard ROS 2 node startup/shutdown pattern.
    rclpy.init(args=args)
    node = MotorTelemetryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
