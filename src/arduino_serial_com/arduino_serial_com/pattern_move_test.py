"""Sequenced strafe test: publish [distance_m, heading_rad] on /bean_bag_trajectory (same as move_test).

Fixed wall-clock program: 5s left @ medium, 1s right @ medium, 2s left @ max, 1s right @ slow.
+π/2 = strafe left (+y body); -π/2 = strafe right. Distance controls 1s-horizon speed; tune via params
to match omni_pwm_subscribe --speed-fraction caps.
"""

import math
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

# Republish often so each motion phase is held for the full duration (not one-shot).
TIMER_S = 0.02  # 50 Hz


class PatternMoveTestPublisher(Node):
    def __init__(self) -> None:
        super().__init__("pattern_move_test_publisher")
        self.declare_parameter("result_topic", "/bean_bag_trajectory")
        self.declare_parameter("dist_slow_m", 0.04)
        self.declare_parameter("dist_medium_m", 0.08)
        self.declare_parameter("dist_max_m", 0.10)

        topic = self.get_parameter("result_topic").get_parameter_value().string_value
        d_slow = self.get_parameter("dist_slow_m").get_parameter_value().double_value
        d_med = self.get_parameter("dist_medium_m").get_parameter_value().double_value
        d_max = self.get_parameter("dist_max_m").get_parameter_value().double_value

        self.publisher = self.create_publisher(Float32MultiArray, topic, 10)

        # (duration_s, heading_rad, distance_m, dir_name, speed_label) — 4/4
        x = math.pi / 2.0
        self._phases: list = [
            (5.0, x, d_med, "LEFT", "medium"),
            (1.0, -x, d_med, "RIGHT", "medium"),
            (2.0, x, d_max, "LEFT", "max"),
            (1.0, -x, d_slow, "RIGHT", "slow"),
        ]

        self._sequence_start: float | None = None
        self._phase_start: float | None = None
        self._phase_index: int = 0
        self._idle: bool = False
        self.timer = self.create_timer(TIMER_S, self._timer_callback)

    def _publish(self, dist: float, heading: float) -> None:
        msg = Float32MultiArray()
        msg.data = [float(dist), float(heading)]
        self.publisher.publish(msg)

    def _log_phase(self, i: int) -> None:
        duration, heading, dist, dname, st = self._phases[i]
        self.get_logger().info(
            f"Phase {i + 1}/4: {duration:.1f}s {dname} strafe, {st} — "
            f"Published: Range={dist}, Radian={heading:.4f}"
        )

    def _timer_callback(self) -> None:
        now = time.monotonic()
        if self._sequence_start is None:
            self._sequence_start = now
            self._phase_start = now
            self._log_phase(0)

        if self._idle:
            self._publish(0.0, 0.0)
            return

        # Catch up if the timer was delayed (e.g. jump multiple phase boundaries)
        while self._phase_index < len(self._phases):
            duration, _, _, _, _ = self._phases[self._phase_index]
            if self._phase_start is not None and now - self._phase_start < duration - 1e-9:
                break
            self._phase_index += 1
            self._phase_start = now
            if self._phase_index >= len(self._phases):
                self._idle = True
                self.get_logger().info("Test sequence complete (holding [0, 0] stop).")
                self._publish(0.0, 0.0)
                return
            self._log_phase(self._phase_index)

        if self._idle:
            self._publish(0.0, 0.0)
            return

        _, h, d, _, _ = self._phases[self._phase_index]
        self._publish(d, h)


def main(args: object | None = None) -> None:
    rclpy.init(args=args)
    node = PatternMoveTestPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
