#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, TwistStamped


class TwistToStamped(Node):
    def __init__(self):
        super().__init__('twist_to_stamped')

        # Parameters
        self.declare_parameter('input_topic', '/cmd_vel_nav')
        self.declare_parameter('output_topic', '/cmd_vel_nav_stamped')
        self.declare_parameter('frame_id', '')

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.frame_id = self.get_parameter('frame_id').value

        self.subscription = self.create_subscription(
            Twist,
            input_topic,
            self.twist_callback,
            10
        )

        self.publisher = self.create_publisher(
            TwistStamped,
            output_topic,
            10
        )

        self.get_logger().info(
            f'Converting {input_topic} (Twist) -> {output_topic} (TwistStamped)'
        )

    def twist_callback(self, msg: Twist):
        stamped = TwistStamped()

        stamped.header.stamp = self.get_clock().now().to_msg()
        stamped.header.frame_id = self.frame_id

        stamped.twist = msg

        self.publisher.publish(stamped)


def main(args=None):
    rclpy.init(args=args)

    node = TwistToStamped()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()