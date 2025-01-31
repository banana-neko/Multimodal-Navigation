from camera_capture import CameraNode
from go2_controller import Go2ControlNode

import rclpy

def main():
    rclpy.init()

    camera_node = CameraNode()
    control_node = Go2ControlNode(rate=15, set_avoid=False)

    rclpy.spin(camera_node)
    rclpy.spin(control_node)

    camera_node.destroy_node()
    control_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()