import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv_bridge
import cv2
import numpy as np
import sys
import yaml

from topic_names import IMAGE_TOPIC


with open("../config/navigate.yaml", "r") as f:
    navigate_config = yaml.safe_load(f)

FPS = navigate_config["fps"]


class CameraNode(Node):
    def __init__(self):
        super().__init__("camera_capture")

        self.create_timer(1/FPS, self.callback_timer)
        self.pub = self.create_publisher(Image, IMAGE_TOPIC, 10)
        self.bridge = cv_bridge.CvBridge()
        self.cap = cv2.VideoCapture(0)


    def callback_timer(self):
        success, image_bgr = self.cap.read()
        if not success:
            return
        
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_msg = self.bridge.cv2_to_imgmsg(image_rgb, encoding='rgb8')
        self.pub.publish(img_msg)
        self.get_logger().info('publish message.')


def main():
    rclpy.init()
    camera_node = CameraNode()

    rclpy.spin(camera_node)

    camera_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()