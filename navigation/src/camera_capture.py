import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv_bridge
import cv2
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient
import numpy as np
import sys
from topic_names import IMAGE_TOPIC


class CameraNode(Node):
    def __init__(self):
        super().__init__("camera_capture")
        
        self.video_client = VideoClient()
        self.video_client.SetTimeout(3.0)
        self.video_client.Init()

        fps = 30
        self.create_timer(1/fps, self.callback_timer)
        self.pub = self.create_publisher(Image, IMAGE_TOPIC, 10)
        self.bridge = cv_bridge.CvBridge()

    def callback_timer(self):
        _, data = self.video_client.GetImageSample()
        image_data = np.frombuffer(bytes(data), dtype=np.uint8)
        image_bgr = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_msg = self.bridge.cv2_to_imgmsg(image_rgb, encoding='rgb8')
        self.pub.publish(img_msg)
        self.get_logger().info('publish message.')

def main():
    rclpy.init()
    ChannelFactoryInitialize(0, "eth0")
    camera_node = CameraNode()
    rclpy.spin(camera_node)
    camera_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()