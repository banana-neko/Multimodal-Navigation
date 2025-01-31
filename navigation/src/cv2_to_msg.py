import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv_bridge
import cv2
import sys


IMAGE_TOPIC = "/camera_image"


class ImgPublisher(Node):
    def __init__(self):
        super().__init__("camera_capture")

        self.cap = cv2.VideoCapture(0)
        fps = 30
        self.timer = self.create_timer(1/fps, self.timer_callback)
        
        self.pub = self.create_publisher(Image, IMAGE_TOPIC, 10)
        self.bridge = cv_bridge.CvBridge()

    def timer_callback(self):
        _, frame_bgr = self.cap.read()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_msg = self.bridge.cv2_to_imgmsg(frame_rgb, encoding="rgb8")
        self.pub.publish(img_msg)
        
        log = "[*] image publish..."
        sys.stdout.write(f"\r\033[2K{log}")
        sys.stdout.flush()
    
    def release_cap(self):
        self.cap.release()


def main():
    rclpy.init()
    img_node = ImgPublisher()
    rclpy.spin(img_node)
    img_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()