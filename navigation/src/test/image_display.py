import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv_bridge
import cv2
import numpy as np
import sys
from topic_names import IMAGE_TOPIC


class CameraNode(Node):
    def __init__(self):
        super().__init__("image_display")

        fps = 30
        self.create_timer(1/fps, self.callback_timer)
        self.pub = self.create_publisher(Image, IMAGE_TOPIC, 10)
        self.sub = self.create_subscription(Image, IMAGE_TOPIC, self.callback_obs, 10)
        self.bridge = cv_bridge.CvBridge()
        self.cap = cv2.VideoCapture(0)


    def callback_obs(self, msg: Image):
        image_array = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, -1
        )
        rgb_img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        cv2.imshow("camera", bgr_img)


    def callback_timer(self):
        success, image_bgr = self.cap.read()
        if not success:
            return
        
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_msg = self.bridge.cv2_to_imgmsg(image_rgb, encoding='rgb8')
        self.pub.publish(img_msg)
        self.get_logger().info('publish message.')


def msg_to_pil(msg: Image) -> PILImage.Image:
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
        msg.height, msg.width, -1
    )
    pil_image = PILImage.fromarray(img)

    return pil_image


def main():
    rclpy.init()
    camera_node = CameraNode()

    rclpy.spin(camera_node)

    camera_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()