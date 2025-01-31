import numpy as np
import yaml

# ROS
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Bool

from topic_names import WAYPOINT_TOPIC, REACHED_GOAL_TOPIC


with open("../config/navigate.yaml", "r") as f:
    navigate_config = yaml.safe_load(f)

MAX_V = navigate_config["max_v"]
MAX_W = navigate_config["max_w"]
RATE = navigate_config["action_rate"]
AUTO_AVOID = navigate_config["auto_avoid"]


class Go2ControlNode(Node):
    def __init__(self):
        super().__init__("go2_controller")
        
        if AUTO_AVOID:
            """
            self.go2_client = ObstaclesAvoidClient()
            self.go2_client.SetTimeout(5.0)
            self.go2_client.Init()
            self.go2_client.SwitchSet(True)
            time.sleep(1)
            self.go2_client.UseRemoteCommandFromApi(True)
            """
            print("[*] Auto Avoid Mode")
        else:
            """
            self.go2_client = SportClient()
            self.go2_client.SetTimeout(5.0)
            self.go2_client.Init()
            """
            print("[*] Manual Mode")

        self.waypoint_sub = self.create_subscription(Float32MultiArray, WAYPOINT_TOPIC, self.callback_waypoint, 10)
        self.reached_goal_sub = self.create_subscription(Bool, REACHED_GOAL_TOPIC, self.callback_reached_goal, 10)

        self.reached_goal = False
        self.dt = 1/RATE


    def callback_waypoint(self, msg: Float32MultiArray):
        if self.reached_goal:
            # self.go2_client.Move(0, 0, 0)
            print("[*] Move(0, 0, 0)")
            return

        waypoint = msg.data
        v, w = waypoint

        if np.abs(v) < 1e-8:
            v = 0
            w = np.sign(w) * np.pi/(2*self.dt)
        else:
            v = v / self.dt
            w = np.arctan(w/v) / self.dt

        v = np.clip(v, 0, MAX_V)
        w = np.clip(w, -MAX_W, MAX_W)

        # self.go2_client.Move(v, 0, w)
        print(f"[*] Move({v}, 0, {w})")

	
    def callback_reached_goal(self, reached_goal_msg: Bool):
        print("[*] reached_goal")
        self.reached_goal = reached_goal_msg.data


def main():
    rclpy.init()
    go2_control_node = Go2ControlNode()
    
    rclpy.spin(go2_control_node)

    go2_control_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()