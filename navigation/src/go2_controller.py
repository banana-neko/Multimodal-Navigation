import numpy as np
import yaml
from typing import Tuple
import argparse

# ROS
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Bool

from topic_names import (WAYPOINT_TOPIC, 
			 			REACHED_GOAL_TOPIC)
from ros_data import ROSData
from utils import clip_angle

import sys

# unitree sdk
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.go2.obstacles_avoid.obstacles_avoid_client import ObstaclesAvoidClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
import time


# CONSTS
CONFIG_PATH = "../config/robot.yaml"
with open(CONFIG_PATH, "r") as f:
	robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
# VEL_TOPIC = robot_config["vel_navi_topic"]
DT = 1/robot_config["frame_rate"]
RATE = robot_config["frame_rate"]
EPS = 1e-8
WAYPOINT_TIMEOUT = 1 # seconds # TODO: tune this
FLIP_ANG_VEL = np.pi/4

# GLOBALS
current_yaw = None
reached_goal = False

class Go2Controller(Node):
    def __init__(self, args):
        super().__init__("PD_CONTROLLER")
        self.waypoint = ROSData(WAYPOINT_TIMEOUT, name="waypoint")
        self.reached_goal = False
        self.reverse_mode = False
        
        if args.set_avoid:
            self.go2_client = ObstaclesAvoidClient()
            self.go2_client.SetTimeout(5.0)
            self.go2_client.Init()
            self.go2_client.SwitchSet(True)
            time.sleep(1)
            self.go2_client.UseRemoteCommandFromApi(True)
        else:
            self.go2_client = SportClient()
            self.go2_client.SetTimeout(5.0)
            self.go2_client.Init()

        self.waypoint_sub = self.create_subscription(Float32MultiArray, WAYPOINT_TOPIC, self.callback_drive, 10)
        self.reached_goal_sub = self.create_subscription(Bool, REACHED_GOAL_TOPIC, self.callback_reached_goal, 10)
        
        if args.rate is not None:
            rate = args.rate
        else:
            rate = RATE
        
        self.dt = 1/rate
        
        self.timer = self.create_timer(1/rate, self.callback_timer)

    def callback_timer(self):
        if self.reached_goal:
            reached_goal = True
            self.move_stop()
            print("Reached goal")
            self.destroy_node()
        
        elif self.waypoint.data is not None:
            v, w = self.pd_controller(self.waypoint.get())
            if self.reverse_mode:
                v *= -1
            
            v *= 1
            w *= 1
            print(f"command: (v:{v}, w:{w})")
            self.go2_client.Move(v, 0, w)
    
    def callback_drive(self, waypoint_msg: Float32MultiArray):
        """Callback function for the waypoint subscriber"""
        print("seting waypoint")
        self.waypoint.set(waypoint_msg.data)
	
    def callback_reached_goal(self, reached_goal_msg: Bool):
        """Callback function for the reached goal subscriber"""
        self.reached_goal = reached_goal_msg.data

    def clip_angle(self, theta) -> float:
        """Clip angle to [-pi, pi]"""
        theta %= 2 * np.pi
        if -np.pi < theta < np.pi:
            return theta
        return theta - 2 * np.pi

    def pd_controller(self, waypoint: np.ndarray) -> Tuple[float]:
        """PD controller for the robot"""
        assert len(waypoint) == 2 or len(waypoint) == 4, "waypoint must be a 2D or 4D vector"
        if len(waypoint) == 2:
            dx, dy = waypoint
        else:
            dx, dy, hx, hy = waypoint
        # this controller only uses the predicted heading if dx and dy near zero
        if len(waypoint) == 4 and np.abs(dx) < EPS and np.abs(dy) < EPS:
            v = 0
            w = self.clip_angle(np.arctan2(hy, hx))/self.dt
        elif np.abs(dx) < EPS:
            v =  0
            w = np.sign(dy) * np.pi/(2*self.dt)
        else:
            v = dx / self.dt
            w = np.arctan(dy/dx) / self.dt
        v = np.clip(v, 0, MAX_V)
        w = np.clip(w, -MAX_W, MAX_W)
        return v, w
    
    def move_stop(self):
        self.go2_client.Move(0, 0, 0)



def main(args):
    rclpy.init()
    ChannelFactoryInitialize(0, "enp14s0")
    go2_controller = Go2Controller(args)
    
    try:
        rclpy.spin(go2_controller)
    except Exception as e:
        print(e)
    finally:
        go2_controller.move_stop()
        if not reached_goal:
            go2_controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--set-avoid",
        action="store_true",
        help="Set obstacles avoid mode"
    )
    parser.add_argument(
        "--rate",
        "-r",
        default=None,
        type=float,
        help="robot action rate"
    )
    args = parser.parse_args()

    main(args)