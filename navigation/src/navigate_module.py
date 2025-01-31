import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import UInt32

import yaml
import os
from PIL import Image as PILImage
from queue import Queue
from typing import List

from nomad.model import NoMaD
from utils import msg_to_pil
from const import IMAGE_TOPIC, GOAL_TOPIC

import time


class Navigation:
    def __init__(self, map_path, device="cpu"):
        self.map_path = map_path # ../maps/0
        self.map = self._load_map()
        self.graph = self._load_graph()
        self.nodes = self._load_nodes()

        self.nomad = NoMaD(device=device)


    def navigate(self, goal_node):
        current_node = self.get_current_node()

        print(f"# current_node: {current_node}")
        print(f"# goal_node: {goal_node}")

        if current_node == goal_node:
            print(f"[*] Already reached [{goal_node}]")
            return True
        
        reach = self.check_reach(current_node, goal_node)

        if not reach:
            print(f"[*] Unreachable ({current_node} -> {goal_node})")
            return False
        
        shortest_path = self.dijkstra(current_node, goal_node)
        sub_goal_images = [self.map[node][3] for node in shortest_path]
        sub_goal_idx = 1

        print(f"# shortest_path: {shortest_path}")

        i = 0
        while sub_goal_idx != len(shortest_path):
            waypoint = self.get_waypoint(shortest_path[sub_goal_idx])
            print(f"# ({i}) waypoint: {waypoint}")
            i += 1

            goals = []
            current_idx = self.nomad.get_closest_node(self.obs_buffer, sub_goal_images)
            sub_goal_idx = current_idx + 1

            print(f"# current_idx: {current_idx}, current_node: {shortest_path[current_idx]}")
        
        return True


    def get_waypoint(self, goal_node):
        obs_images = self.obs_buffer
        goal_image = self.map[goal_node][3]
        obsgoal_cond = self.nomad.get_obsgoal_cond(obs_images, goal_image)
        waypoint = self.nomad.get_waypoint(obsgoal_cond)

        return waypoint


    def get_current_node(self):
        obs_images = self.obs_buffer
        goals = [node_imgs[3] for node_imgs in self.map]
        closest_node = self.nomad.get_closest_node(obs_images=obs_images, goal_images=goals)

        return closest_node


    def check_reach(self, start, goal):
        def rec(start, goal, nodes, visited):
            if start in visited:
                return False
            visited.append(start)

            if goal in nodes[start]: #接続先にゴールがある
                return True
            if not nodes[start]: #終端ノード
                return False
            
            result = False
            for con_node in nodes[start]:
                if rec(con_node, goal, nodes, visited):
                    result = True
            
            return result

        return rec(start, goal, self.nodes, visited=[])
    

    def dijkstra(self, start, goal):
        # startからgoalまでの最短経路を返す
        return [0, 1, 2, 3, 6, 7]


    def _load_map(self):
        frames_path = os.path.join(self.map_path, "frames")
        node_num = len(os.listdir(frames_path))

        map = []

        for node in range(node_num):
            images = []
            for i in range(4):
                image = os.path.join(frames_path, f"{node}/{i}.jpg")
                image = PILImage.open(image)
                images.append(image)

            map.append(images)
        
        return map
    

    def _load_graph(self):
        with open(os.path.join(self.map_path, "graph", "graph.yaml"), "r") as f:
            graph = yaml.safe_load(f)
        
        return graph


    def _load_nodes(self):
        with open(os.path.join(self.map_path, "graph", "nodes.yaml"), "r") as f:
            nodes_dict = yaml.safe_load(f)
        
        nodes = []
        for node in range(len(nodes_dict)):
            value = nodes_dict[node]

            if value is None:
                value = []
            elif type(value) != list:
                value = [value]
            
            nodes.append(value)
        
        return nodes