from nomad.model import NoMaD
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from std_msgs.msg import UInt32, Bool, Float32MultiArray
from PIL import Image as PILImage

import yaml
import os
import time

import warnings
warnings.filterwarnings("ignore")

from utils import msg_to_pil
from topic_names import IMAGE_TOPIC, GOAL_TOPIC, WAYPOINT_TOPIC, REACHED_GOAL_TOPIC


with open("./config/navigate.yaml", "r") as f:
    navigate_config = yaml.safe_load(f)

MAP_PATH = navigate_config["map_path"]
NAV_RATE = navigate_config["nav_rate"]


class NavigationNode(Node):
    def __init__(self, device="cpu"):
        super().__init__("navigation")

        self.nomad = NoMaD(device=device)
        self.context_size = self.nomad.context_size

        # マップの読み込み
        self.map_path = MAP_PATH
        self.topomap = self.load_topomap() # トポマップの読み込み

        self.sub_goal_images = None
        self.obs_buffer = []
        
        # ROSのセットアップ
        self.obs_sub = self.create_subscription(Image, IMAGE_TOPIC, self.callback_obs, 10)
        self.goal_sub = self.create_subscription(UInt32, GOAL_TOPIC, self.callback_goal, 10)
        self.waypoint_pub = self.create_publisher(Float32MultiArray, WAYPOINT_TOPIC, 10)
        self.reached_goal_pub = self.create_publisher(Bool, REACHED_GOAL_TOPIC, 10)

        rate = NAV_RATE
        self.timer = self.create_timer(1/rate, self.callback_timer)


    def callback_timer(self):
        start = time.time()

        if self.sub_goal_images is None:
            # self.get_logger().info("sub_goal_images is None.")
            return
        
        current_idx = self.nomad.get_closest_idx(self.obs_buffer, self.sub_goal_images) # サブゴール上での現在ノードの推定
        print(f"[*] current_idx: {current_idx}")

        # ゴール時の処理
        if current_idx == len(self.sub_goal_images) - 1:
            self.get_logger().info("Goal Reached.")
            reached_goal = Bool(data=True)
            self.reached_goal_pub.publish(reached_goal)
            self.sub_goal_images = None
            return

        sub_goal_image = self.sub_goal_images[current_idx + 1]
        waypoint = self.nomad.get_waypoint(self.obs_buffer, sub_goal_image)
        waypoint_msg = Float32MultiArray(data=waypoint)
        self.waypoint_pub.publish(waypoint_msg)

        end = time.time()
        self.get_logger().info(f"waypoint_inf: {end - start} sec")
        #print(f"waypoint_inf: {end - start} sec")


    def callback_obs(self, msg: Image):
        pil_image = msg_to_pil(msg)
        # obs_bufferの画像数をcontext_size+1枚に固定
        if len(self.obs_buffer) < self.context_size + 1:
            self.obs_buffer.append(pil_image)
        else:
            self.obs_buffer.pop(0)
            self.obs_buffer.append(pil_image)

    
    def callback_goal(self, msg: UInt32):
        goal_node = msg.data
        print(f"[*] goal_node: {goal_node}")
        
        current_node = self.nomad.get_closest_idx(self.obs_buffer, self.topomap) # 現在地の取得
        nodes = self.load_nodes()
        reachable = self.check_reach(current_node, goal_node, nodes) # 到達可能かどうか

        print(f"[*] current_node: {current_node}, reachable: {reachable}")

        if reachable:
            graph = self.load_graph()
            goal_path = self.dijkstra(current_node, goal_node, graph) # 最短経路の取得
            self.sub_goal_images = self.get_subgoals(goal_path, self.topomap) # サブゴール画像の取得
        else:
            print(f"[*] Node({goal_node}) is unreachable.")


    def check_reach(self, start, goal, nodes):
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

        return rec(start, goal, nodes, visited=[])
    

    def dijkstra(self, start, goal, graph):
        # startからgoalまでの最短経路を返す
        return [0, 1, 2, 3, 6, 7]


    def load_topomap(self):
        frames_path = os.path.join(self.map_path, "frames")
        node_num = len(os.listdir(frames_path))

        topomap = []

        for node in range(node_num):
            node_image = PILImage.open(os.path.join(frames_path, f"{node}/3.jpg"))
            topomap.append(node_image)
        
        return topomap # [Node(0), Node(1), Node(2), ...]
    

    def load_graph(self):
        with open(os.path.join(self.map_path, "graph", "graph.yaml"), "r") as f:
            graph_dict = yaml.safe_load(f)
        
        graph = []
        for node in range(len(graph_dict)):
            graph.append(graph_dict[node])

        return graph # [{1: 4.232683181762695}, {2: }, ..., None]


    def load_nodes(self):
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
        
        return nodes # [[1], [2], [3,4,5], ..., []]
    

    def get_subgoals(self, goal_path, topomap):
        sub_goals = []

        for node in goal_path:
            sub_goal = topomap[node]
            sub_goals.append(sub_goal)
        
        return sub_goals


def main():
    rclpy.init()
    nav_node = NavigationNode(device="mps")
    
    rclpy.spin(nav_node)

    nav_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()