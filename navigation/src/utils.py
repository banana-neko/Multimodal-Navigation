import matplotlib.pyplot as plt
import os
from typing import List, Tuple, Sequence, Dict, Union, Optional, Callable

import numpy as np
from sensor_msgs.msg import Image
from PIL import Image as PILImage

import matplotlib.pyplot as plt
import yaml


def load_images(frames_path: str) -> dict:
    images = []
    for node in range(len(os.listdir(frames_path))):
        node_images = []
        
        for i in range(4):
            image = PILImage.open(os.path.join(frames_path, f"{node}/{i}.jpg"))
            node_images.append(image)
        
        images.append(node_images)
    
    return images


def load_nodes(nodes_path: str) -> dict:
    with open(nodes_path, "r") as f:
        nodes = yaml.safe_load(f)
    
    new_nodes = {}
    for node, con_nodes in nodes.items():
        if not con_nodes:
            con_nodes = []
        if type(con_nodes) != list:
            con_nodes = [con_nodes]
            
        new_nodes[node] = con_nodes
    
    return new_nodes


def load_graph(graph_path: str) -> dict:
    with open(graph_path, "r") as f:
        graph = yaml.safe_load(f)
    
    return graph


def msg_to_pil(msg: Image) -> PILImage.Image:
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
        msg.height, msg.width, -1
    )
    pil_image = PILImage.fromarray(img)

    return pil_image