import yaml
import os
from PIL import Image as PILImage
from pathlib import Path
import argparse

from nomad.model import NoMaD

import warnings
warnings.filterwarnings("ignore")


def load_images(frames_path: str) -> dict:
    images = {}
    nodes = os.listdir(frames_path)

    for node in nodes:
        node_images = []
        for i in range(4):
            image = PILImage.open(os.path.join(frames_path, node, f"{i}.jpg"))
            node_images.append(image)    
        images[int(node)] = node_images
    
    return images


def load_nodes(nodes_path: str) -> dict:
    with open(nodes_path, "r") as f:
        nodes = yaml.safe_load(f)
    
    new_nodes = {}
    for node, con_nodes in nodes.items():
        if type(con_nodes) != list:
            con_nodes = [con_nodes]
        new_nodes[node] = con_nodes
    
    return new_nodes


def create_graph(nodes: dict, images: dict, nomad: NoMaD) -> dict:
    graph = {}
    for node, con_nodes in nodes.items():
        dists = {}
        for con_node in con_nodes:
            if con_node is None:
                dists = None
                continue

            obsgoal_cond = nomad.get_obsgoal_cond(images[node], images[con_node][3])
            dist = nomad.get_distances(obsgoal_cond)
            dists[con_node] = float(dist[0])
        graph[node] = dists
    
    return graph


def main(args):
    maps_dir = Path(__file__).parent.parent

    map_id = args.map_id
    map_dir = os.path.join(maps_dir, map_id)

    print(f"[*] map_id: {map_id}")

    nodes = load_nodes(os.path.join(map_dir, "graph", "nodes.yaml")) # {0: [1], 1: [2], 3: [4], ..., 7: [None]}
    images = load_images(os.path.join(map_dir, "frames")) # {0: [Image(0.jpg),...], 1: [Image(0.jpg),...], ...}

    nomad = NoMaD(device="mps")
    graph = create_graph(nodes, images, nomad)

    with open(os.path.join(map_dir, "graph", "graph.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(graph, f)

    print("[*] Save completed.")
    print(f"[*] graph = {graph}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--map-id", type=str, help="対象のマップID")
    args = parser.parse_args()

    main(args)