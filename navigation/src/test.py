from nomad.model import NoMaD
from utils import load_images
import yaml
import torch

import warnings
warnings.filterwarnings("ignore")

import threading


image_dir = "../maps/0/frames"

nomad = NoMaD(device="mps")

pil_images = load_images(image_dir)

waypoint = nomad.get_waypoint(pil_images[0], pil_images[1][3])
print(waypoint)

closest_node = nomad.get_closest_node(pil_images[0], pil_images[1])
print(closest_node)

distances = nomad.get_distances(pil_images[0], pil_images[1])
print(distances)

distances = nomad.get_distances(pil_images[0], pil_images[1][3])
print(distances)