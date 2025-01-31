from nomad.model import NoMaD
from utils import load_images
import warnings
warnings.filterwarnings("ignore")

image_dir = "../maps/0/frames"

nomad = NoMaD(device="mps")
node_images = load_images(image_dir)
start = node_images[0]
print(len(start))
goal = node_images[1][3]

obsgoal_cond = nomad.get_obsgoal_cond(start, goal)
print(obsgoal_cond.shape)