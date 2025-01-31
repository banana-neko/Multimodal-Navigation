import yaml
import os
from typing import List, Union
from PIL import Image as PILImage
import numpy as np

import torch
from torchvision import transforms
import torchvision.transforms.functional as TF

from nomad.models.nomad.nomad import NoMaD as NoMaD_Model
from nomad.models.nomad.nomad import DenseNetwork
from nomad.models.nomad.nomad_vint import NoMaD_ViNT
from nomad.models.diffusion_policy.conditional_unet1d import ConditionalUnet1D
from diffusers import DDPMScheduler


current_dir = os.path.dirname(__file__)
with open(os.path.join(current_dir, "config", "config.yaml"), "r") as f:
    CONFIG_DICT = yaml.safe_load(f)

MODEL_WEIGHTS_PATH = os.path.join(current_dir, "model_weights", "nomad.pth")


class NoMaD:
    def __init__(self, device: str="cpu"):
        self.config = CONFIG_DICT
        self.model_weights_path = MODEL_WEIGHTS_PATH
        self.device = device

        self.model = self._load_model()
        self.model.eval()
        self.scheduler = self._load_scheduler()

        self.context_size = self.config["context_size"]


    def get_closest_idx(self, obs_images: List[PILImage.Image], goal_images: List[PILImage.Image]) -> int:
        obsgoal_cond = self._get_obsgoal_cond(obs_images=obs_images, goal_images=goal_images)
        dists = self._get_distances_cond(obsgoal_cond)
        closest_node = int(np.argmin(dists))

        return closest_node


    def get_waypoint(self, obs_images: List[PILImage.Image], goal_image: PILImage.Image, num_samples: int=8, waypoint :int=2) -> List[float]:
        obs_cond = self._get_obsgoal_cond(obs_images=obs_images, goal_images=goal_image)
        waypoint = self._get_waypoint_cond(obs_cond=obs_cond, num_samples=num_samples, waypoint=waypoint)

        return waypoint
    

    def get_distances(self, obs_images: List[PILImage.Image], goal_images: List[PILImage.Image]) -> List[float]:
        obsgoal_cond = self._get_obsgoal_cond(obs_images=obs_images, goal_images=goal_images)
        distances = self._get_distances_cond(obsgoal_cond=obsgoal_cond)

        return distances.tolist()

    
    def _get_obsgoal_cond(self, obs_images: List[PILImage.Image], goal_images: Union[PILImage.Image, List[PILImage.Image]]) -> torch.Tensor:
        obs_tensor = self._transform_images(obs_images, self.config["image_size"], center_crop=False).to(self.device) # [1, 3xN, 96, 96]
        mask = torch.zeros(1).long().to(self.device) # [0]

        if type(goal_images) != list:
            goal_images = [goal_images]
        goal_tensor = [self._transform_images(goal_image, self.config["image_size"], center_crop=False) for goal_image in goal_images] # [[3, 96, 96] x M]
        goal_tensor = torch.concat(goal_tensor, dim=0).to(self.device) # [M, 3, 96, 96]

        obsgoal_cond = self.model(
            "vision_encoder",
            obs_img = obs_tensor.repeat(len(goal_tensor), 1, 1, 1), # [M, 3xN, 96, 96]
            goal_img = goal_tensor, # [M, 3, 96, 96]
            input_goal_mask = mask.repeat(len(goal_tensor)) # [0 x M]
        )

        return obsgoal_cond
    

    def _get_distances_cond(self, obsgoal_cond: torch.Tensor) -> np.ndarray:
        dists = self.model("dist_pred_net", obsgoal_cond=obsgoal_cond)
        dists = self._to_numpy(dists).flatten()

        return dists
    

    def _get_waypoint_cond(self, obs_cond: torch.Tensor, num_samples: int, waypoint: int) -> List[float]:
        with torch.no_grad():
            obs_cond = obs_cond.repeat(num_samples, 1) # [num_samples, 256]

            # initialize action from Gaussian noise
            # [num_samples, 8, 2]
            noisy_action = torch.randn(
                (num_samples, self.config["len_traj_pred"], 2), # len_traj_pred: 予測するウェイポイントの個数。デフォルトでは8通りのウェイポイントを予測。
                device = self.device
            )

            # init scheduler
            self.scheduler.set_timesteps(self.config["num_diffusion_iters"]) # self.scheduler.timesteps: tensor([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])

            for k in self.scheduler.timesteps[:]:
                # predict noise
                noise_pred = self.model(
                    "noise_pred_net",
                    sample = noisy_action,
                    timestep = k,
                    global_cond = obs_cond
                )
                # inverse diffusion step (remove noise)
                noisy_action = self.scheduler.step(
                    model_output = noise_pred,
                    timestep = k,
                    sample = noisy_action
                ).prev_sample

        pred_action = self._get_action(noisy_action)
        pred_action = pred_action[0]
        waypoint = pred_action[waypoint]
        if self.config["normalize"]:
            max_v = self.config["max_v"]
            rate = self.config["frame_rate"]
            waypoint[:2] *= (max_v / rate)
        
        return waypoint.tolist()


    def _load_model(self) -> NoMaD_Model:
        config = self.config
        device = self.device

        vision_encoder = NoMaD_ViNT(
            obs_encoding_size=config["encoding_size"],
            context_size=config["context_size"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"]
        )

        noise_pred_net = ConditionalUnet1D(
            input_dim=2,
            global_cond_dim=config["encoding_size"],
            down_dims=config["down_dims"],
            cond_predict_scale=config["cond_predict_scale"],
        )

        dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])

        model = NoMaD_Model(
            vision_encoder=vision_encoder,
            noise_pred_net=noise_pred_net,
            dist_pred_net=dist_pred_network
        )

        state_dict = torch.load(self.model_weights_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        print(f"[*] model loaded. (device: {device})")

        return model


    def _load_scheduler(self) -> DDPMScheduler:
        config = self.config
        
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=config["num_diffusion_iters"],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )

        return noise_scheduler
    

    def _transform_images(self, pil_imgs: List[PILImage.Image], image_size: List[int], center_crop: bool=False) -> torch.Tensor:
        IMAGE_ASPECT_RATIO = (
            4 / 3
        )  # all images are centered cropped to a 4:3 aspect ratio in training
        
        """Transforms a list of PIL image to a torch tensor."""

        transform_type = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                        0.229, 0.224, 0.225]),
            ]
        )
        if type(pil_imgs) != list:
            pil_imgs = [pil_imgs]
        transf_imgs = []
        for pil_img in pil_imgs:
            w, h = pil_img.size
            if center_crop:
                if w > h:
                    pil_img = TF.center_crop(pil_img, (h, int(h * IMAGE_ASPECT_RATIO)))  # crop to the right ratio
                else:
                    pil_img = TF.center_crop(pil_img, (int(w / IMAGE_ASPECT_RATIO), w))
            pil_img = pil_img.resize(image_size) 
            transf_img = transform_type(pil_img)
            transf_img = torch.unsqueeze(transf_img, 0)
            transf_imgs.append(transf_img)

        return torch.concat(transf_imgs, dim=1)
    

    def _get_action(self, noisy_action: torch.Tensor) -> np.ndarray:
        # ndeltas = noisy_action.reshape(noisy_action.shape[0], -1, 2)
        ndeltas = self._to_numpy(noisy_action)
        ndeltas = self._unnomalize_data(ndeltas)
        actions = np.cumsum(ndeltas, axis=1)

        return actions


    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        return tensor.cpu().detach().numpy()
    

    def _from_numpy(self, array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(array).float()


    def _unnomalize_data(self, ndata: np.ndarray) -> np.ndarray:
        stats_min = np.array([-2.5, -4])
        stats_max = np.array([5, 4])
        ndata = (ndata + 1) / 2
        data = ndata * (stats_max - stats_min) + stats_min
        
        return data