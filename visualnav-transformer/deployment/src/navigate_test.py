import matplotlib.pyplot as plt
import os
from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import numpy as np
import torch
import torch.nn as nn
from diffusers import DDPMScheduler
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

import matplotlib.pyplot as plt
import yaml

from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT

def load_nomad(model_path="../model_weights/nomad.pth", device="mps"):
    vision_encoder = NoMaD_ViNT(
        obs_encoding_size=256,
        context_size=3,
        mha_num_attention_heads=4,
        mha_num_attention_layers=4,
        mha_ff_dim_factor=4
    )

    noise_pred_net = ConditionalUnet1D(
        input_dim=2,
        global_cond_dim=256,
        down_dims=[64, 128, 256],
        cond_predict_scale=False,
    )

    dist_pred_network = DenseNetwork(embedding_dim=256)

    model = NoMaD(
        vision_encoder=vision_encoder,
        noise_pred_net=noise_pred_net,
        dist_pred_net=dist_pred_network
    )

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    print(f"[*] NoMaD loaded.(device: {device})")

    return model

model = load_nomad()