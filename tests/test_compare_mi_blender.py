import mitsuba as mi
import numpy as np
import math
import mitsuba_tools as mit
import drjit as dr
import matplotlib.pyplot as plt
import torchvision
import torch
import time
import json
from normal import get_normal_image
from PIL import Image
from pathlib import Path
import copy

mi.set_variant("cuda_ad_rgb")
PILToTensor = torchvision.transforms.PILToTensor()
ToPILImage = torchvision.transforms.ToPILImage()

def main():
    ply_path = "../scenes/lego.obj"
    scene = mit.create_mitsuba_scene(ply_path)

    for i in range(100):
        name = f"test_{i:03d}"
        save_compared_result(name, scene)


def save_compared_result(name, scene):
    metadata_path = f"../scenes/lego_sunset_low/{name}/metadata.json"
    """read information"""
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    cam_transform_mat = metadata['cam_transform_mat']
    cam_angle_x = metadata['cam_angle_x']
    imw = metadata['imw']
    imh = metadata['imh']

    """compute mi"""
    normal_mi_raw, mask_mi_raw = get_normal_image(cam_angle_x, cam_transform_mat, imh, imw, scene, spp=512)
    normal_mi_raw = PILToTensor(normal_mi_raw)  # [3,512,512]
    mask_mi_raw = PILToTensor(mask_mi_raw)  # [1, 512, 512]

    """load bl image"""
    normal_bl_path = f"../scenes/lego_sunset_low/{name}/Normal_0001.png"
    rgb_gt_path = f"../scenes/lego_sunset_low/{name}/GlossIndCol_0001.png"
    normal_bl_raw = Image.open(normal_bl_path).convert("RGB")
    rgb_gt_raw = Image.open(rgb_gt_path).convert("RGB")
    normal_bl_raw = PILToTensor(normal_bl_raw)  # [3, 512, 512]
    rgb_gt_raw = PILToTensor(rgb_gt_raw)  # [3, 512, 512]

    """normal"""
    normal_mi = normal_mi_raw
    normal_bl = normal_bl_raw
    # tone mapping
    normal_mi = ((normal_mi.float()/255)**(1/2.2) * 255).byte()

    """mask"""
    mask_mi_bool = mask_mi_raw > 0  # [1, 512, 512]
    rgb_gt_raw_1d, _ = torch.max(rgb_gt_raw, axis=0, keepdim=True)
    mask_bl_bool = rgb_gt_raw_1d > 0  # [1, 512, 512]
    zero1 = torch.zeros(1, imh, imw, dtype=torch.uint8)
    mask_mi = torch.concat([mask_mi_bool.byte()*255, zero1, zero1], dim=0)  # Red
    mask_bl = torch.concat([zero1, zero1, mask_bl_bool.byte()*255])  # Blue

    """sum color"""
    sum_color = mask_mi + mask_bl

    """diff_color"""
    mask_eq_bool = torch.eq(mask_mi_bool, mask_bl_bool)  # [1, 512, 512]
    mask_eq_bool = torch.repeat_interleave(mask_eq_bool, 3, dim=0)
    diff_color = copy.deepcopy(sum_color)  # [3, 512, 512]
    diff_color[mask_eq_bool] = 0

    """rgb_gt"""
    rgb_gt = rgb_gt_raw

    """diff_color_gt"""
    diff_color_gt = copy.deepcopy(rgb_gt)
    diff_color_gt[mask_eq_bool] = 0

    ToPILImage = torchvision.transforms.ToPILImage()
    output1 = [normal_mi, mask_mi, sum_color, rgb_gt]
    output2 = [normal_bl, mask_bl, diff_color, diff_color_gt]


    output1 = torch.cat(output1, dim=2)
    output2 = torch.cat(output2, dim=2)
    output = torch.cat([output1, output2], dim=1)
    output = ToPILImage(output)
    output_root = "../tests/compare_mi_blender/lego"
    Path(output_root).mkdir(parents=True, exist_ok=True)
    output.save(Path(output_root, f"{name}.png"))

if __name__ == '__main__':
    main()
