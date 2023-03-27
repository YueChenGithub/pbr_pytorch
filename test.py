import mitsuba as mi
import drjit as dr
import mitsuba_tools as mit
import torchvision
import torch
from tone_mapping import show_image, save_image
import json
import math
from path_tool import get_ply_path, get_light_probe_path, get_light_inten
import matplotlib.pyplot as plt

mi.set_variant("cuda_ad_rgb")


def create_mitsuba_scene_envmap(ply_path, envmap_path, inten):


    envmap_dict = {'type': 'envmap',
                   'filename': envmap_path,
                   'scale': inten,
                   'to_world': mi.ScalarTransform4f.rotate(axis=[0, 0, 1], angle=90) @
                               mi.ScalarTransform4f.rotate(axis=[1, 0, 0], angle=90)}

    object_dict = {'type': 'obj',
                   'filename': ply_path,
                   'face_normals': True  # todo check this
                   }


    scene_dict = {'type': 'scene',
                  'object': object_dict,
                  'emitter': envmap_dict}
    scene = mi.load_dict(scene_dict)
    return scene


expeirment = 'cube'
ply_path = get_ply_path(expeirment)
envmap_path = get_light_probe_path(expeirment)
inten = get_light_inten(expeirment)
# inten = 1
scene = create_mitsuba_scene_envmap(ply_path, envmap_path, inten)

params = mi.traverse(scene)

key = 'emitter.data'

env_map_old = params[key]
print(env_map_old)
env_map = torch.zeros_like(env_map_old.torch())
params[key] = env_map
params.update()



