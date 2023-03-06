import mitsuba as mi
import drjit as dr
import mitsuba_tools as mit
import torchvision
import torch
mi.set_variant("cuda_ad_rgb")
from tone_mapping import show_image, save_image
import imageio


ply_path = "../scenes/lego.obj"
envmap_path = "../scenes/bunny/museum.exr"

object_dict = {'type': 'obj',
               'filename': ply_path}

envmap_dict = {'type': 'envmap',
               'filename': envmap_path,
               'scale': 1}

scene_dict = {'type': 'scene',
              'object': object_dict,
              'emitter': envmap_dict}
scene = mi.load_dict(scene_dict)

params = mi.traverse(scene)
print(params)

envmap = params['emitter.data'].torch()

print(envmap.unique())
# show_image(envmap)

envmap = torch.clip(envmap,0, None)
print(envmap.unique())
# show_image(envmap)

envmap = imageio.imread(envmap_path, 'exr')
envmap = torch.tensor(envmap)
print(envmap.unique())






