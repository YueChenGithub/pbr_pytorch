import mitsuba as mi
import drjit as dr
import mitsuba_tools as mit
import torchvision
import torch

mi.set_variant("cuda_ad_rgb")

ply_path = "../scenes/lego.obj"
scene = mit.create_mitsuba_scene(ply_path)

envmap_dict2 = {'type': 'constant',
                'radiance': {
                    'type': 'rgb',
                    'value': 1.0,
                }}

envmap = mi.load_dict(envmap_dict2)
print('Initialization:\n',envmap)
envmap.set_scene(scene)
print('--------------------------')
print('After assigning a scene:\n', envmap)


