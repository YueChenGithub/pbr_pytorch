import mitsuba as mi
import numpy as np
import math

import torch


def create_mitsuba_scene(ply_path):
    if ply_path[-3:] == 'obj':
        object_dict = {'type': 'obj',
                       'filename': ply_path}

    if ply_path[-3:] == 'ply':
        object_dict = {'type': 'ply',
                       'filename': ply_path}

    scene_dict = {'type': 'scene',
                  'object': object_dict}
    scene = mi.load_dict(scene_dict)
    return scene

def create_mitsuba_sensor(cam_transform_mat, cam_angle_x, imw, imh):
    cam_transform_mat = np.array(list(cam_transform_mat.split(',')), dtype=float)  # str2array
    cam_transform_mat = cam_transform_mat.reshape(4, 4)

    cam_transform_mat = mi.ScalarTransform4f(cam_transform_mat) @ mi.ScalarTransform4f.scale(
        [-1, 1, -1])  # change coordinate from blender to mitsuba (flip x and z axis)
    # cam_transform_mat = mi.ScalarTransform4f(cam_transform_mat)


    sensor_dict = {'type': 'perspective',
                   'to_world': cam_transform_mat,
                   'fov': float(cam_angle_x * 180 / math.pi),
                   'film': {'type': 'hdrfilm',
                            'width': int(imw),
                            'height': int(imh)},
                   'sampler': {'type': 'independent',
                               'sample_count': 1
                               }
                   }
    sensor = mi.load_dict(sensor_dict)

    return sensor

def print_min_max(x:torch.Tensor):
    print(x.min(), x.max())
