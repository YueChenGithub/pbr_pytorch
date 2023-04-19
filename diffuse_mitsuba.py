import mitsuba as mi
import drjit as dr
import mitsuba_tools as mit
import torchvision
import torch
from tone_mapping import show_image, save_image
import json
from path_tool import get_ply_path, get_light_probe_path, get_light_inten

mi.set_variant("cuda_ad_rgb")


def create_mitsuba_scene_envmap(ply_path, envmap_path, inten):
    diffuse_dict = {'type': 'diffuse',
                    'reflectance': {
                        'type': 'rgb',
                        'value': [0.8, 0.8, 0.8]
                    }}

    envmap_dict = {'type': 'envmap',
                   'filename': envmap_path,
                   'scale': inten,
                   'to_world': mi.ScalarTransform4f.rotate(axis=[0, 0, 1], angle=90) @
                               mi.ScalarTransform4f.rotate(axis=[1, 0, 0], angle=90)}

    # envmap_constant_dict = {'type': 'constant',
    #                         'radiance': {
    #                             'type': 'rgb',
    #                             'value': 1.0,
    #                         }}

    # point_emitter_dict = {'type': 'point',
    #                       'position': [0.0, 5.0, 0.0],
    #                       'intensity': {
    #                           'type': 'rgb',
    #                           'value': 1,
    #                       }}

    object_dict = {'type': 'obj',
                   'filename': ply_path,
                   'bsdf': diffuse_dict,
                   'face_normals': True  # todo check this
                   }

    scene_dict = {'type': 'scene',
                  'object': object_dict,
                  'emitter': envmap_dict}
    scene = mi.load_dict(scene_dict)
    return scene


def render(cam_angle_x, cam_transform_mat, imh, imw, scene):
    sensor = mit.create_mitsuba_sensor(cam_transform_mat, cam_angle_x, imw, imh)
    Path_tracer_dict = {'type': 'path',
                        'max_depth': bounce+1,
                        'hide_emitters': True}
    Depth_integrator = {'type': 'depth'}
    Direct_integrator = {'type': 'direct',
                         'hide_emitters': True}
    integrator = mi.load_dict(Path_tracer_dict)
    image = mi.render(scene, spp=128, sensor=sensor, integrator=integrator)
    return image


def main():
    expeirment = 'cube'
    ply_path = get_ply_path(expeirment)
    envmap_path = get_light_probe_path(expeirment)
    inten = get_light_inten(expeirment)

    scene = create_mitsuba_scene_envmap(ply_path, envmap_path, inten)
    global bounce
    bounce = 3

    for i in range(200):
        view = f"test_{i:03d}"
        metadata_path = f"./scenes/cube_rough/{view}/metadata.json"
        """read information"""
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        cam_transform_mat = metadata['cam_transform_mat']
        cam_angle_x = metadata['cam_angle_x']
        imw = metadata['imw']
        imh = metadata['imh']

        """rendering"""
        image = render(cam_angle_x, cam_transform_mat, imh, imw, scene)
        save_image_path = f"./tests/diffuse_mitsuba/{expeirment}_bounce{bounce}/{view}.png"
        save_image(image, save_image_path, tone_mapping=True)




if __name__ == '__main__':
    main()
