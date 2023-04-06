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
from brdf import Diffuse
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


def render(cam_angle_x, cam_transform_mat, imh, imw, scene, spp, debug):
    emitter = scene.emitters()[0]
    sensor = mit.create_mitsuba_sensor(cam_transform_mat, cam_angle_x, imw, imh)

    # calculate surface intersection of rays shooting from camera
    si, sampler = camera_intersect(scene, sensor, spp)

    global bounce
    bounce = 5

    prev_w = 1
    color = 0
    # start a loop for each bounce
    for i in range(bounce):
        L, w, si = cws(emitter, sampler, scene, si)
        prev_w = prev_w * w
        color = color + prev_w * L



    # pdf can be zero => color can be inf
    color = torch.nan_to_num(color, nan=0)

    # take average over spp
    color = color.reshape(imh, imw, spp, 3)
    color = color.mean(axis=2)  # [imh, imw, 3]

    return color




def cws(emitter, sampler, scene, si):
    # query diffuse color of the intersection
    diffuse = eval_diffuse_trivial(si)

    # sample incident direction
    brdf = Diffuse(diffuse, si, sampler)
    w, wi_world, _ = brdf.sample()

    # create incident rays
    ray_i = si.spawn_ray(wi_world)

    # calculate surface intersection of incident rays
    si2 = scene.ray_intersect(ray_i, active=si.is_valid())  # reject invalid ray_i

    # calculate radiance
    L = emitter.eval(si2, active=~si2.is_valid())
    L = L.torch()  # [N,3]
    return L, w, si2


def eval_diffuse_trivial(si: mi.Interaction3f):
    mask = si.is_valid().torch().bool()
    p = si.p.torch()

    # return RGB value [1,1,1] for each valid point
    color = torch.zeros_like(p)
    color[mask] = torch.tensor([0.8, 0.8, 0.8], dtype=torch.float32).cuda()
    return color


def camera_intersect(scene, sensor, spp):
    """create rays shooting from the camera"""
    film = sensor.film()
    sampler = sensor.sampler()
    film_size = film.crop_size()
    total_sample_count = dr.prod(film_size) * spp
    if sampler.wavefront_size() != total_sample_count:
        # sampler.seed(0, total_sample_count)  # todo look into the seed
        sampler.seed(torch.randint(0, 10000, (1,)).item(), total_sample_count)
    pos = dr.arange(mi.UInt32, total_sample_count)
    pos //= spp
    scale = mi.Vector2f(1.0 / film_size[0], 1.0 / film_size[1])
    pos = mi.Vector2f(mi.Float(pos % int(film_size[0])),
                      mi.Float(pos // int(film_size[0])))  # [[p1]*spp, [p2]*spp, ...]
    pos += sampler.next_2d()
    rays, weights = sensor.sample_ray_differential(  # todo figure out the camera center
        time=0,
        sample1=0,
        # A uniformly distributed 1D value that is used to sample the spectral dimension of the sensitivity profile.
        sample2=pos * scale,
        # This argument corresponds to the sample position in fractional pixel coordinates relative to the crop window of the underlying film.
        sample3=0
        # A uniformly distributed sample on the domain [0,1]^2. This argument determines the position on the aperture of the sensor.
    )
    """find intersection"""
    si = scene.ray_intersect(rays)

    return si, sampler


def main():
    expeirment = 'cube'
    ply_path = get_ply_path(expeirment)
    envmap_path = get_light_probe_path(expeirment)
    inten = get_light_inten(expeirment)
    scene = create_mitsuba_scene_envmap(ply_path, envmap_path, inten)
    spp = 128
    debug = False

    if debug:
        n = 1
    else:
        n = 200

    for i in range(n):
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
        image_list = []
        n = 1
        for i in range(n):
            image = render(cam_angle_x, cam_transform_mat, imh, imw, scene, spp, debug)
            image_list.append(image)
        image = torch.mean(torch.stack(image_list), dim=0)

        if not debug:
            save_image_path = f"./tests/diffuse_decomposed/{expeirment}_{n}_cws_bounce{bounce}_brdf/{view}.png"
            save_image(image, save_image_path)
        else:
            show_image(image)


if __name__ == '__main__':
    main()
