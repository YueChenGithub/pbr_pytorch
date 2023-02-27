import mitsuba as mi
import numpy as np
import math
import mitsuba_tools as mit
import drjit as dr
import matplotlib.pyplot as plt
import torchvision
import torch
import time

mi.set_variant("cuda_ad_rgb")


def main():
    ply_path = "./scenes/Cube_rough.obj"
    scene = mit.create_mitsuba_scene(ply_path)
    cam_transform_mat = "-0.9999999403953552,0.0,0.0,0.0,0.0,-0.7341099977493286,0.6790305972099304,2.737260103225708,0.0,0.6790306568145752,0.7341098785400391,2.959291696548462,0.0,0.0,0.0,1.0"
    cam_angle_x = 0.6911112070083618
    imw = 512
    imh = 512
    normal, mask = get_normal_image(cam_angle_x, cam_transform_mat, imh, imw, scene, spp=512)
    normal.show()
    mask.show()


def get_normal_image(cam_angle_x, cam_transform_mat, imh, imw, scene, spp):
    sensor = mit.create_mitsuba_sensor(cam_transform_mat, cam_angle_x, imw, imh)
    """create rays shooting from the camera"""
    film = sensor.film()
    sampler = sensor.sampler()
    film_size = film.crop_size()
    total_sample_count = dr.prod(film_size) * spp
    if sampler.wavefront_size() != total_sample_count:
        sampler.seed(0, total_sample_count)
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
    # t0 = time.time()
    si = scene.ray_intersect(rays)
    # t1 = time.time()
    # print(f"intersection time: {t1 - t0}")

    """integrator"""
    mask = si.is_valid()
    result = dr.select(mask, si.n, [0,0,0])

    normal = result.torch()
    normal = normal.reshape(imh, imw, spp, 3)
    normal = normal.mean(axis=2)
    normal = torch.clip(normal, 0, 1)  # todo it is correct?

    mask = mask.torch().byte() * 255
    mask = mask.reshape(imh, imw, spp)
    mask, _ = torch.max(mask, axis=2)

    ToPILImage = torchvision.transforms.ToPILImage()
    normal = ToPILImage(normal.permute(2,0,1))
    mask = ToPILImage(mask)

    return normal, mask


if __name__ == '__main__':
    main()
