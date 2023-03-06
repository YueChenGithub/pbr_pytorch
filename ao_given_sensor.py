import mitsuba as mi
import numpy as np
import math
import mitsuba_tools as mit
import drjit as dr
import matplotlib.pyplot as plt
import torchvision
import time

mi.set_variant("cuda_ad_rgb")

if __name__ == '__main__':
    ply_path = "./scenes/cube.ply"
    scene = mit.create_mitsuba_scene(ply_path)

    cam_transform_mat = "-0.9999999403953552,0.0,0.0,0.0,0.0,-0.7341099977493286,0.6790305972099304,2.737260103225708,0.0,0.6790306568145752,0.7341098785400391,2.959291696548462,0.0,0.0,0.0,1.0"
    cam_angle_x = 0.6911112070083618
    imw = 512
    imh = 512
    sensor = mit.create_mitsuba_sensor(cam_transform_mat, cam_angle_x, imw, imh)

    """create rays shooting from the camera"""
    film = sensor.film()
    sampler = sensor.sampler()
    film_size = film.crop_size()
    spp = 1
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
        sample1=0,  # A uniformly distributed 1D value that is used to sample the spectral dimension of the sensitivity profile.
        sample2=pos * scale,  # This argument corresponds to the sample position in fractional pixel coordinates relative to the crop window of the underlying film.
        sample3=0  # A uniformly distributed sample on the domain [0,1]^2. This argument determines the position on the aperture of the sensor.
    )
    print(rays)


    """find intersection"""
    t0 = time.time()
    si = scene.ray_intersect(rays)
    t1 = time.time()
    print(f"intersection time: {t1 - t0}")

    """ao parameters"""
    ambient_range = 0.75
    ambient_ray_count = 256

    """random number generator"""
    # Initialize the random number generator
    rng = mi.PCG32(size=dr.prod(film_size))

    """integrator"""
    # Accumulated result
    result = mi.Float(0)

    t0 = time.time()
    """use python for loop"""
    for i in range(ambient_ray_count):
        sample_1, sample_2 = rng.next_float32(), rng.next_float32()
        wo_local = mi.warp.square_to_uniform_hemisphere([sample_1, sample_2])
        wo_world = si.sh_frame.to_world(wo_local)
        ray_2 = si.spawn_ray(wo_world)
        ray_2.maxt = ambient_range
        result[~scene.ray_test(ray_2)] += 1.0
    mask = si.is_valid()
    result = dr.select(mask, result, 0)

    # Divide the result by the number of samples
    result = result / ambient_ray_count

    t1 = time.time()
    print(f"loop time: {t1 - t0}")

    image = mi.TensorXf(result, shape=film_size)

    image = image.torch()
    ToPILImage = torchvision.transforms.ToPILImage()
    image = ToPILImage(image)
    image.show()
