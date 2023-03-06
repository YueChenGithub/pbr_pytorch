import mitsuba as mi
import numpy as np
import math
import mitsuba_tools as mit
import drjit as dr
import matplotlib.pyplot as plt
import torchvision
import time

mi.set_variant("cuda_ad_rgb")


def main():
    ply_path = "./scenes/Cube_rough.obj"
    scene = mit.create_mitsuba_scene(ply_path)
    cam_transform_mat = "-0.9999999403953552,0.0,0.0,0.0,0.0,-0.7341099977493286,0.6790305972099304,2.737260103225708,0.0,0.6790306568145752,0.7341098785400391,2.959291696548462,0.0,0.0,0.0,1.0"
    cam_angle_x = 0.6911112070083618
    imw = 512
    imh = 512
    image = get_ao_image(cam_angle_x, cam_transform_mat, imh, imw, scene, spp=512)
    image.show()


def get_ao_image(cam_angle_x, cam_transform_mat, imh, imw, scene, spp):
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
    t0 = time.time()
    si = scene.ray_intersect(rays)
    t1 = time.time()
    print(f"intersection time: {t1 - t0}")
    """ao parameters"""
    ambient_range = 0.75
    ambient_ray_count = 1
    """integrator"""
    # Loop iteration counter
    i = mi.UInt32(0)
    # Accumulated result
    result = mi.Float(0)
    t0 = time.time()
    # Initialize the loop state (listing all variables that are modified inside the loop)
    loop = mi.Loop(name="", state=lambda: (sampler, i, result))
    while loop(si.is_valid() & (i < ambient_ray_count)):
        # 2. Compute directions on the hemisphere using the random numbers
        wo_local = mi.warp.square_to_uniform_hemisphere(sampler.next_2d())

        # Alternatively, we could also sample a cosine-weighted hemisphere
        # wo_local = mi.warp.square_to_cosine_hemisphere([sample_1, sample_2])

        # 3. Transform the sampled directions to world space
        wo_world = si.sh_frame.to_world(wo_local)

        # 4. Spawn a new ray starting at the surface interactions
        ray_2 = si.spawn_ray(wo_world)

        # 5. Set a maximum intersection distance to only account for the close-by geometry
        ray_2.maxt = ambient_range

        # 6. Accumulate a value of 1 if not occluded (0 otherwise)
        result[~scene.ray_test(ray_2)] += 1.0

        # 7. Increase loop iteration counter
        i += 1
    # Divide the result by the number of samples
    result = result / ambient_ray_count
    t1 = time.time()
    print(f"loop time: {t1 - t0}")
    image = mi.TensorXf(result, shape=list(film_size) + [spp])
    image = image.torch()
    image = image.mean(axis=2)
    ToPILImage = torchvision.transforms.ToPILImage()
    image = ToPILImage(image)
    return image


if __name__ == '__main__':
    main()
