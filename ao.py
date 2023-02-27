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
    ply_path = "./scenes/Cube_rough.obj"
    scene = mit.create_mitsuba_scene(ply_path)

    """camera initialization"""
    # Camera origin in world space
    cam_origin = mi.Point3f(0, 1, 3)

    # Camera view direction in world space
    cam_dir = dr.normalize(mi.Vector3f(0, -0.5, -1))

    # Camera width and height in world space
    cam_width = 2.0
    cam_height = 2.0

    # Image pixel resolution
    image_res = [512, 512]

    """create rays shooting from the camera"""
    # Construct a grid of 2D coordinates
    x, y = dr.meshgrid(
        dr.linspace(mi.Float, -cam_width / 2, cam_width / 2, image_res[0]),
        dr.linspace(mi.Float, -cam_height / 2, cam_height / 2, image_res[1])
    )

    # Ray origin in local coordinates
    ray_origin_local = mi.Vector3f(x, y, 0)

    # Ray origin in world coordinates
    ray_origin = mi.Frame3f(cam_dir).to_world(ray_origin_local) + cam_origin

    # Create rays
    ray = mi.Ray3f(o=ray_origin, d=cam_dir)


    """find intersection"""
    t0 = time.time()
    si = scene.ray_intersect(ray)
    t1 = time.time()
    print(f"intersection time: {t1 - t0}")

    """ao parameters"""
    ambient_range = 0.75
    ambient_ray_count = 256

    """random number generator"""
    # Initialize the random number generator
    rng = mi.PCG32(size=dr.prod(image_res))

    """integrator"""
    # Loop iteration counter
    i = mi.UInt32(0)

    # Accumulated result
    result = mi.Float(0)


    t0 = time.time()
    # Initialize the loop state (listing all variables that are modified inside the loop)
    loop = mi.Loop(name="", state=lambda: (rng, i, result))

    while loop(si.is_valid() & (i < ambient_ray_count)):
        # 1. Draw some random numbers
        sample_1, sample_2 = rng.next_float32(), rng.next_float32()

        # 2. Compute directions on the hemisphere using the random numbers
        wo_local = mi.warp.square_to_uniform_hemisphere([sample_1, sample_2])

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
    print(f"loop time: {t1-t0}")

    image = mi.TensorXf(result, shape=image_res)

    image = image.torch()
    ToPILImage = torchvision.transforms.ToPILImage()
    image = ToPILImage(image)
    image.show()
