import mitsuba as mi
import numpy as np
import math

import torch

import mitsuba_tools as mit
import drjit as dr
import matplotlib.pyplot as plt
import torchvision
import time
from sampling import visualize3dscatter
mi.set_variant("cuda_ad_rgb")

if __name__ == '__main__':
    ply_path = "./scenes/cube_rough.obj"
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
    # Accumulated result
    result = mi.Float(0)

    # si.is_valid()[100] = True
    xl_100 = []
    yl_100 = []
    zl_100 = []

    xw_100 = []
    yw_100 = []
    zw_100 = []

    coord_100 = [xl_100, yl_100, zl_100, xw_100, yw_100, zw_100]

    t0 = time.time()
    """use python for loop"""
    for i in range(ambient_ray_count):
        sample_1, sample_2 = rng.next_float32(), rng.next_float32()
        wo_local = mi.warp.square_to_cosine_hemisphere([sample_1, sample_2])  # or uniform
        wo_world = si.sh_frame.to_world(wo_local)
        ray_2 = si.spawn_ray(wo_world)
        ray_2.maxt = ambient_range
        result[~scene.ray_test(ray_2)] += 1.0

        for index, j in enumerate([wo_local.x[100], wo_local.y[100], wo_local.z[100], wo_world.x[100], wo_world.y[100], wo_world.z[100]]):
            coord_100[index].append(j)

    mask = si.is_valid()
    result = dr.select(mask, result, 0)

    # Divide the result by the number of samples
    result = result / ambient_ray_count

    t1 = time.time()
    print(f"loop time: {t1-t0}")

    image = mi.TensorXf(result, shape=image_res)

    image = image.torch()
    ToPILImage = torchvision.transforms.ToPILImage()
    image = ToPILImage(image)
    image.show()

    # """test"""
    # visualize3dscatter(xl_100, yl_100, zl_100, size=3)
    # visualize3dscatter(xw_100, yw_100, zw_100, size=3)


    # """test"""
    # x = wo_local.x.torch()
    # y = wo_local.y.torch()
    # z = wo_local.z.torch()
    # print(x*x + y*y + z*z)
    #
    # x = wo_world.x.torch()
    # y = wo_world.y.torch()
    # z = wo_world.z.torch()
    # print(x*x + y*y + z*z)
    #
    # norm = torch.sqrt(x*x + y*y + z*z)
    # x = x / norm
    # y = y / norm
    # z = z / norm
    # print(x * x + y * y + z * z)
    #
    #
    # visualize3dscatter(wo_local.x.numpy(), wo_local.y.numpy(), wo_local.z.numpy())
    # visualize3dscatter(wo_world.x.numpy(), wo_world.y.numpy(), wo_world.z.numpy())
    # visualize3dscatter(x.cpu(), y.cpu(), z.cpu())

    # print(si.sh_frame)
