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
    # inten = 1
    scene = create_mitsuba_scene_envmap(ply_path, envmap_path, inten)

    spp = 64
    debug = True

    i = 1

    view = f"test_{i:03d}"
    metadata_path = f"./scenes/cube_rough/{view}/metadata.json"
    """read information"""
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    cam_transform_mat = metadata['cam_transform_mat']
    cam_angle_x = metadata['cam_angle_x']
    imw = metadata['imw']
    imh = metadata['imh']


    emitter = scene.emitters()[0]
    sensor = mit.create_mitsuba_sensor(cam_transform_mat, cam_angle_x, imw, imh)

    # calculate surface intersection of rays shooting from camera
    si, sampler = camera_intersect(scene, sensor, spp)

    # # visualize normal and intersection mask
    # show_normal_mask_image(imh, imw, si, spp)


    # # visualize diffuse
    # show_diffuse_image(diffuse, imh, imw, spp)

    # Sampling incident direction using cos-weighted sampling (diffuse sampling)
    # todo below consider invalid si to save calculation?
    wi_local = mi.warp.square_to_cosine_hemisphere(sampler.next_2d())
    pdf = mi.warp.square_to_cosine_hemisphere_pdf(wi_local)
    pdf = pdf.torch()[:, None]  # [N,1]

    pdf = torch.clip(pdf, 0, None)  # todo pdf can be negative sometimes? Should we clip it like this?

    # transform local incident direction to world coordinate
    wi_world = si.sh_frame.to_world(wi_local)  # outgoing direction

    #  Spawn incident rays at the surface interactions towards incident direction
    ray_i = si.spawn_ray(wi_world)

    print(ray_i)





if __name__ == '__main__':
    main()
