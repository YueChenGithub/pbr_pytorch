import mitsuba as mi
import drjit as dr
import mitsuba_tools as mit
import torchvision
import torch
from tone_mapping import show_image, save_image
import json
import math
from path_tool import get_ply_path, get_light_probe_path, get_light_inten

mi.set_variant("cuda_ad_rgb")


def create_mitsuba_scene_envmap(ply_path, envmap_path, inten):


    envmap_dict = {'type': 'envmap',
                   'filename': envmap_path,
                   'scale': inten,
                   'to_world': mi.ScalarTransform4f.rotate(axis=[0, 0, 1], angle=90) @
                               mi.ScalarTransform4f.rotate(axis=[1, 0, 0], angle=90)}

    object_dict = {'type': 'obj',
                   'filename': ply_path,
                   # 'face_normals': True  # todo check this
                   }


    scene_dict = {'type': 'scene',
                  'object': object_dict,
                  'emitter': envmap_dict}
    scene = mi.load_dict(scene_dict)
    return scene


def render(cam_angle_x, cam_transform_mat, imh, imw, scene, spp):
    emitter = scene.emitters()[0]
    sensor = mit.create_mitsuba_sensor(cam_transform_mat, cam_angle_x, imw, imh)

    si, sampler = camera_intersect(scene, sensor, spp)

    # # visualize normal and intersection mask
    # show_normal_mask_image(imh, imw, si, spp)

    # get diffuse color
    diffuse = eval_diffuse_trivial(si)
    diffuse = diffuse / math.pi  # [N,3]
    diffuse = torch.clip(diffuse, 0, None)

    # # visualize diffuse
    # show_diffuse_image(diffuse, imh, imw, spp)

    # BRDF Sampling
    # todo below consider invalid si
    wo_local = mi.warp.square_to_uniform_hemisphere(sampler.next_2d())
    pdf = mi.warp.square_to_cosine_hemisphere_pdf(wo_local)
    pdf = pdf.torch()[:, None]  # [N,1]

    # print(pdf.min())
    pdf = torch.clip(pdf, 1e-12, None)  # todo pdf can be negative sometimes?
    # pdf = torch.abs(pdf)

    wo_world = si.sh_frame.to_world(wo_local)  # outgoing direction
    Epsilon = 1e-6
    wo_ray_world = mi.Ray3f(o=si.p + Epsilon * wo_world,
                            d=wo_world)  # move origin towards outgoing direction a little bit

    # calculate intersection of outgoing ray and object
    si2 = scene.ray_intersect(wo_ray_world)

    # evaluate environment light intensity for rays that did not hit object
    L = emitter.eval(si2, active=~si2.is_valid())  # Attention: did not work for mitsuba constant env map
    L = L.torch()  # [N,3]
    L = torch.clip(L, 0, None)

    # rendering equation
    cos_term = si.sh_frame.cos_theta(wo_local).torch()[:, None]  # [N,1]
    cos_term = torch.clip(cos_term, 0, None)  # consider only the upper hemisphere
    color = diffuse * L * cos_term / pdf  # [N, 3]

    # color[~si.is_valid().torch().bool()] = torch.tensor([0,0,0], dtype=torch.float32).cuda()  # <- already exist in diffuse

    color = color.reshape(imh, imw, spp, 3)
    color = color.mean(axis=2)


    return color


def show_diffuse_image(diffuse, imh, imw, spp):
    color = diffuse.reshape(imh, imw, spp, 3)
    color = torch.mean(color, dim=2)
    show_image(color)


def eval_diffuse_trivial(si: mi.Interaction3f):
    mask = si.is_valid().torch().bool()
    p = si.p.torch()

    # return RGB value [1,1,1] for each valid point
    color = torch.zeros_like(p)
    color[mask] = torch.tensor([0.8, 0.8, 0.8], dtype=torch.float32).cuda()
    return color


def show_normal_mask_image(imh, imw, si, spp):
    mask = si.is_valid()
    result = dr.select(mask, si.n, [0, 0, 0])
    normal = result.torch()
    normal = normal.reshape(imh, imw, spp, 3)
    normal = normal.mean(axis=2)  # average over spp
    normal = torch.clip(normal, 0, 1)  # todo it is correct?
    mask = mask.torch().byte() * 255
    mask = mask.reshape(imh, imw, spp)
    mask, _ = torch.max(mask, axis=2, keepdim=True)  # If any camera ray intersects with the object, mask = 1
    show_image(normal)
    show_image(mask)


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
    expeirment = 'lego'
    ply_path = get_ply_path(expeirment)
    envmap_path = get_light_probe_path(expeirment)
    inten = get_light_inten(expeirment)
    scene = create_mitsuba_scene_envmap(ply_path, envmap_path, inten)
    spp = 128

    for i in range(100):
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
        n = 10
        for i in range(n):
            image = render(cam_angle_x, cam_transform_mat, imh, imw, scene, spp)
            image_list.append(image)
        image = torch.mean(torch.stack(image_list), dim=0)
        # show_image(image)
        save_image_path = f"./tests/diffuse_decomposed/{expeirment}_{n}_cws/{view}.png"
        save_image(image, save_image_path)


if __name__ == '__main__':
    main()
