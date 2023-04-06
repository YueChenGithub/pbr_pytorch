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


def mis(emitter, sampler, scene, si):
    # query diffuse color of the intersection
    diffuse = eval_diffuse_trivial(si)  # contains info about si => invalid si has diffuse zero
    diffuse = diffuse / math.pi  # [N,3]
    diffuse = torch.clip(diffuse, 0, None)
    L, w, pdf, si2 = cws(emitter, sampler, scene, si)
    L_ems, cos_term_ems, pdf_ems = ems(emitter, sampler, scene, si)
    # the balance heuristic with beta = 1
    w_cws = torch.pow(pdf_cws, beta - 1) / (torch.pow(pdf_cws, beta) + torch.pow(pdf_ems, beta))
    w_ems = torch.pow(pdf_ems, beta - 1) / (torch.pow(pdf_cws, beta) + torch.pow(pdf_ems, beta))
    w_cws = torch.where(pdf_cws > 1e-8, w_cws, 0)
    w_ems = torch.where(pdf_ems > 1e-8, w_ems, 0)
    # rendering equation
    color_ems = w_ems * diffuse * L_ems * cos_term_ems
    return L_cws, color_ems, cos_term_cws*diffuse*w_cws, si2


def ems(emitter, sampler, scene, si):
    # Sampling incident direction using emitter sampling
    DirectionSample, _ = emitter.sample_direction(it=si,
                                                  sample=sampler.next_2d(),
                                                  active=si.is_valid())
    wi_world = DirectionSample.d  # incident direction
    pdf = DirectionSample.pdf.torch()[:, None]
    pdf = torch.clip(pdf, 0, None)
    # transform global incident direction to local frame
    wi_local = si.sh_frame.to_local(wi_world)



    #  Spawn incident rays at the surface interactions towards incident direction
    ray_i = si.spawn_ray(wi_world)
    # calculate intersection of incident rays and object
    si2 = scene.ray_intersect(ray_i, active=si.is_valid())  # reject invalid ray_i



    L = emitter.eval(si2, active=~si2.is_valid()).torch()
    L = torch.clip(L, 0, None)  # todo can be negative in some hdr envmap

    return L, cos_term, pdf


def cws(emitter, sampler, scene, si):
    # query diffuse color of the intersection
    diffuse = eval_diffuse_trivial(si)

    # sample incident direction
    brdf = Diffuse(diffuse, si, sampler)
    w, wi_world, pdf = brdf.sample()

    # create incident rays
    ray_i = si.spawn_ray(wi_world)

    # calculate surface intersection of incident rays
    si2 = scene.ray_intersect(ray_i, active=si.is_valid())  # reject invalid ray_i

    # calculate radiance
    L = emitter.eval(si2, active=~si2.is_valid())
    L = L.torch()  # [N,3]
    return L, w, pdf, si2


def eval(si, emitter, active):
    active = active.torch().bool()
    v = emitter.world_transform().inverse().transform_affine(-si.wi)
    uv = mi.Point2f(dr.atan2(v.x, -v.z) / (2*math.pi), dr.safe_acos(v.y) / math.pi)
    params = mi.traverse(emitter)
    data = params['data'].torch()  # [imh, imw+1, 3]
    res = mi.Vector2u(data.shape[1], data.shape[0])  # [imw+1, imh]

    scale = params['scale'].torch()

    uv.x -= .5 / (res.x - 1)

    uv -= dr.floor(uv)
    uv *= mi.Vector2f(res - 1)
    pos = dr.minimum(mi.Point2u(uv), res - 2)

    w1 = uv - mi.Point2f(pos)
    w0 = 1. - w1
    width = res.x
    index = dr.fma(pos.y, width, pos.x)

    width = mi.Int64(width).torch()
    index = mi.Int64(index).torch()

    data = data.reshape(-1, data.shape[-1])

    v00 = gather(data, index, active)
    v10 = gather(data, index + 1, active)
    v01 = gather(data, index + width, active)
    v11 = gather(data, index + width + 1, active)

    w0x = w0.x.torch()[:, None]
    w0y = w0.y.torch()[:, None]
    w1x = w1.x.torch()[:, None]
    w1y = w1.y.torch()[:, None]

    v0 = w0x * v00 + w1x * v10
    v1 = w0x * v01 + w1x * v11

    v = w0y * v0 + w1y * v1

    output = v * scale

    return output


def gather(data, index, active):
    v = data[index, :3]
    v[~active,:] = 0
    return v



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
    # inten = 1
    scene = create_mitsuba_scene_envmap(ply_path, envmap_path, inten)
    spp = 128
    debug = True

    global beta  # fixme: beta increases, the output will be darker
    beta = 2

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
            save_image_path = f"./tests/diffuse_decomposed/{expeirment}_{n}_mis_beta{beta}_bounce{bounce}/{view}.png"
            save_image(image, save_image_path)
        else:
            show_image(image)


if __name__ == '__main__':
    main()
