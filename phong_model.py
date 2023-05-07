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
from bsdf_phong import BSDF_Diffuse, Diffuse_MLP_trivial, Diffuse_Model
import copy

mi.set_variant("cuda_ad_rgb")


def create_mitsuba_scene_envmap(ply_path, envmap_path, inten):
    envmap_dict = {'type': 'envmap',
                   'filename': envmap_path,
                   'scale': inten,
                   'to_world': mi.ScalarTransform4f.rotate(axis=[0, 0, 1], angle=90) @
                               mi.ScalarTransform4f.rotate(axis=[1, 0, 0], angle=90)}

    bsdf_dict = {'type': 'diffuse',
                 'reflectance': {
                     'type': 'rgb',
                     'value': [0.8, 0.8, 0.8]
                 }}

    object_dict = {'type': 'obj',
                   'filename': ply_path,
                   'face_normals': True,  # todo check this
                   # 'bsdf': bsdf_dict  # todo implement this part
                   }

    scene_dict = {'type': 'scene',
                  'object': object_dict,
                  'emitter': envmap_dict}
    scene = mi.load_dict(scene_dict)
    return scene


@dr.wrap_ad(source='torch', target='drjit')
def emitter_eval(emitter, si, active=mi.Mask(True)):
    L = emitter.eval(si, active=~si.is_valid() & active)
    tensor = dr.zeros(mi.TensorXf, shape=dr.shape(L))
    tensor[0] = L.x
    tensor[1] = L.y
    tensor[2] = L.z
    return tensor


def render(cam_angle_x, cam_transform_mat, imh, imw, scene, spp, debug):
    sensor = mit.create_mitsuba_sensor(cam_transform_mat, cam_angle_x, imw, imh)
    emitter = scene.emitters()[0]
    ray, sampler = generate_camera_rays(scene, sensor, spp)

    diffuse_mlp = Diffuse_MLP_trivial()
    diffuse_model = Diffuse_Model(diffuse_mlp)
    bsdf = BSDF_Diffuse(diffuse_model)

    N_rays = dr.shape(ray.o)[1]
    result = torch.zeros((N_rays, 3), dtype=torch.float32, device='cuda')  # todo
    active = mi.Mask(True)


    si = scene.ray_intersect(ray, active=active)
    # cws
    L, brdf_multiplier, si2 = cws(bsdf, emitter, sampler, scene, si)
    result += brdf_multiplier * L

    return result


def cws(bsdf, emitter, sampler, scene, si):
    sample1 = sampler.next_1d()
    sample2 = sampler.next_2d()
    wo_local, brdf_multiplier, pdf = bsdf.sample(si, sample1, sample2)
    wo_world = si.sh_frame.to_world(wo_local)
    #  Spawn incident rays
    ray_o = si.spawn_ray(wo_world)
    si2 = scene.ray_intersect(ray_o, active=si.is_valid())  # reject invalid ray_i
    L = emitter.eval(si2, active=~si2.is_valid()).torch()
    L = torch.clip(L, 0, None)
    return L, brdf_multiplier, si2


@dr.wrap_ad(source='torch', target='drjit')
def emitter_eval(emitter, si):
    L = emitter.eval(si, active=~si.is_valid())
    tensor = dr.zeros(mi.TensorXf, shape=dr.shape(L))
    tensor[0] = L.x
    tensor[1] = L.y
    tensor[2] = L.z
    return tensor


def print_result(result, spp):
    result = result.reshape(512, 512, spp, 3)
    result = result.mean(dim=2)
    show_image(result)


def mis_weight(pdf_a: torch.Tensor, pdf_b: torch.Tensor):  # power heuristic
    pdf_a = pdf_a * pdf_a
    pdf_b = pdf_b * pdf_b
    w = pdf_a / (pdf_a + pdf_b)
    return torch.where(pdf_a == 0, 0, w)


def generate_camera_rays(scene, sensor, spp):
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

    return rays, sampler


def main():
    expeirment = 'cube'
    ply_path = get_ply_path(expeirment)
    envmap_path = get_light_probe_path(expeirment)
    inten = get_light_inten(expeirment)
    # inten = 1
    scene = create_mitsuba_scene_envmap(ply_path, envmap_path, inten)
    spp = 128
    debug = True

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
            # image = image.torch()
            image = image.reshape(512, 512, spp, 3)
            image = image.mean(dim=2)
            image_list.append(image)
        image = torch.mean(torch.stack(image_list), dim=0)

        if not debug:
            save_image_path = f"./tests/diffuse_decomposed/{expeirment}_mis/{view}.png"
            save_image(image, save_image_path)
        else:
            show_image(image)


if __name__ == '__main__':
    main()
