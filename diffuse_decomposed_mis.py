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
from bsdf_diffuse2 import BSDF_Diffuse, Diffuse_MLP_trivial, Diffuse_Model
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
    # Input parameters
    sensor = mit.create_mitsuba_sensor(cam_transform_mat, cam_angle_x, imw, imh)
    emitter = scene.emitters()[0]
    ray, sampler = generate_camera_rays(scene, sensor, spp)

    diffuse_mlp = Diffuse_MLP_trivial()
    diffuse_model = Diffuse_Model(diffuse_mlp)
    bsdf = BSDF_Diffuse(diffuse_model)
    throughput = torch.tensor([1], dtype=torch.float32, device='cuda')
    active = mi.Mask(True)

    N_rays = dr.shape(ray.o)[1]
    result = torch.zeros((N_rays, 3), dtype=torch.float32, device='cuda')  # todo

    emitter_sampling = True

    m_max_depth = 4
    for depth in range(m_max_depth):

        si = scene.ray_intersect(ray, active=active)
        if depth == 0:
            pass
        else:
            # BSDF sampling
            ds = mi.DirectionSample3f(scene, si, prev_si)
            if emitter_sampling:
                em_pdf = scene.pdf_emitter_direction(si, ds, active=active)
            else:
                em_pdf = mi.Float(0)
            mis_bsdf = mis_weight(prev_bsdf_pdf.torch(), em_pdf.torch())[:, None]  # [N, 1]
            L = emitter_eval(emitter, si)
            L = torch.permute(L, (1, 0))  # [N, 3]
            result = throughput * L * mis_bsdf + result  # [N, 3]

        if depth == m_max_depth - 1:
            break

        active_next = si.is_valid()

        if emitter_sampling:
            # emitter sampling
            # ds, em_weight = scene.sample_emitter_direction(si, sampler.next_2d(), True, active_next)
            # em_weight = em_weight.torch()

            ds, _ = scene.sample_emitter_direction(si, sampler.next_2d(), True, active_next)
            # recompute em_weight (L/pdf)
            ds.d = dr.normalize(ds.p - si.p)
            ray2 = si.spawn_ray(ds.d)
            si2 = scene.ray_intersect(ray2, active=active_next)
            L = emitter_eval(emitter, si2)
            L = torch.permute(L, (1, 0))  # [N, 3]
            em_weight = L / ds.pdf.torch()[:, None]
            em_weight = torch.where(ds.pdf.torch()[:, None] == 0, 0, em_weight)

            wo = si.to_local(ds.d)
            bsdf_val = bsdf.eval(si, wo, active=active)
            bsdf_pdf = bsdf.pdf(si, wo, active=active)
            bsdf_val = torch.where((bsdf_pdf == 0)[:, None], 0, bsdf_val)
            mis_em = mis_weight(ds.pdf.torch(), bsdf_pdf)[:, None]  # [N, 1]
            result = (throughput * bsdf_val * em_weight * mis_em + result)

        sample1 = sampler.next_1d()
        sample2 = sampler.next_2d()
        bsdf_sample, bsdf_weight = bsdf.sample(si, sample1, sample2, active=active_next)

        ray = si.spawn_ray(si.to_world(bsdf_sample.wo))

        throughput = throughput * bsdf_weight  # [N, 3]
        prev_si = si
        prev_bsdf_pdf = bsdf_sample.pdf
        active = active_next

    return result


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
