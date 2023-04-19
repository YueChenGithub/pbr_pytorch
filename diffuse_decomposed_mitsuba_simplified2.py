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
from bsdf_diffuse import Diffuse, Diffuse_model_trivial
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


@dr.wrap_ad(source='torch', target='drjit')
def sample_emitter_direction(scene, si, sampler, active=mi.Mask(True)):
    ds, em_weight = scene.sample_emitter_direction(si, sampler.next_2d(), True, active=active)
    tensor = dr.zeros(mi.TensorXf, shape=dr.shape(em_weight))
    tensor[0] = em_weight.x
    tensor[1] = em_weight.y
    tensor[2] = em_weight.z
    return ds, tensor


def render(cam_angle_x, cam_transform_mat, imh, imw, scene, spp, debug):
    # Input parameters
    sensor = mit.create_mitsuba_sensor(cam_transform_mat, cam_angle_x, imw, imh)
    emitter = scene.emitters()[0]
    ray, sampler = generate_camera_rays(scene, sensor, spp)

    m_max_depth = 4
    active = mi.Mask(True)

    if m_max_depth < 1: return {0.}

    throughput = torch.tensor([1], dtype=torch.float32, device='cuda')
    result = torch.zeros((imh * imw * spp, 3), dtype=torch.float32, device='cuda')
    depth = mi.UInt32(0)

    prev_si = dr.zeros(mi.Interaction3f)
    prev_bsdf_pdf = mi.Float(1.)
    diffuse_model_trivial = Diffuse_model_trivial()

    while (1):
        si = scene.ray_intersect(ray, active)

        '''---------------------- Direct emission ----------------------'''
        if dr.all(depth == 0):  # not include background
            pass
        else:
            ds = mi.DirectionSample3f(scene, si, prev_si)
            em_pdf = scene.pdf_emitter_direction(prev_si, ds, active=active)
            mis_bsdf = mis_weight(prev_bsdf_pdf.torch(), em_pdf.torch())
            L = emitter_eval(emitter, si, active=((prev_bsdf_pdf > 0) & active))
            L = torch.permute(L, (1, 0))  # [N, 3]
            result = throughput * L * mis_bsdf[:, None] + result


        '''-------------------- Stopping criterion ---------------------'''
        # Continue tracing the path at this point?
        active_next = (depth + 1 < m_max_depth) & si.is_valid()
        if dr.all(~active_next):
            break  # check if all active_next are negative


        '''---------------------- Emitter sampling ----------------------'''
        bsdf = Diffuse(diffuse_model_trivial)
        active_em = active_next & mi.Mask(True)  # always smooth
        # Sample the emitter
        ds, em_weight = scene.sample_emitter_direction(si, sampler.next_2d(), True, active_em & active)

        # recompute em_weight (L/pdf)
        ds.d = dr.normalize(ds.p - si.p)
        ray2 = si.spawn_ray(ds.d)
        si2 = scene.ray_intersect(ray2, active & active_em)
        L = emitter_eval(emitter, si2, active=active & active_em)
        L = torch.permute(L, (1, 0))  # [N, 3]
        em_weight = L / ds.pdf.torch()[:, None]
        em_weight = torch.where(ds.pdf.torch()[:, None]==0, 0, em_weight)


        active_em &= dr.neq(ds.pdf, 0.)
        wo = si.to_local(ds.d)

        '''------ Evaluate BSDF * cos(theta) and sample direction -------'''
        sample_1 = sampler.next_1d()
        sample_2 = sampler.next_2d()
        bsdf_val, bsdf_pdf, bsdf_sample, bsdf_weight = bsdf.eval_pdf_sample(si, wo, sample_1, sample_2, active=active)
        '''--------------- Emitter sampling contribution ----------------'''
        if dr.any(active_em):
            mis_em = mis_weight(ds.pdf.torch(), bsdf_pdf)
            result[active_em.torch().bool()] = (throughput * bsdf_val * em_weight * mis_em[:, None] + result)[
                active_em.torch().bool()]

        '''---------------------- BSDF sampling ----------------------'''
        ray = si.spawn_ray(si.to_world(bsdf_sample.wo))

        '''------ Update loop variables based on current interaction ------'''
        throughput = throughput * bsdf_weight
        prev_si = si
        prev_bsdf_pdf = bsdf_sample.pdf
        depth[si.is_valid()] += 1
        active = active_next

    return result


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
    spp = 64
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
            save_image_path = f"./tests/diffuse_decomposed/{expeirment}_mis_single/{view}.png"
            save_image(image, save_image_path)
        else:
            show_image(image)


if __name__ == '__main__':
    main()
