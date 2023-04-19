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


def render(cam_angle_x, cam_transform_mat, imh, imw, scene, spp, debug):
    # Class member:
    m_max_depth = 2
    m_hide_emitters = mi.Mask(True)
    m_rr_depth = 4

    # Input parameters
    sensor = mit.create_mitsuba_sensor(cam_transform_mat, cam_angle_x, imw, imh)
    emitter = scene.emitters()[0]
    ray, sampler = generate_camera_rays(scene, sensor, spp)
    active = mi.Mask(True)


    if m_max_depth < 1: return {0., False}

    '''--------------------- Configure loop state ----------------------'''
    throughput = mi.Spectrum(1.)
    result = torch.zeros((imh * imw * spp, 3), dtype=torch.float32, device='cuda')
    eta = mi.Float(1.)
    depth = mi.UInt32(0)

    '''If m_hide_emitters == false, the environment emitter will be visible'''
    valid_ray = ~m_hide_emitters  # T/F

    '''Variables caching information from the previous bounce'''
    prev_si = dr.zeros(mi.Interaction3f)
    prev_bsdf_pdf = mi.Float(1.)
    prev_bsdf_delta = mi.Bool(True)
    diffuse_model_trivial = Diffuse_model_trivial()


    while(1):
        si = scene.ray_intersect(ray, active)

        '''---------------------- Direct emission ----------------------'''
        if dr.any(dr.neq(si.emitter(scene), None)):
            ds = mi.DirectionSample3f(scene, si, prev_si)
            em_pdf = mi.Float(0)

            if dr.any(~prev_bsdf_delta):
                em_pdf = scene.pdf_emitter_direction(prev_si, ds, ~prev_bsdf_delta)

            mis_bsdf = mis_weight(prev_bsdf_pdf.torch(), em_pdf.torch())

            L = emitter_eval(emitter, si, active=((prev_bsdf_pdf > 0) & active))
            L = torch.permute(L, (1, 0))  # [N, 3]

            result = throughput.torch() * L * mis_bsdf[:, None] + result
            if debug:
                print_result(result, spp)
            # result = L * mis_bsdf[:, None] + result


            # L = ds.emitter.eval(si, active=((prev_bsdf_pdf > 0) & active))
            # result = throughput * L * mis_bsdf + result
            # print_result(result, spp, valid_ray)


        # Continue tracing the path at this point?
        active_next = (depth + 1 < m_max_depth) & si.is_valid()

        if dr.all(~active_next):
            break  # check if all active_next are negative

        bsdf = Diffuse(diffuse_model_trivial)

        '''---------------------- Emitter sampling ----------------------'''
        active_em = active_next & mi.Mask(True) # always smooth


        if dr.any(active_em):
            # Sample the emitter
            ds, em_weight = scene.sample_emitter_direction(si, sampler.next_2d(), True, active=active_em & active)
            active_em &= dr.neq(ds.pdf, 0.)


            wo = si.to_local(ds.d)

        '''------ Evaluate BSDF * cos(theta) and sample direction -------'''
        sample_1 = sampler.next_1d()
        sample_2 = sampler.next_2d()

        bsdf_val, bsdf_pdf, bsdf_sample, bsdf_weight = bsdf.eval_pdf_sample(si, wo, sample_1, sample_2, active=active)

        '''--------------- Emitter sampling contribution ----------------'''
        if dr.any(active_em):
            mis_em = torch.where(ds.delta.torch().bool(), 1, mis_weight(ds.pdf.torch(), bsdf_pdf))
            result[active_em.torch().bool()] = (throughput.torch() * bsdf_val * em_weight.torch() * mis_em[:, None] + result)[active_em.torch().bool()]
            if debug:
                print_result(result, spp)
            # result[active_em.torch().bool()] = (bsdf_val * em_weight.torch() * mis_em[:, None] + result)[active_em.torch().bool()]


        '''---------------------- BSDF sampling ----------------------'''
        ray = si.spawn_ray(si.to_world(bsdf_sample.wo))

        '''------ Update loop variables based on current interaction ------'''
        throughput = throughput * mi.Spectrum(bsdf_weight)
        eta = eta * bsdf_sample.eta
        valid_ray |= active & si.is_valid()
        prev_si = si
        prev_bsdf_pdf = bsdf_sample.pdf
        prev_bsdf_delta = mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta)

        '''-------------------- Stopping criterion ---------------------'''
        depth[si.is_valid()] += 1
        throughput_max = dr.max(mi.unpolarized_spectrum(throughput))
        rr_prob = dr.minimum(throughput_max * dr.sqr(eta), 0.95)
        rr_active = depth >= m_rr_depth
        rr_continue = sampler.next_1d() < rr_prob

        throughput[rr_active] *= dr.rcp(dr.detach(rr_prob))
        active = active_next & (~rr_active | rr_continue) & dr.neq(throughput_max, 0.)
        # active = active_next



    return torch.where(valid_ray.torch().bool()[:, None], result, 0), valid_ray


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
        n0 = 66
        n = n0+1
    else:
        n = 200
        n0=0

    for i in range(n0, n):
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
            image, valid_ray = render(cam_angle_x, cam_transform_mat, imh, imw, scene, spp, debug)
            # image = image.torch()
            image = image.reshape(512, 512, spp, 3)
            image = image.mean(dim=2)
            image_list.append(image)
        image = torch.mean(torch.stack(image_list), dim=0)

        if not debug:
            save_image_path = f"./tests/diffuse_decomposed/{expeirment}_ours_b3_tp/{view}.png"
            save_image(image, save_image_path)
        else:
            show_image(image)


if __name__ == '__main__':
    main()
