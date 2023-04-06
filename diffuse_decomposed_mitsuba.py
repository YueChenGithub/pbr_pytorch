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

    bsdf_dict = {'type': 'diffuse',
                 'reflectance': {
                     'type': 'rgb',
                     'value': [0.8, 0.8, 0.8]
                 }}

    object_dict = {'type': 'obj',
                   'filename': ply_path,
                   'face_normals': True,  # todo check this
                   'bsdf': bsdf_dict  # todo implement this part
                   }

    scene_dict = {'type': 'scene',
                  'object': object_dict,
                  'emitter': envmap_dict}
    scene = mi.load_dict(scene_dict)
    return scene


def render(cam_angle_x, cam_transform_mat, imh, imw, scene, spp, debug):
    # Class member:
    m_max_depth = 10
    m_hide_emitters = mi.Mask(True)
    m_rr_depth = 10000

    # Input parameters
    sensor = mit.create_mitsuba_sensor(cam_transform_mat, cam_angle_x, imw, imh)
    ray, sampler = generate_camera_rays(scene, sensor, spp)
    active = mi.Mask(True)

    if m_max_depth < 1: return {0., False}

    '''--------------------- Configure loop state ----------------------'''
    throughput = mi.Spectrum(1.)
    result = mi.Spectrum(0.)
    eta = 1.
    depth = mi.UInt32(0)

    '''If m_hide_emitters == false, the environment emitter will be visible'''
    valid_ray = ~m_hide_emitters & dr.neq(scene.environment(), None)  # T/F


    '''Variables caching information from the previous bounce'''
    prev_si = dr.zeros(mi.Interaction3f)
    prev_bsdf_pdf = 1.
    prev_bsdf_delta = mi.Bool(True)
    bsdf_ctx = mi.BSDFContext()

    while (1):
        si = scene.ray_intersect(ray)

        '''---------------------- Direct emission ----------------------'''
        if dr.any(dr.neq(si.emitter(scene), None)):  # if any si located in emitter
            ds = mi.DirectionSample3f(scene, si, prev_si)
            em_pdf = 0.
            if dr.any(~prev_bsdf_delta):  # if any prev_bsdf_delta is False  # todo what is delta?
                em_pdf = scene.pdf_emitter_direction(prev_si, ds, active=~prev_bsdf_delta)


            '''Compute MIS weight for emitter sample from previous bounce'''
            mis_bsdf = mis_weight(prev_bsdf_pdf, em_pdf)

            '''Accumulate, being careful with polarization (see spec_fma)'''
            result = spec_fma(throughput, ds.emitter.eval(si, prev_bsdf_pdf > 0.) * mis_bsdf, result)
            print(torch.unique(throughput.torch()))

        '''Continue tracing the path at this point?'''
        active_next = (depth + 1 < m_max_depth) & si.is_valid()

        # print_result(result, spp, valid_ray)
        '''early exit for scalar mode'''
        if dr.all(~active_next):
            # print('not active_next found')
            break  # check if all active_next are negative

        bsdf = si.bsdf(ray)

        '''---------------------- Emitter sampling ----------------------'''

        '''Perform emitter sampling?'''
        active_em = active_next & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)

        if dr.any(active_em):
            '''Sample the emitter'''
            ds, em_weight = scene.sample_emitter_direction(si, sampler.next_2d(), True, active=active_em)
            active_em &= dr.neq(ds.pdf, 0.)

            wo = si.to_local(ds.d)

        '''------ Evaluate BSDF * cos(theta) and sample direction -------'''
        sample_1 = sampler.next_1d()
        sample_2 = sampler.next_2d()
        bsdf_val, bsdf_pdf = bsdf.eval_pdf(bsdf_ctx, si, wo)  # seperate eval_pdf_sample because it is not found
        bsdf_sample, bsdf_weight = bsdf.sample(bsdf_ctx, si, sample_1, sample_2)

        '''--------------- Emitter sampling contribution ----------------'''
        if dr.any(active_em):
            bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi)

            '''Compute the MIS weight'''
            mis_em = dr.select(ds.delta, 1., mis_weight(ds.pdf, bsdf_pdf))

            '''Accumulate, no polarization'''
            result[active_em] = spec_fma(throughput, bsdf_val * em_weight * mis_em, result)

        '''---------------------- BSDF sampling ----------------------'''
        bsdf_weight = si.to_world_mueller(bsdf_weight, -bsdf_sample.wo, si.wi)
        ray = si.spawn_ray(si.to_world(bsdf_sample.wo))

        '''------ Update loop variables based on current interaction ------'''
        throughput *= bsdf_weight
        eta *= bsdf_sample.eta
        valid_ray |= active & si.is_valid() & ~mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Null)

        '''Information about the current vertex needed by the next iteration'''
        prev_si = si
        prev_bsdf_pdf = bsdf_sample.pdf
        prev_bsdf_delta = mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta)
        '''-------------------- Stopping criterion ---------------------'''
        depth[si.is_valid()] += 1
        throughput_max = dr.max(mi.unpolarized_spectrum(throughput))
        rr_prob = dr.minimum(throughput_max * dr.sqr(eta), 0.95)
        rr_active = mi.Mask(depth >= m_rr_depth)
        rr_continue = mi.Mask(sampler.next_1d() < rr_prob)
        throughput[rr_active] *= dr.rcp(dr.detach(rr_prob))
        # throughput[rr_active] *= 1/rr_prob

        active = active_next & (~rr_active | rr_continue) & dr.neq(throughput_max, 0.)


    return dr.select(valid_ray, result, 0.), valid_ray

def spec_fma(a, b, c):
    return dr.fma(a, b, c)


def print_result(result, spp, valid_ray):
    result = dr.select(valid_ray, result, 0.)
    result = result.torch()
    result = result.reshape(512, 512, spp, 3)
    result = result.mean(dim=2)
    show_image(result)


def mis_weight(pdf_a, pdf_b):  # power heuristic
    pdf_a *= pdf_a
    pdf_b *= pdf_b
    w = pdf_a / (pdf_a + pdf_b)
    return dr.detach(dr.select(dr.isfinite(w), w, 0.))
    # return dr::detach<true>(dr::select(dr::isfinite(w), w, 0.f))


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
        n = 50

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
            image, valid_ray = render(cam_angle_x, cam_transform_mat, imh, imw, scene, spp, debug)
            image = image.torch()
            image = image.reshape(512, 512, spp, 3)
            image = image.mean(dim=2)
            image_list.append(image)
        image = torch.mean(torch.stack(image_list), dim=0)


        if not debug:
            save_image_path = f"./tests/diffuse_decomposed/{expeirment}_{n}_mitsuba/{view}.png"
            save_image(image, save_image_path)
        else:
            show_image(image)


if __name__ == '__main__':
    main()
