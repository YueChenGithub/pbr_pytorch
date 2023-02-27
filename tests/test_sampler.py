import mitsuba as mi
import drjit as dr

mi.set_variant("cuda_ad_rgb")

image_res = [2,2]

"""PCG32 random number generator"""
rng = mi.PCG32(size=dr.prod(image_res))
print(len(rng.next_float32()))

"""independent sampler, sample_count=1"""
spp = 1
sampler_dict = {'type': 'independent',
                'sample_count': spp}  # <- setting sample_count has no mean

sampler = mi.load_dict(sampler_dict)
total_sample_count = dr.prod(image_res) * spp  # wavefront?
sampler.seed(0, total_sample_count)
print(len(sampler.next_1d()))


"""independent sampler, sample_count=256"""
spp = 256
sampler_dict = {'type': 'independent',
                'sample_count': spp}

sampler = mi.load_dict(sampler_dict)
total_sample_count = dr.prod(image_res) * spp  # wavefront?
sampler.seed(0, total_sample_count)
print(len(sampler.next_1d()))