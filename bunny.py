import mitsuba as mi
import drjit as dr
from sampling import visualize3dscatter
mi.set_variant("cuda_ad_rgb")

scene = mi.load_file('scenes/bunny/bunny.xml')

params = mi.traverse(scene)

# print(params)
#
# # Make a backup copy
# param_res = params['my_envmap.scale']
# param_ref = params['my_envmap.data']

spp = 1
sensor = scene.sensors()[0]
film = sensor.film()
sampler = sensor.sampler()
film_size = film.crop_size()
total_sample_count = dr.prod(film_size) * spp
if sampler.wavefront_size() != total_sample_count:
    sampler.seed(0, total_sample_count)
pos = dr.arange(mi.UInt32, total_sample_count)
pos //= spp
scale = mi.Vector2f(1.0 / film_size[0], 1.0 / film_size[1])
pos = mi.Vector2f(mi.Float(pos % int(film_size[0])),
                  mi.Float(pos // int(film_size[0])))  # [[p1]*spp, [p2]*spp, ...]
pos += sampler.next_2d()

rays, weights = sensor.sample_ray_differential(
    time=0,
    sample1=0,
    # A uniformly distributed 1D value that is used to sample the spectral dimension of the sensitivity profile.
    sample2=pos * scale,
    # This argument corresponds to the sample position in fractional pixel coordinates relative to the crop window of the underlying film.
    sample3=0
    # A uniformly distributed sample on the domain [0,1]^2. This argument determines the position on the aperture of the sensor.
)

si = scene.ray_intersect(rays)

emitter = scene.environment()

# print(emitter.sample_ray(time=0,
#                          sample1=0,  # A uniformly distributed 1D value that is used to sample the spectral dimension of the emission profile.
#                          sample2=pos * scale,  # A uniformly distributed sample on the domain [0,1]^2. For sensor endpoints, this argument corresponds to the sample position in fractional pixel coordinates relative to the crop window of the underlying film.
#                          sample3=0,))  # A uniformly distributed sample on the domain [0,1]^2. For sensor endpoints, this argument determines the position on the aperture of the sensor.


DirectionSample, Color = emitter.sample_direction(it=si,
                                                  sample=sampler.next_2d(),
                                                  active=si.is_valid())

wo = DirectionSample.p.torch().cpu()  # outgoing direction
pdf = DirectionSample.pdf.torch().cpu()

wo_valid = wo[si.is_valid()]
pdf_valid = pdf[si.is_valid()]

visualize3dscatter(wo_valid[:,0], wo_valid[:,1], wo_valid[:,2])