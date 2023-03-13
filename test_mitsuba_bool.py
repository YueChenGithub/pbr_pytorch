import mitsuba as mi
mi.set_variant("cuda_ad_rgb")

a = mi.Bool(True)
b = mi.Bool(False)

print(~b)