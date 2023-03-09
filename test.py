import imageio
import numpy as np


input_path = "light_probe/cube.exr"
envmap = imageio.v3.imread(input_path)
print(envmap.dtype)

new_envmap = np.zeros_like(envmap)
print(new_envmap.shape)
new_envmap = new_envmap[:,:,:3]  # remove alpha
print(new_envmap.shape)
new_envmap[0:3,0:3,:] = 1


output_path = "light_probe/point.exr"
imageio.v3.imwrite(output_path, [new_envmap])
