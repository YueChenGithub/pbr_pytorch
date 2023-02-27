import mitsuba as mi
import numpy as np
import math
import mitsuba_tools as mit
import drjit as dr
import matplotlib.pyplot as plt
import torch
import torchvision
import time
import plotly
import plotly.graph_objs as go

mi.set_variant("cuda_ad_rgb")


def main():
    # cos_weighted()

    """environment map sampling"""
    


def cos_weighted():
    """cos-weighted sampling"""
    n = torch.tensor([-1, 1, 1], dtype=float)
    n = n / torch.norm(n)
    image_res = [1, 1]
    # Initialize the random number generator
    rng = mi.PCG32(size=dr.prod(image_res))
    frame = mi.Frame3f(n.tolist())
    ray_count = 512
    coord = []
    for i in range(ray_count):
        # draw random variables on 2D
        sample_1, sample_2 = rng.next_float32(), rng.next_float32()

        # warp to 3d with cos-weighted sampling
        wo_local = mi.warp.square_to_cosine_hemisphere([sample_1, sample_2])
        pdf = mi.warp.square_to_cosine_hemisphere_pdf(wo_local)

        # Transform the sampled directions to world space

        wo_world = frame.to_world(wo_local)

        coord.append([wo_local, wo_world])
    coord = torch.tensor(coord)
    print(coord.shape)
    i_pt = 0
    x_local = coord[:, 0, 0, i_pt]
    y_local = coord[:, 0, 1, i_pt]
    z_local = coord[:, 0, 2, i_pt]
    x_world = coord[:, 1, 0, i_pt]
    y_world = coord[:, 1, 1, i_pt]
    z_world = coord[:, 1, 2, i_pt]
    visualize3dscatter(x_local, y_local, z_local, size=3)
    visualize3dscatter(x_world, y_world, z_world, size=3)


def visualize3dscatter(x, y, z, size=1):
    trace = go.Scatter3d(
        x=x,  # <-- Put your data instead
        y=y,  # <-- Put your data instead
        z=z,  # <-- Put your data instead
        mode='markers',
        marker={
            'size': size,
            'opacity': 0.8,
        }
    )
    layout = go.Layout(
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
    )
    data = [trace]
    plot_figure = go.Figure(data=data, layout=layout)
    plot_figure.show()


if __name__ == '__main__':
    main()
