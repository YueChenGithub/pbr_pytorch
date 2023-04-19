import mitsuba as mi
import math
import drjit as dr
import torch

mi.set_variant("cuda_ad_rgb")


class BSDF_Diffuse:

    def __init__(self, diffuse_model):
        self.diffuse_model = diffuse_model

    def sample(self,
               si: mi.Interaction3f,
               sample1: mi.Float,
               sample2: mi.Point2f):
        wo = mi.warp.square_to_cosine_hemisphere(sample2)
        pdf = self.pdf(wo)
        brdf_multiplier = self.eval(si, wo) / pdf  # [N,3]
        brdf_multiplier = torch.where(pdf == 0, 0, brdf_multiplier)
        return wo, brdf_multiplier, pdf

    def eval(self,
             si: mi.Interaction3f,
             wo: mi.Vector3f):
        albedo = self.diffuse_model(si, si.is_valid())  # [N,3]
        cos_term = si.sh_frame.cos_theta(wo).torch()[:, None]  # [N,1]
        cos_term = torch.clip(cos_term, 0, None)  # consider only the upper hemisphere
        value = albedo / math.pi * cos_term  # [N, 3]
        return value

    def pdf(self,
            wo: mi.Vector3f):
        pdf = mi.warp.square_to_cosine_hemisphere_pdf(wo)
        pdf = pdf.torch()[:, None]  # [N,1]
        pdf = torch.clip(pdf, 0, None)
        return pdf


class Diffuse_Model():
    def __init__(self, diffuse_mlp):
        self.diffuse_mlp = diffuse_mlp

    def __call__(self, si: mi.Interaction3f, active: mi.Mask):
        active = active.torch().bool()
        N = active.shape[0]
        xsurf_valid = si.p.torch()[active]
        diffuse_valid = self.diffuse_mlp(xsurf_valid)
        diffuse = torch.zeros(N, 3, dtype=torch.float32).cuda()
        diffuse[active] = diffuse_valid
        diffuse = torch.clip(diffuse, 0, None)
        return diffuse


class Diffuse_MLP_trivial():
    def __call__(self, xsurf_valid: torch.Tensor):
        # always return [0.8, 0.8, 0.8]
        output = torch.ones_like(xsurf_valid) * 0.8
        return output