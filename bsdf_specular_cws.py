import mitsuba as mi
import math
import drjit as dr
import torch

mi.set_variant("cuda_ad_rgb")


class BSDF_Diffuse:

    def __init__(self, diffuse_model, alpha_model):
        self.diffuse_model = diffuse_model
        self.alpha_model = alpha_model

    def sample(self,
               si: mi.Interaction3f,
               sample1: mi.Float,
               sample2: mi.Point2f):
        wo = mi.warp.square_to_cosine_hemisphere(sample2)
        pdf = self.pdf(wo)
        w = 1 / pdf
        w[pdf == 0] = 0
        brdf_multiplier = self.eval(si, wo)
        brdf_multiplier = brdf_multiplier * w  # [N,3]
        return wo, brdf_multiplier, pdf

    def eval(self,
             si: mi.Interaction3f,
             wo: mi.Vector3f):
        albedo = self.diffuse_model(si, si.is_valid())  # [N,3]
        cos_term = si.sh_frame.cos_theta(wo).torch()[:, None]  # [N,1]
        cos_term = torch.clip(cos_term, 0, None)  # consider only the upper hemisphere
        wi_reflect = mi.reflect(si.wi, si.n)
        n = 100
        a = dr.dot(wo, wi_reflect).torch()[:, None]  # [N,1]
        a = torch.clip(a, 0, None)
        # print(a.min(), a.max())

        albedo_max, _ = torch.max(albedo, 1)
        albedo_max = albedo_max[:, None]
        pho_s = 1 - albedo_max

        specular_raw = pho_s * (n + 2) / (2 * math.pi) * torch.pow(a, n)
        specular = torch.zeros_like(specular_raw)
        specular[si.is_valid().torch().bool()] = specular_raw[si.is_valid().torch().bool()]

        brdf = albedo / math.pi + specular
        # brdf = albedo / math.pi
        value = brdf * cos_term  # [N, 3]
        # print(value.min(), value.max())
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
        # output[:, 1] = 0
        # output[:, 2] = 0
        return output


class Alpha_Model():
    def __init__(self, diffuse_mlp):
        self.diffuse_mlp = diffuse_mlp

    def __call__(self, si: mi.Interaction3f, active: mi.Mask):
        active = active.torch().bool()
        N = active.shape[0]
        xsurf_valid = si.p.torch()[active]
        diffuse_valid = self.diffuse_mlp(xsurf_valid)
        diffuse = torch.zeros(N, dtype=torch.float32).cuda()
        diffuse[active] = diffuse_valid
        diffuse = torch.clip(diffuse, 0, None)
        return diffuse


class Alpha_MLP_trivial():
    def __call__(self, xsurf_valid: torch.Tensor):
        output = torch.ones_like(xsurf_valid[:, 0]) * 0.7
        return output
