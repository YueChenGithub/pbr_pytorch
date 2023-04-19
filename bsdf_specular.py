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
        m_alpha = self.alpha_model(si, si.is_valid()) # roughness
        m_sample_visible = True
        distr = mi.MicrofacetDistribution(mi.MicrofacetType.Beckmann, m_alpha, m_sample_visible)

        m, pdf = distr.sample(si.wi, sample2)
        wo = mi.reflect(si.wi, m)
        pdf = self.pdf(si, wo)
        w = 1 / pdf
        w[pdf == 0] = 0
        brdf_multiplier = self.eval(si, wo) * w # [N,3]
        return wo, brdf_multiplier, pdf

    def eval(self,
             si: mi.Interaction3f,
             wo: mi.Vector3f):
        albedo = self.diffuse_model(si, si.is_valid())  # [N,3]
        H = dr.normalize(wo + si.wi)

        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)

        m_alpha = self.alpha_model(si, si.is_valid() & (cos_theta_i > 0.) & (cos_theta_o > 0.)) # roughness
        m_sample_visible = True
        distr = mi.MicrofacetDistribution(mi.MicrofacetType.Beckmann, m_alpha, m_sample_visible)
        D = distr.eval(H)
        G = distr.G(si.wi, wo, H)
        result = D * G / (4. * mi.Frame3f.cos_theta(si.wi))
        eta_c = 0
        F = mi.fresnel_conductor(dr.dot(si.wi, H), eta_c)
        result = albedo * result.torch()[:, None]
        value = F.torch()[:, None] * result
        return value

    def pdf(self,
            si: mi.Interaction3f,
            wo: mi.Vector3f):
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)
        m = dr.normalize(wo + si.wi)
        active = si.is_valid() & (cos_theta_i > 0.) & (cos_theta_o > 0.) & (dr.dot(si.wi, m)>0) & (dr.dot(wo, m) > 0)
        m_alpha = self.alpha_model(si, active) # roughness
        m_sample_visible = True
        distr = mi.MicrofacetDistribution(mi.MicrofacetType.Beckmann, m_alpha, m_sample_visible)
        result = distr.eval(m) * distr.smith_g1(si.wi, m) / (4. * cos_theta_i)
        pdf = result.torch()[:, None]  # [N,1]
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
        # always return [0.8, 0.8, 0.8]
        output = torch.ones_like(xsurf_valid[:, 0]) * 0.1
        return output