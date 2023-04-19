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
        wh = dr.normalize(wo + si.wi)

        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)

        m_alpha = self.alpha_model(si, si.is_valid())  # roughness
        m_alpha = mi.Float(m_alpha)
        D = DistributeBeckmann(wh, m_alpha)
        G = G1(si.wi, wh, m_alpha) * G1(wo, wh, m_alpha)
        eta_c = 0
        F = mi.fresnel_conductor(dr.dot(si.wi, wh), eta_c)
        result = (D * G * F / (4. * cos_theta_i * cos_theta_o)).torch()[:, None]
        result = albedo / math.pi * (1-albedo.max()) * result
        return result

    def pdf(self,
            si: mi.Interaction3f,
            wo: mi.Vector3f):


        cos_theta_o = mi.Frame3f.cos_theta(wo)
        wh = dr.normalize(wo + si.wi)
        active = si.is_valid() & (cos_theta_o > 0.)
        m_alpha = self.alpha_model(si, active) # roughness
        m_alpha = mi.Float(m_alpha)
        albedo = self.diffuse_model(si, si.is_valid())  # [N,3]

        D = DistributeBeckmann(wh, m_alpha).torch()[:, None]
        jacobian = 1 / (4 * dr.abs(dr.dot(wh, wo))).torch()[:, None]
        pdf = albedo * D * mi.Frame3f.cos_theta(wh).torch()[:, None] * jacobian + (1-albedo) * mi.Frame3f.cos_theta(wo).torch()[:, None] / math.pi
        pdf = torch.clip(pdf, 0, None)
        pdf = torch.where(cos_theta_o.torch()[:, None] <= 0, 0, pdf)
        return pdf

def DistributeBeckmann(wh, alpha):
    tanTheta = mi.Frame3f.tan_theta(wh)
    cosTheta = mi.Frame3f.cos_theta(wh)
    a = dr.exp(-(tanTheta * tanTheta) / (alpha * alpha))
    b = math.pi * alpha * alpha * dr.power(cosTheta, 4)
    return a / b

def G1(wv, wh, alpha):
    c = dr.dot(wv, wh) / mi.Frame3f.cos_theta(wv)
    b = 1 / (alpha * mi.Frame3f.tan_theta(wv))
    result = dr.select(b<1.6, (3.535 * b + 2.181 * b * b) / (1. + 2.276 * b + 2.577 * b * b), 1)
    result = dr.select(c<=0, 0, result)
    return result

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
        output = torch.ones_like(xsurf_valid[:, 0]) * .2
        return output