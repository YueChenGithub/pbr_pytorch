import mitsuba as mi
import math
import drjit as dr
import torch

mi.set_variant("cuda_ad_rgb")


class BSDF_Diffuse:

    def __init__(self, diffuse_model):
        self.diffuse_model = diffuse_model
        self.m_specularSamplingWeight = 0.5
        self.e = 100000


    def sample(self,
               si: mi.Interaction3f,
               sample1: mi.Float,
               sample2: mi.Point2f):
        R = mi.reflect(si.wi)
        exponent = torch.tensor([self.e]).cuda()
        exponent = mi.Float(exponent.flatten().float())

        # sample phong lobe
        sinAlpha = dr.sqrt(1 - dr.power(sample2.y, 2 / (exponent + 1)))
        cosAlpha = dr.power(sample2.y, 1 / (exponent + 1))
        phi = (2.0 * math.pi) * sample2.x
        localDir = mi.Vector3f(sinAlpha * dr.cos(phi),
                               sinAlpha * dr.sin(phi),
                               cosAlpha)
        woSpec = mi.Frame3f(R).to_world(localDir)

        # sample diffuse lobe
        woDiff = mi.warp.square_to_cosine_hemisphere(sample2)

        # choose between specular and diffuse
        choseSpecular = (sample1 < self.m_specularSamplingWeight)
        wo = dr.select(choseSpecular, woSpec, woDiff)

        # compute pdf and brdf_multiplier
        pdf = self.pdf(si, wo)
        w = 1 / pdf
        w[pdf == 0] = 0
        brdf_multiplier = self.eval(si, wo) * w  # [N,3]
        brdf_multiplier[(si.sh_frame.cos_theta(wo) <= 0).torch().bool()] = 0  # consider only the upper hemisphere
        return wo, brdf_multiplier, pdf

    def eval(self,
             si: mi.Interaction3f,
             wo: mi.Vector3f):
        frame = si.sh_frame
        unvalid = ((frame.cos_theta(si.wi) <= 0) | (frame.cos_theta(wo) <= 0)).torch().bool()

        # specular term
        alpha = dr.dot(wo, mi.reflect(si.wi)).torch()[:, None]
        alpha = torch.clip(alpha, 0, None)
        exponent = torch.tensor([self.e]).cuda()
        m_specularReflectance = torch.ones_like(alpha) * 0.1
        m_specularReflectance[~si.is_valid().torch().bool()] = 0
        specTerm = m_specularReflectance * ((exponent + 2) * (1 / (2 * math.pi)) * torch.pow(alpha, exponent))

        # diffuse term
        albedo = self.diffuse_model(si, si.is_valid())  # [N,3]
        diffuseTerm = albedo / math.pi

        # combine
        value = (diffuseTerm + specTerm) * frame.cos_theta(wo).torch()[:, None]
        value[unvalid] = 0
        return value

    def pdf(self,
            si: mi.Interaction3f,
            wo: mi.Vector3f):
        frame = si.sh_frame

        # diffuse pdf
        diffuseProb = mi.warp.square_to_cosine_hemisphere_pdf(wo).torch()[:, None]  # [N,1]

        # specular pdf
        alpha = dr.dot(wo, mi.reflect(si.wi)).torch()[:, None]
        alpha = torch.clip(alpha, 0, None)
        exponent = torch.tensor([self.e]).cuda()
        specProb = torch.pow(alpha, exponent) * (exponent + 1.0) / (2.0 * math.pi)

        # combine
        pdf = self.m_specularSamplingWeight * specProb + (1 - self.m_specularSamplingWeight) * diffuseProb
        unvalid = ((frame.cos_theta(si.wi) <= 0) | (frame.cos_theta(wo) <= 0)).torch().bool()
        pdf[unvalid] = 0

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
        output = torch.ones_like(xsurf_valid) * 0.1
        return output
