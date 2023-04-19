import mitsuba as mi
import torch
import math
import drjit as dr

mi.set_variant("cuda_ad_rgb")


class BSDF:
    def eval_pdf(self,
                 si: mi.Interaction3f,
                 wo: mi.Vector3f,
                 active: mi.Mask):
        return self.eval(si, wo, active), self.pdf(si, wo, active)

    def eval_pdf_sample(self,
                        si: mi.Interaction3f,
                        wo: mi.Vector3f,
                        sample1: mi.Float,
                        sample2: mi.Point2f,
                        active: mi.Mask):
        e_val, pdf_val = self.eval_pdf(si, wo, active)
        bs, bsdf_weight = self.sample(si, sample1, sample2, active)
        return e_val, pdf_val, bs, bsdf_weight


class Diffuse(BSDF):

    def __init__(self, diffuse_model):
        self.diffuse_model = diffuse_model

    def sample(self,
               si: mi.Interaction3f,
               sample1: mi.Float,
               sample2: mi.Point2f,
               active: mi.Mask):
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        bs = dr.zeros(mi.BSDFSample3f)

        active = active & (cos_theta_i > 0)
        bs.wo = mi.warp.square_to_cosine_hemisphere(sample2)
        bs.pdf = mi.warp.square_to_cosine_hemisphere_pdf(bs.wo)
        bs.eta = 1.0
        bs.sampled_type = mi.BSDFFlags.DiffuseReflection
        bs.sampled_component = 0

        albedo = self.diffuse_model(si, active)

        return bs, albedo


    def eval(self,
             si: mi.Interaction3f,
             wo: mi.Vector3f,
             active: mi.Mask):
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)

        active = active & (cos_theta_i > 0) & (cos_theta_o > 0)
        albedo = self.diffuse_model(si, active)
        value = albedo / math.pi * cos_theta_o.torch()[:, None]  # [N, 3]
        return value

    def pdf(self,
            si: mi.Interaction3f,
            wo: mi.Vector3f,
            active: mi.Mask):
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)

        pdf = mi.warp.square_to_cosine_hemisphere_pdf(wo)
        pdf = dr.select((cos_theta_i > 0) & (cos_theta_o > 0), pdf, 0)
        return pdf.torch()


class Mlp_trivial():
    def __call__(self, xsurf_valid: torch.Tensor):
        # always return [0.8, 0.8, 0.8]
        output = torch.ones_like(xsurf_valid) * 0.8
        return output


class Diffuse_model_trivial():
    def __init__(self):
        self.diffuse_model = Mlp_trivial()

    def __call__(self, si: mi.Interaction3f, active: mi.Mask):
        active = active.torch().bool()
        N = active.shape[0]
        xsurf_valid = si.p.torch()[active]
        diffuse_valid = self.diffuse_model(xsurf_valid)
        diffuse = torch.zeros(N, 3, dtype=torch.float32).cuda()
        diffuse[active] = diffuse_valid
        diffuse = torch.clip(diffuse, 0, None)
        return diffuse


def main():
    pass


if __name__ == '__main__':
    main()
