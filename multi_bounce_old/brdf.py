import mitsuba as mi
import torch
import math


class BRDF():
    def eval(self):
        NotImplemented

    def pdf(self):
        NotImplemented

    def sample(self):
        NotImplemented


class Diffuse(BRDF):
    def __init__(self, albedo, si, sampler):
        self.albedo = albedo  # torch.tensor() [N,3]
        self.si = si  # mitsuba intersection
        self.sampler = sampler  # mitsuba sampler

    def eval(self):
        return self.albedo / math.pi

    def pdf(self, wi_local):
        pdf = mi.warp.square_to_cosine_hemisphere_pdf(wi_local).torch()[:, None]  # [N,1]
        return torch.clip(pdf, 0, None)

    def sample(self):
        wi_local = mi.warp.square_to_cosine_hemisphere(self.sampler.next_2d())
        wi_world = self.si.sh_frame.to_world(wi_local)
        cos_term = self.si.sh_frame.cos_theta(wi_local).torch()[:, None]  # [N,1]
        cos_term = torch.clip(cos_term, 0, None)  # consider only reflection
        eval = self.eval()
        pdf = self.pdf(wi_local)
        output = cos_term * eval / pdf
        return torch.where(pdf == 0, 0, output), wi_world, pdf



