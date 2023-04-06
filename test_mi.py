import torch
import drjit as dr

a = torch.tensor([1.0, 2.0, 3.0])
a = a.cuda()
a.requires_grad = True

@dr.wrap_ad(source='torch', target='drjit')
def func(a):
    return dr.cos(a)


b = func(a)
print(b)


b.sum().backward()

print(b)
print(a.grad)
