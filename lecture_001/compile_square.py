# TORCH_LOGS="output_code" python compile_square.py > compiler_square_triton 2>&1
import torch

def square(a):
    return torch.square(a)

opt_square = torch.compile(square)
opt_square(torch.randn(10000, 10000).cuda())