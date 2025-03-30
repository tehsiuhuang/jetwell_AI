import torch
import os
import importlib.util

from torch.utils.cpp_extension import load

this_dir = os.path.dirname(__file__)

my_ops = load(
    name="my_ops",
    sources=[
        os.path.join(this_dir, "leaky_relu.cpp"),
        os.path.join(this_dir, "leaky_relu_cpu.cpp"),
        os.path.join(this_dir, "leaky_relu_cuda.cu"),
    ],
    verbose=True
)

leaky_relu = my_ops.leaky_relu

