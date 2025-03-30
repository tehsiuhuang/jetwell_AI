from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="my_ops",
    ext_modules=[
        CUDAExtension(
            name="my_ops",
            sources=[
                "leaky_relu.cpp",
                "leaky_relu_cpu.cpp",
                "leaky_relu_cuda.cu"
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)

