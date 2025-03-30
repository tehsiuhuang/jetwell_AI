#include <torch/extension.h>

__global__ void leaky_relu_kernel(float* out, const float* in, float slope, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float val = in[i];
        out[i] = val > 0 ? val : val * slope;
    }
}

torch::Tensor leaky_relu_forward_cuda(torch::Tensor input, double slope) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    leaky_relu_kernel<<<blocks, threads>>>(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        static_cast<float>(slope),
        size
    );
    return output;
}

