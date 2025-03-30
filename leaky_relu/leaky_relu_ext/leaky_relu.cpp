#include <torch/extension.h>

torch::Tensor leaky_relu_forward_cpu(torch::Tensor input, double negative_slope);
torch::Tensor leaky_relu_forward_cuda(torch::Tensor input, double negative_slope);


torch::Tensor leaky_relu_forward(torch::Tensor input, double slope) {
    if (input.device().is_cuda()) {
        return leaky_relu_forward_cuda(input, slope);
    } else {
        return leaky_relu_forward_cpu(input, slope);
    }
}

// Python binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("leaky_relu", &leaky_relu_forward, "Custom Leaky ReLU (CPU+CUDA)");
}
