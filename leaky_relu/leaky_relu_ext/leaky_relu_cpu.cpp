#include <torch/extension.h>

torch::Tensor leaky_relu_forward_cpu(torch::Tensor input, double negative_slope) {
    auto output = torch::empty_like(input);
    auto input_data = input.data_ptr<float>();
    auto output_data = output.data_ptr<float>();
    auto size = input.numel();

    for (int i = 0; i < size; ++i) {
        output_data[i] = input_data[i] > 0 ? input_data[i] : input_data[i] * negative_slope;
    }
    return output;
}

