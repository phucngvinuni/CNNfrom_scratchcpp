#include "relu_activation.h"
#include <algorithm>

std::vector<float> ReLUActivation::forward(const std::vector<float>& input) {
    last_input = input; // Store input for backpropagation
    std::vector<float> output(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = std::max(0.0f, input[i]);
    }
    return output;
}

std::vector<float> ReLUActivation::backward(const std::vector<float>& grad_output) {
    std::vector<float> grad_input(grad_output.size());
    for (size_t i = 0; i < grad_output.size(); ++i) {
        grad_input[i] = (last_input[i] > 0) ? grad_output[i] : 0.0f;
    }
    return grad_input;
}