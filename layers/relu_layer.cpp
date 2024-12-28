#include "relu_layer.h"
#include <algorithm>

Tensor ReLULayer::forward(const Tensor& input) {
    Tensor output = input;
    for (int h = 0; h < input.getHeight(); ++h) {
        for (int w = 0; w < input.getWidth(); ++w) {
            for (int d = 0; d < input.getDepth(); ++d) {
                output.at(h, w, d) = std::max(0.0f, input.at(h, w, d));
            }
        }
    }
    return output;
}

Tensor ReLULayer::backward(const Tensor& grad_output) {
    Tensor grad_input = grad_output;
    for (int h = 0; h < grad_output.getHeight(); ++h) {
        for (int w = 0; w < grad_output.getWidth(); ++w) {
            for (int d = 0; d < grad_output.getDepth(); ++d) {
                grad_input.at(h, w, d) = grad_output.at(h, w, d) > 0 ? grad_output.at(h, w, d) : 0.0f;
            }
        }
    }
    return grad_input;
}