#include "flatten_layer.h"

// Forward pass: Flatten a 3D tensor into a 1D vector
std::vector<float> FlattenLayer::forward(const Tensor& input) {
    std::vector<float> output;

    // Iterate through all dimensions of the tensor and push values into the 1D vector
    for (int d = 0; d < input.getDepth(); ++d) {
        for (int h = 0; h < input.getHeight(); ++h) {
            for (int w = 0; w < input.getWidth(); ++w) {
                output.push_back(input.at(h, w, d));
            }
        }
    }
    return output;
}

// Backward pass: Reshape a 1D gradient vector into the original 3D tensor dimensions
Tensor FlattenLayer::backward(const std::vector<float>& grad_output, const Tensor& input_shape) {
    Tensor grad_input(input_shape.getHeight(), input_shape.getWidth(), input_shape.getDepth());
    int index = 0;

    // Map the 1D gradient vector back into the 3D tensor structure
    for (int d = 0; d < input_shape.getDepth(); ++d) {
        for (int h = 0; h < input_shape.getHeight(); ++h) {
            for (int w = 0; w < input_shape.getWidth(); ++w) {
                grad_input.at(h, w, d) = grad_output[index++];
            }
        }
    }

    return grad_input;
}
