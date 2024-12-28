#ifndef FLATTEN_LAYER_H
#define FLATTEN_LAYER_H

#include "../utils/tensor.h"
#include <vector>

class FlattenLayer {
public:
    // Forward pass: Flattens a 3D tensor into a 1D vector
    std::vector<float> forward(const Tensor& input);

    // Backward pass: Reshapes a gradient vector into the original tensor dimensions
    Tensor backward(const std::vector<float>& grad_output, const Tensor& input_shape);
};

#endif // FLATTEN_LAYER_H
