#ifndef MAXPOOL_LAYER_H
#define MAXPOOL_LAYER_H

#include "../utils/tensor.h"

class MaxPoolingLayer {
public:
    // Constructor: Takes the size of the pooling window
    MaxPoolingLayer(int pool_size);

    // Forward pass: Applies max pooling to the input tensor
    Tensor forward(const Tensor& input);

    // Backward pass: Computes gradients for the pooling layer
    Tensor backward(const Tensor& grad_output);
private:
    int pool_size; // Size of the pooling window (e.g., 2x2 or 3x3)
};

#endif // MAXPOOL_LAYER_H
