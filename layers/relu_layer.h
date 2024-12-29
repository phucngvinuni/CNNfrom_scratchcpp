#ifndef RELU_LAYER_H
#define RELU_LAYER_H

#include "../utils/tensor.h"

class ReLULayer {
public:
    ReLULayer();  // Declare default constructor
    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& grad_output);

private:
    Tensor last_input; // Store input for backward pass
};

#endif // RELU_LAYER_H