#ifndef RELU_LAYER_H
#define RELU_LAYER_H

#include "../utils/tensor.h"

class ReLULayer {
public:
    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& grad_output);  // Just the declaration, no implementation here
};

#endif // RELU_LAYER_H