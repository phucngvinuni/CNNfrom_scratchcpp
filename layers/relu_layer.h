#ifndef RELU_LAYER_H
#define RELU_LAYER_H

#include "../utils/tensor.h"

class ReLULayer {
public:
    ReLULayer();  // Default constructor
    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& grad_output);
private:
    Tensor last_input;
};

#endif // RELU_LAYER_H