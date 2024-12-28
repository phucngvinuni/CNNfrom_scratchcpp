#ifndef RELU_LAYER_H
#define RELU_LAYER_H

#include "../utils/tensor.h"

class ReLULayer {
public:
    Tensor forward(const Tensor& input);
    // Example for ReLULayer backward function
    Tensor ReLULayer::backward(const Tensor& grad_output) {
    // Implement backward logic here
}

};

#endif // RELU_LAYER_H
