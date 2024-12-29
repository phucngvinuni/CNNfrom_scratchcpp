#ifndef MAXPOOL_LAYER_H
#define MAXPOOL_LAYER_H

#include "../utils/tensor.h"

class MaxPoolingLayer {
public:
    MaxPoolingLayer(int pool_size);

    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& grad_output);

private:
    int pool_size;
    Tensor indices;     // Store indices of max values
    Tensor input_shape; // Store input shape for backward
};

#endif // MAXPOOL_LAYER_H