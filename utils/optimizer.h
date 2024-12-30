#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <vector>
#include "../utils/tensor.h"
#include <algorithm>
class SGD {
public:
    // Constructor with default momentum value
    SGD(float momentum = 0.0f) : momentum(momentum) {}

    void update(std::vector<std::vector<float>>& weights,
                std::vector<float>& biases,
                const std::vector<std::vector<float>>& grad_weights,
                const std::vector<float>& grad_biases,
                float learning_rate);

    void update_conv(std::vector<Tensor>& kernels,
                    std::vector<float>& biases,
                    const std::vector<Tensor>& grad_kernels,
                    const std::vector<float>& grad_biases,
                    float learning_rate);

    float momentum;
};

#endif // OPTIMIZER_H