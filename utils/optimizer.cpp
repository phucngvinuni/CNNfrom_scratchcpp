#include "optimizer.h"

void SGD::update(std::vector<std::vector<float>>& weights,
                std::vector<float>& biases,
                const std::vector<std::vector<float>>& grad_weights,
                const std::vector<float>& grad_biases,
                float learning_rate) {
    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[i].size(); ++j) {
            weights[i][j] -= learning_rate * grad_weights[i][j];
        }
        biases[i] -= learning_rate * grad_biases[i];
    }
}

void SGD::update_conv(std::vector<Tensor>& kernels,
                     std::vector<float>& biases,
                     const std::vector<Tensor>& grad_kernels,
                     const std::vector<float>& grad_biases,
                     float learning_rate) {
    for (size_t i = 0; i < kernels.size(); ++i) {
        for (int h = 0; h < kernels[i].getHeight(); ++h) {
            for (int w = 0; w < kernels[i].getWidth(); ++w) {
                for (int d = 0; d < kernels[i].getDepth(); ++d) {
                    kernels[i].at(h, w, d) -= learning_rate * grad_kernels[i].at(h, w, d);
                }
            }
        }
        biases[i] -= learning_rate * grad_biases[i];
    }
}