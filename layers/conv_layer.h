#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include "../utils/tensor.h"
#include <vector>

class ConvLayer {
public:
    ConvLayer(int input_channels, int output_channels, int kernel_size, int stride, int padding);

    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& input, const Tensor& grad_output);

    // Accessors for weights, biases, and gradients
    std::vector<Tensor>& getKernels();
    std::vector<float>& getBiases();
    std::vector<Tensor>& getGradKernels();
    std::vector<float>& getGradBiases();

    int input_channels;
    int output_channels;
    int kernel_size;
    int stride;
    int padding;

    std::vector<Tensor> kernels;       // Convolution kernels (weights)
    std::vector<float> biases;         // Biases
    std::vector<Tensor> grad_kernels;  // Gradients for kernels
    std::vector<float> grad_biases;    // Gradients for biases
    
};

#endif // CONV_LAYER_H
