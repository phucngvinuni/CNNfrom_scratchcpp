// conv_layer.h

#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include "../utils/tensor.h"
#include <vector>

class ConvLayer {
public:
    ConvLayer(int input_channels, int output_channels, int kernel_size, int stride, int padding);

    // Forward and backward methods
    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& grad_output);

    // Weight update and gradient reset methods
    void update_weights(float learning_rate);
    void zero_gradients();

    // Accessor methods for weights and gradients
    std::vector<Tensor>& getKernels();       // Declaration added
    std::vector<float>& getBiases();         // Declaration added
    std::vector<Tensor>& getGradKernels();   // Declaration added
    std::vector<float>& getGradBiases();     // Declaration added

private:
    int input_channels;
    int output_channels;
    int kernel_size;
    int stride;
    int padding;

    std::vector<Tensor> kernels;
    std::vector<float> biases;
    std::vector<Tensor> grad_kernels;
    std::vector<float> grad_biases;

    Tensor last_input;  // Store input for backward pass
};

#endif // CONV_LAYER_H