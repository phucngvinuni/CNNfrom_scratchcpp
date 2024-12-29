// conv_layer.h

#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include "../utils/tensor.h"
#include <vector>

class ConvLayer {
public:
    ConvLayer(int input_channels, int output_channels, int kernel_size, int stride, int padding);

    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& grad_output);

    void update_weights(float learning_rate);
    void zero_gradients();
    std::vector<Tensor>& getKernels();

    std::vector<float>& getBiases();

    std::vector<Tensor>& getGradKernels();

    std::vector<float>& getGradBiases();

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

    Tensor last_input;


};

#endif // CONV_LAYER_H