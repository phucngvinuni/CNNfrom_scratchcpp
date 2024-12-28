#include "conv_layer.h"
#include <cmath>

// Constructor
ConvLayer::ConvLayer(int input_channels, int output_channels, int kernel_size, int stride, int padding)
    : input_channels(input_channels), output_channels(output_channels), kernel_size(kernel_size),
      stride(stride), padding(padding) {
    // Initialize kernels and biases with random values
    for (int i = 0; i < output_channels; ++i) {
        kernels.push_back(Tensor(kernel_size, kernel_size, input_channels));
        biases.push_back(0.0f); // Initial biases set to 0
    }
}

// Forward pass
Tensor ConvLayer::forward(const Tensor& input) {
    int output_height = (input.getHeight() - kernel_size + 2 * padding) / stride + 1;
    int output_width = (input.getWidth() - kernel_size + 2 * padding) / stride + 1;

    Tensor output(output_height, output_width, output_channels);

    // Perform convolution
    for (int oc = 0; oc < output_channels; ++oc) {
        for (int h = 0; h < output_height; ++h) {
            for (int w = 0; w < output_width; ++w) {
                float value = 0.0f;
                for (int ic = 0; ic < input.getDepth(); ++ic) {
                    for (int kh = 0; kh < kernel_size; ++kh) {
                        for (int kw = 0; kw < kernel_size; ++kw) {
                            int ih = h * stride + kh - padding;
                            int iw = w * stride + kw - padding;
                            if (ih >= 0 && iw >= 0 && ih < input.getHeight() && iw < input.getWidth()) {
                                value += input.at(ih, iw, ic) * kernels[oc].at(kh, kw, ic);
                            }
                        }
                    }
                }
                output.at(h, w, oc) = value + biases[oc];
            }
        }
    }

    return output;
}

// Backward pass
Tensor ConvLayer::backward(const Tensor& input, const Tensor& grad_output) {
    Tensor grad_input(input.getHeight(), input.getWidth(), input.getDepth());
    grad_kernels = std::vector<Tensor>(output_channels, Tensor(kernel_size, kernel_size, input_channels));
    grad_biases = std::vector<float>(output_channels, 0.0f);

    // Compute gradients for input, kernels, and biases
    for (int oc = 0; oc < output_channels; ++oc) {
        for (int h = 0; h < grad_output.getHeight(); ++h) {
            for (int w = 0; w < grad_output.getWidth(); ++w) {
                float grad_val = grad_output.at(h, w, oc);
                grad_biases[oc] += grad_val;

                for (int ic = 0; ic < input_channels; ++ic) {
                    for (int kh = 0; kh < kernel_size; ++kh) {
                        for (int kw = 0; kw < kernel_size; ++kw) {
                            int ih = h * stride + kh - padding;
                            int iw = w * stride + kw - padding;

                            // Update gradients for kernels
                            if (ih >= 0 && iw >= 0 && ih < input.getHeight() && iw < input.getWidth()) {
                                grad_kernels[oc].at(kh, kw, ic) += input.at(ih, iw, ic) * grad_val;
                                grad_input.at(ih, iw, ic) += kernels[oc].at(kh, kw, ic) * grad_val;
                            }
                        }
                    }
                }
            }
        }
    }

    return grad_input;
}

// Accessor methods

std::vector<Tensor>& ConvLayer::getKernels() {
    return kernels;
}

std::vector<float>& ConvLayer::getBiases() { return biases; }
std::vector<Tensor>& ConvLayer::getGradKernels() { return grad_kernels; }
std::vector<float>& ConvLayer::getGradBiases() { return grad_biases; }
