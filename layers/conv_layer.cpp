#include "conv_layer.h"
#include <cmath>
#include <random>

ConvLayer::ConvLayer(int input_channels, int output_channels, int kernel_size, int stride, int padding)
    : input_channels(input_channels), output_channels(output_channels), kernel_size(kernel_size),
      stride(stride), padding(padding) {
    
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Calculate initialization scale (Xavier initialization)
    float scale = std::sqrt(2.0f / (input_channels * kernel_size * kernel_size));
    std::normal_distribution<float> dist(0.0f, scale);

    // Initialize kernels and biases
    kernels.resize(output_channels);
    for (int i = 0; i < output_channels; ++i) {
        kernels[i] = Tensor(kernel_size, kernel_size, input_channels);
        for (int h = 0; h < kernel_size; ++h) {
            for (int w = 0; w < kernel_size; ++w) {
                for (int c = 0; c < input_channels; ++c) {
                    kernels[i].at(h, w, c) = dist(gen);
                }
            }
        }
        biases.push_back(0.0f);
    }

    // Initialize gradient storage
    grad_kernels.resize(output_channels, Tensor(kernel_size, kernel_size, input_channels));
    grad_biases.resize(output_channels, 0.0f);
}

Tensor ConvLayer::forward(const Tensor& input) {
    last_input = input;  // Store input for backward pass

    int output_height = (input.getHeight() - kernel_size + 2 * padding) / stride + 1;
    int output_width = (input.getWidth() - kernel_size + 2 * padding) / stride + 1;

    Tensor output(output_height, output_width, output_channels);

    // Perform convolution
    for (int oc = 0; oc < output_channels; ++oc) {
        for (int h = 0; h < output_height; ++h) {
            for (int w = 0; w < output_width; ++w) {
                float value = 0.0f;
                for (int ic = 0; ic < input_channels; ++ic) {
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

Tensor ConvLayer::backward(const Tensor& grad_output) {
    Tensor grad_input(last_input.getHeight(), last_input.getWidth(), last_input.getDepth());

    // Compute gradients and accumulate over the batch
    for (int oc = 0; oc < output_channels; ++oc) {
        for (int h = 0; h < grad_output.getHeight(); ++h) {
            for (int w = 0; w < grad_output.getWidth(); ++w) {
                float grad_val = grad_output.at(h, w, oc);

                grad_biases[oc] += grad_val;  // Accumulate gradient for bias

                for (int ic = 0; ic < input_channels; ++ic) {
                    for (int kh = 0; kh < kernel_size; ++kh) {
                        for (int kw = 0; kw < kernel_size; ++kw) {
                            int ih = h * stride + kh - padding;
                            int iw = w * stride + kw - padding;

                            // Update gradients for kernels and input
                            if (ih >= 0 && iw >= 0 && ih < last_input.getHeight() && iw < last_input.getWidth()) {
                                grad_kernels[oc].at(kh, kw, ic) += last_input.at(ih, iw, ic) * grad_val;  // Accumulate
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

void ConvLayer::update_weights(float learning_rate) {
    // Update kernels and biases using accumulated gradients
    for (int oc = 0; oc < output_channels; ++oc) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                for (int ic = 0; ic < input_channels; ++ic) {
                    kernels[oc].at(kh, kw, ic) -= learning_rate * grad_kernels[oc].at(kh, kw, ic);
                    grad_kernels[oc].at(kh, kw, ic) = 0.0f;  // Reset gradient after update
                }
            }
        }
        biases[oc] -= learning_rate * grad_biases[oc];
        grad_biases[oc] = 0.0f;  // Reset gradient after update
    }
}

void ConvLayer::zero_gradients() {
    for (auto& grad_kernel : grad_kernels) {
        grad_kernel.zero();  // Use Tensor's zero() method
    }
    std::fill(grad_biases.begin(), grad_biases.end(), 0.0f);
}
