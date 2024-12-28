#include "maxpool_layer.h"
#include <algorithm>
#include <limits> // For numeric limits

MaxPoolingLayer::MaxPoolingLayer(int pool_size) : pool_size(pool_size) {}

Tensor MaxPoolingLayer::forward(const Tensor& input) {
    // Calculate output dimensions after pooling
    int out_height = input.getHeight() / pool_size;
    int out_width = input.getWidth() / pool_size;
    Tensor output(out_height, out_width, input.getDepth());

    // Perform max pooling for each depth channel
    for (int d = 0; d < input.getDepth(); ++d) {
        for (int h = 0; h < out_height; ++h) {
            for (int w = 0; w < out_width; ++w) {
                // Initialize max value to the smallest possible float
                float max_val = std::numeric_limits<float>::lowest();

                // Iterate over the pooling window
                for (int ph = 0; ph < pool_size; ++ph) {
                    for (int pw = 0; pw < pool_size; ++pw) {
                        int input_h = h * pool_size + ph;
                        int input_w = w * pool_size + pw;

                        // Ensure we are within input bounds
                        if (input_h < input.getHeight() && input_w < input.getWidth()) {
                            max_val = std::max(max_val, input.at(input_h, input_w, d));
                        }
                    }
                }

                // Assign the max value to the output tensor
                output.at(h, w, d) = max_val;
            }
        }
    }

    return output;
}


Tensor MaxPoolingLayer::backward(const Tensor& grad_output) {
    // Initialize the gradient tensor with zeros
    Tensor grad_input(grad_output.getHeight() * pool_size, grad_output.getWidth() * pool_size, grad_output.getDepth());

    // Iterate over the gradient tensor
    for (int d = 0; d < grad_output.getDepth(); ++d) {
        for (int h = 0; h < grad_output.getHeight(); ++h) {
            for (int w = 0; w < grad_output.getWidth(); ++w) {
                // Find the index of the max value in the pooling window
                float max_val = std::numeric_limits<float>::lowest();
                int max_h = -1, max_w = -1;

                for (int ph = 0; ph < pool_size; ++ph) {
                    for (int pw = 0; pw < pool_size; ++pw) {
                        int input_h = h * pool_size + ph;
                        int input_w = w * pool_size + pw;

                        // Ensure we are within input bounds
                        if (input_h < grad_input.getHeight() && input_w < grad_input.getWidth()) {
                            float val = grad_output.at(h, w, d);
                            if (val > max_val) {
                                max_val = val;
                                max_h = input_h;
                                max_w = input_w;
                            }
                        }
                    }
                }

                // Assign the gradient to the index of the max value
                if (max_h != -1 && max_w != -1) {
                    grad_input.at(max_h, max_w, d) = grad_output.at(h, w, d);
                }
            }
        }
    }

    return grad_input;
}
