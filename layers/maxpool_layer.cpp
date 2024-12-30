#include "maxpool_layer.h"
#include <algorithm>
#include <limits> // For numeric limits

MaxPoolingLayer::MaxPoolingLayer(int pool_size)
    : pool_size(pool_size),
      indices(),
      input_shape()
{}

Tensor MaxPoolingLayer::forward(const Tensor& input) {
    input_shape = input;
    int out_height = input.getHeight() / pool_size;
    int out_width = input.getWidth() / pool_size;
    Tensor output(out_height, out_width, input.getDepth());
    indices = Tensor(out_height, out_width, input.getDepth());
    // Perform max pooling for each depth channel
    for (int d = 0; d < input.getDepth(); ++d) {
        for (int h = 0; h < out_height; ++h) {
            for (int w = 0; w < out_width; ++w) {
                float max_val = std::numeric_limits<float>::lowest();
                int max_h = -1;
                int max_w = -1;

                for (int ph = 0; ph < pool_size; ++ph) {
                    for (int pw = 0; pw < pool_size; ++pw) {
                        int input_h = h * pool_size + ph;
                        int input_w = w * pool_size + pw;

                        if (input_h < input.getHeight() && input_w < input.getWidth()) {
                            float val = input.at(input_h, input_w, d);
                            if (val > max_val) {
                                max_val = val;
                                max_h = input_h;
                                max_w = input_w;
                            }
                        }
                    }
                }
                output.at(h, w, d) = max_val;
                // Store index of max value
                indices.at(h, w, d) = max_h * input.getWidth() + max_w;
            }
        }
    }

    return output;
}

Tensor MaxPoolingLayer::backward(const Tensor& grad_output) {
    Tensor grad_input(input_shape.getHeight(), input_shape.getWidth(), input_shape.getDepth());
    grad_input.zero(); // Use zero() method to reset data
   for (size_t i = 0; i < grad_input.data.size(); ++i) {
        grad_input.data[i] = 0.0f;
    }

    // Distribute gradients to where the max values came from
    for (int d = 0; d < grad_output.getDepth(); ++d) {
        for (int h = 0; h < grad_output.getHeight(); ++h) {
            for (int w = 0; w < grad_output.getWidth(); ++w) {
                int index = static_cast<int>(indices.at(h, w, d));
                int max_h = index / input_shape.getWidth();
                int max_w = index % input_shape.getWidth();
                grad_input.at(max_h, max_w, d) += grad_output.at(h, w, d);
            }
        }
    }

    return grad_input;
}

