// fully_connected_layer.cpp
#include "fully_connected_layer.h"
#include <cmath>
#include <random>
#include <algorithm>

FullyConnectedLayer::FullyConnectedLayer(int input_size, int output_size)
    : input_size(input_size), output_size(output_size) {

    // He initialization
    float stddev = std::sqrt(2.0f / input_size);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, stddev);

    weights.resize(output_size, std::vector<float>(input_size));
    biases.resize(output_size, 0.0f);

    for (int o = 0; o < output_size; ++o) {
        for (int i = 0; i < input_size; ++i) {
            weights[o][i] = dist(gen);
        }
    }

    grad_weights.resize(output_size, std::vector<float>(input_size, 0.0f));
    grad_biases.resize(output_size, 0.0f);
}

std::vector<float> FullyConnectedLayer::forward(const std::vector<float>& input) {
    last_input = input;
    std::vector<float> output(output_size, 0.0f);

    // Compute output: y = Wx + b
    for (int o = 0; o < output_size; ++o) {
        output[o] = biases[o];
        for (int i = 0; i < input_size; ++i) {
            output[o] += weights[o][i] * input[i];
        }
    }
    return output;
}

std::vector<float> FullyConnectedLayer::backward(const std::vector<float>& grad_output) {
    std::vector<float> grad_input(input_size, 0.0f);

    // Compute gradients and accumulate over the batch
    for (int o = 0; o < output_size; ++o) {
        for (int i = 0; i < input_size; ++i) {
            grad_weights[o][i] += grad_output[o] * last_input[i];  // Accumulate gradient
            grad_input[i] += grad_output[o] * weights[o][i];
        }
        grad_biases[o] += grad_output[o];  // Accumulate gradient
    }

    return grad_input;
}

void FullyConnectedLayer::zero_gradients() {
    for (auto& grad_row : grad_weights) {
        std::fill(grad_row.begin(), grad_row.end(), 0.0f);
    }
    std::fill(grad_biases.begin(), grad_biases.end(), 0.0f);
}

void FullyConnectedLayer::update_weights(float learning_rate) {
    for (int o = 0; o < output_size; ++o) {
        for (int i = 0; i < input_size; ++i) {
            weights[o][i] -= learning_rate * grad_weights[o][i];
            // Reset gradients after update
            grad_weights[o][i] = 0.0f;
        }
        biases[o] -= learning_rate * grad_biases[o];
        grad_biases[o] = 0.0f;
    }
}