#ifndef FULLY_CONNECTED_LAYER_H
#define FULLY_CONNECTED_LAYER_H

#include <vector>

class FullyConnectedLayer {
public:
    FullyConnectedLayer(int input_size, int output_size);

    // Forward pass: Computes output given the input
    std::vector<float> forward(const std::vector<float>& input);

    // Backward pass: Computes gradients for weights, biases, and input
    std::vector<float> backward(const std::vector<float>& grad_output);

    // Update weights and biases using computed gradients
    void update_weights(float learning_rate);

    // Getters
    int input_size, output_size;
    std::vector<std::vector<float>> weights;      // Weight matrix
    std::vector<float> biases;                    // Bias vector
    std::vector<std::vector<float>> grad_weights; // Gradient of weights
    std::vector<float> grad_biases;               // Gradient of biases
    std::vector<float> last_input;                // Cache input for backprop
};

#endif // FULLY_CONNECTED_LAYER_H
