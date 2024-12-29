// fully_connected_layer.h

#ifndef FULLY_CONNECTED_LAYER_H
#define FULLY_CONNECTED_LAYER_H

#include <vector>

class FullyConnectedLayer {
public:
    FullyConnectedLayer(int input_size, int output_size);

    std::vector<float> forward(const std::vector<float>& input);
    std::vector<float> backward(const std::vector<float>& grad_output);
    void update_weights(float learning_rate);
    void zero_gradients();

private:
    int input_size, output_size;
    std::vector<std::vector<float>> weights;
    std::vector<float> biases;
    std::vector<std::vector<float>> grad_weights;
    std::vector<float> grad_biases;
    std::vector<float> last_input;
};

#endif // FULLY_CONNECTED_LAYER_H