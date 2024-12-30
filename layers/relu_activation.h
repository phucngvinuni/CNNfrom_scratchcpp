#ifndef RELU_ACTIVATION_H
#define RELU_ACTIVATION_H

#include <vector>

class ReLUActivation {
public:
    std::vector<float> forward(const std::vector<float>& input);
    std::vector<float> backward(const std::vector<float>& grad_output);


    std::vector<float> last_input; // Store input for backward pass
};

#endif // RELU_ACTIVATION_H