// loss.cpp
#include "loss.h"
#include <cmath>
#include <algorithm>
#include <numeric>

std::vector<float> Loss::softmax(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    float max_val = *std::max_element(input.begin(), input.end());
    float sum_exp = 0.0f;

    // Compute exponentials with numerical stability
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum_exp += output[i];
    }

    // Normalize
    for (size_t i = 0; i < output.size(); ++i) {
        output[i] /= sum_exp;
    }

    return output;
}

float Loss::cross_entropy(const std::vector<float>& predicted, int true_label) {
    std::vector<float> softmax_output = softmax(predicted);
    return -std::log(std::max(softmax_output[true_label], 1e-7f));
}

std::vector<float> Loss::cross_entropy_gradient(const std::vector<float>& predicted, int true_label) {
    std::vector<float> softmax_output = softmax(predicted);
    softmax_output[true_label] -= 1.0f;
    return softmax_output;
}