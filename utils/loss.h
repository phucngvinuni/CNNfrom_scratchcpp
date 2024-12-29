// loss.h
#ifndef LOSS_H
#define LOSS_H

#include <vector>

class Loss {
public:
    static std::vector<float> softmax(const std::vector<float>& input);
    static float cross_entropy(const std::vector<float>& predicted, int true_label);
    static std::vector<float> cross_entropy_gradient(const std::vector<float>& predicted, int true_label);
};

#endif