#include "tensor.h"
#include "tensor.h"

Tensor::Tensor() : height(0), width(0), depth(0), data() {
    // Default constructor initializes an empty tensor
}

Tensor::Tensor(int height, int width, int depth)
    : height(height), width(width), depth(depth), data(height * width * depth, 0.0f) {}


float& Tensor::at(int h, int w, int d) {
    return data[(h * width + w) * depth + d];
}

const float& Tensor::at(int h, int w, int d) const {
    return data[(h * width + w) * depth + d];
}

int Tensor::getHeight() const { return height; }
int Tensor::getWidth() const { return width; }
int Tensor::getDepth() const { return depth; }

void Tensor::zero() {
    std::fill(data.begin(), data.end(), 0.0f);
}

size_t Tensor::getSize() const {
    return data.size();
}

std::vector<float>& Tensor::getData() {
    return data;
}

const std::vector<float>& Tensor::getData() const {
    return data;
}