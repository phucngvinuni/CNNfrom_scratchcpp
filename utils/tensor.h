#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <cstddef> // For size_t
class Tensor {
public:
    Tensor();  // Default constructor
    Tensor(int height, int width, int depth);
    float& at(int h, int w, int d);
    const float& at(int h, int w, int d) const;
    int getHeight() const;
    int getWidth() const;
    int getDepth() const;

    void zero();                         // Zero out the tensor data
    size_t getSize() const;              // Get the total number of elements
    std::vector<float>& getData();       // Get reference to data (if necessary)
    const std::vector<float>& getData() const; // For const access

private:
    int height, width, depth;
    std::vector<float> data;
};

#endif // TENSOR_H