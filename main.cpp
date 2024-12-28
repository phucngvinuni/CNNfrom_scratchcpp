#include "layers/conv_layer.h"
#include "layers/relu_layer.h"
#include "layers/maxpool_layer.h"
#include "layers/flatten_layer.h"
#include "layers/fully_connected_layer.h"
#include "utils/mnist_loader.h"
#include "utils/loss.h"
#include "utils/optimizer.h"
#include <iostream>

using namespace std;

int main() {
    // Load MNIST dataset
    auto train_images = MNISTLoader::load_images("D:/CNNfrom_scratchcpp/train-images-idx3-ubyte", 60000, 28, 28, 1);
    auto train_labels = MNISTLoader::load_labels("D:/CNNfrom_scratchcpp/train-labels-idx1-ubyte", 60000);

    // Define layers
    ConvLayer conv1(1, 32, 3, 1, 1);
    ReLULayer relu1;
    MaxPoolingLayer pool1(2);
    ConvLayer conv2(32, 64, 3, 1, 1);
    ReLULayer relu2;
    MaxPoolingLayer pool2(2);
    FlattenLayer flatten;
    FullyConnectedLayer fc1(64 * 7 * 7, 128);
    FullyConnectedLayer fc2(128, 10);

    // Training loop
    float learning_rate = 0.01;
    SGD optimizer;
    for (int epoch = 0; epoch < 10; ++epoch) {
    float total_loss = 0.0f;

    for (size_t i = 0; i < train_images.size(); ++i) {
        // Forward pass
        Tensor out1 = conv1.forward(train_images[i]);
        Tensor out2 = relu1.forward(out1);
        Tensor out3 = pool1.forward(out2);
        Tensor out4 = conv2.forward(out3);
        Tensor out5 = relu2.forward(out4);
        Tensor out6 = pool2.forward(out5);
        std::vector<float> flattened = flatten.forward(out6);
        std::vector<float> fc_out1 = fc1.forward(flattened);
        std::vector<float> fc_out2 = fc2.forward(fc_out1);

        // Calculate loss
        float loss = Loss::cross_entropy(fc_out2, train_labels[i]);
        total_loss += loss;

        // Backward pass
        std::vector<float> grad_fc2 = fc2.backward(fc_out2);
        std::vector<float> grad_fc1 = fc1.backward(grad_fc2);
        Tensor grad_pool2 = flatten.backward(grad_fc1, out6);
        Tensor grad_relu2 = relu2.backward(grad_pool2);
        Tensor grad_conv2 = conv2.backward(out3, grad_relu2);
        Tensor grad_pool1 = pool1.backward(grad_conv2);
        Tensor grad_relu1 = relu1.backward(grad_pool1);
        conv1.backward(train_images[i], grad_relu1);

        // Update weights
        optimizer.update(conv1.getKernels(), conv1.biases, conv1.getGradKernels(), conv1.grad_biases, learning_rate);
        optimizer.update(conv2.getKernels(), conv2.biases, conv2.getGradKernels(), conv2.grad_biases, learning_rate);
        optimizer.update(fc1.weights, fc1.biases, fc1.grad_weights, fc1.grad_biases, learning_rate);
        optimizer.update(fc2.weights, fc2.biases, fc2.grad_weights, fc2.grad_biases, learning_rate);
    }

    std::cout << "Epoch " << epoch + 1 << ", Loss: " << total_loss / train_images.size() << std::endl;
}


    return 0;
}
