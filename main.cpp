// main.cpp
#include "layers/conv_layer.h"
#include "layers/relu_layer.h"
#include "layers/maxpool_layer.h"
#include "layers/flatten_layer.h"
#include "layers/fully_connected_layer.h"
#include "utils/mnist_loader.h"
#include "utils/loss.h"
#include <iostream>
#include <algorithm>
#include <vector>
#include <random>       // For random_device and mt19937
#include <numeric>      // For iota
#include <cstddef>      // For size_t

int main() {
    // Load dataset
    const int TRAIN_SIZE = 10000;
    const int BATCH_SIZE = 64;
    std::cout << "Loading MNIST dataset...\n";
    auto train_images = MNISTLoader::load_images("train-images-idx3-ubyte", TRAIN_SIZE, 28, 28, 1);
    auto train_labels = MNISTLoader::load_labels("train-labels-idx1-ubyte", TRAIN_SIZE);

    // Normalize images to [0,1] range
    for (auto& image : train_images) {
        for (int h = 0; h < image.getHeight(); ++h) {
            for (int w = 0; w < image.getWidth(); ++w) {
                for (int d = 0; d < image.getDepth(); ++d) {
                    image.at(h, w, d) /= 255.0f;
                }
            }
        }
    }

    // Initialize layers
    ConvLayer conv1(1, 32, 3, 1, 1);
    ReLULayer relu1;
    MaxPoolingLayer pool1(2);
    ConvLayer conv2(32, 64, 3, 1, 1);
    ReLULayer relu2;
    MaxPoolingLayer pool2(2);
    FlattenLayer flatten;
    FullyConnectedLayer fc1(64 * 7 * 7, 128);
    FullyConnectedLayer fc2(128, 10);

    // Hyperparameters
    float learning_rate = 0.0004f;

    // Training setup
    std::vector<size_t> indices(TRAIN_SIZE);
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 gen(rd());

    // Training loop
    for (int epoch = 0; epoch < 50; ++epoch) {
        // Learning rate decay
        float current_lr = learning_rate;
        // Shuffle data
        std::shuffle(indices.begin(), indices.end(), gen);

        float total_loss = 0.0f;
        int total_correct = 0;

        for (size_t batch_start = 0; batch_start < TRAIN_SIZE; batch_start += BATCH_SIZE) {
            size_t batch_end = std::min(batch_start + BATCH_SIZE, (size_t)TRAIN_SIZE);
            float batch_loss = 0.0f;
            int batch_correct = 0;

            // Zero gradients at the start of the batch
            conv1.zero_gradients();
            conv2.zero_gradients();
            fc1.zero_gradients();
            fc2.zero_gradients();

            // Mini-batch processing
            for (size_t i = batch_start; i < batch_end; ++i) {
                size_t idx = indices[i];

                // Forward pass
                Tensor out_conv1 = conv1.forward(train_images[idx]);
                Tensor out_relu1 = relu1.forward(out_conv1);
                Tensor out_pool1 = pool1.forward(out_relu1);
                Tensor out_conv2 = conv2.forward(out_pool1);
                Tensor out_relu2 = relu2.forward(out_conv2);
                Tensor out_pool2 = pool2.forward(out_relu2);
                std::vector<float> out_flatten = flatten.forward(out_pool2);
                std::vector<float> out_fc1 = fc1.forward(out_flatten);
                std::vector<float> out_fc2 = fc2.forward(out_fc1);

                // Compute loss and accuracy
                std::vector<float> softmax_output = Loss::softmax(out_fc2);
                float loss = Loss::cross_entropy(out_fc2, train_labels[idx]);
                batch_loss += loss;

                int predicted_label = std::distance(softmax_output.begin(), 
                                        std::max_element(softmax_output.begin(), softmax_output.end()));
                if (predicted_label == train_labels[idx]) {
                    batch_correct++;
                }

                // Backward pass
                std::vector<float> grad_loss = Loss::cross_entropy_gradient(out_fc2, train_labels[idx]);
                std::vector<float> grad_fc2 = fc2.backward(grad_loss);
                std::vector<float> grad_fc1 = fc1.backward(grad_fc2);
                Tensor grad_flatten = flatten.backward(grad_fc1, out_pool2);
                Tensor grad_pool2 = pool2.backward(grad_flatten);
                Tensor grad_relu2 = relu2.backward(grad_pool2);
                Tensor grad_conv2 = conv2.backward(grad_relu2);
                Tensor grad_pool1 = pool1.backward(grad_conv2);
                Tensor grad_relu1 = relu1.backward(grad_pool1);
                conv1.backward(grad_relu1);
            }

            // Update weights after processing the batch
            conv1.update_weights(current_lr);
            conv2.update_weights(current_lr);
            fc1.update_weights(current_lr);
            fc2.update_weights(current_lr);

            total_loss += batch_loss;
            total_correct += batch_correct;

            // Print batch progress
            if (((batch_start / BATCH_SIZE) % 10 == 0) || (batch_end == TRAIN_SIZE)) {
                std::cout << "Epoch " << epoch + 1
                          << ", Batch " << (batch_start / BATCH_SIZE)
                          << "/" << (TRAIN_SIZE / BATCH_SIZE)
                          << ", Batch Loss: " << batch_loss / (batch_end - batch_start)
                          << ", LR: " << current_lr << "\r" << std::flush;
            }
        }

        // Epoch summary
        float epoch_loss = total_loss / TRAIN_SIZE;
        float epoch_accuracy = static_cast<float>(total_correct) / TRAIN_SIZE * 100.0f;
        std::cout << "\nEpoch " << epoch + 1
                  << ", Loss: " << epoch_loss
                  << ", Accuracy: " << epoch_accuracy << "%" << std::endl;
    }

    return 0;
}