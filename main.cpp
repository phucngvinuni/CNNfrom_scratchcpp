// main.cpp
#include "layers/conv_layer.h"
#include "layers/relu_layer.h"
#include "layers/relu_activation.h" // Include ReLUActivation
#include "layers/maxpool_layer.h"
#include "layers/flatten_layer.h"
#include "layers/fully_connected_layer.h"
#include "utils/mnist_loader.h"
#include "utils/loss.h"
#include <iostream>
#include <algorithm>
#include <vector>
#include <random>
#include <numeric>
#include <cstddef>

int main() {
    // Load dataset
    const int TOTAL_SIZE = 1000;     // Total number of samples to load
    const int TRAIN_SIZE = 800;      // Number of training samples
    const int VALIDATION_SIZE = 200; // Number of validation samples
    const int BATCH_SIZE = 32;

    std::cout << "Loading MNIST dataset...\n";
    auto images = MNISTLoader::load_images("train-images-idx3-ubyte", TOTAL_SIZE, 28, 28, 1);
    auto labels = MNISTLoader::load_labels("train-labels-idx1-ubyte", TOTAL_SIZE);

    // Normalize images to [0,1] range
    for (auto& image : images) {
        for (int h = 0; h < image.getHeight(); ++h) {
            for (int w = 0; w < image.getWidth(); ++w) {
                for (int d = 0; d < image.getDepth(); ++d) {
                    image.at(h, w, d) /= 255.0f;
                }
            }
        }
    }

    // Initialize layers
    ConvLayer conv1(1, 16, 3, 1, 1);          // First convolutional layer with 16 filters
    ReLULayer relu1;
    MaxPoolingLayer pool1(2);

    ConvLayer conv2(16, 32, 3, 1, 1);         // Second convolutional layer with 32 filters
    ReLULayer relu2;
    MaxPoolingLayer pool2(2);

    FlattenLayer flatten;
    FullyConnectedLayer fc1(32 * 7 * 7, 64);  // First fully connected layer
    ReLUActivation relu_fc1;                  // Use ReLUActivation for vectors
    FullyConnectedLayer fc2(64, 10);          // Output layer

    // Learning parameters
    float learning_rate = 0.005f; // Adjusted learning rate
    int num_epochs = 30;          // Increased number of epochs

    // Training setup
    std::vector<size_t> indices(TRAIN_SIZE);
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 gen(rd());

    // Training loop
    for (int epoch = 0; epoch < num_epochs; ++epoch) {

        // Shuffle data indices
        std::shuffle(indices.begin(), indices.end(), gen);

        float total_loss = 0.0f;
        int total_correct = 0;


for (size_t i = 0; i < 5; ++i) {
    std::cout << "Label for sample " << i << ": " << labels[i] << std::endl;
    // Optionally, inspect the sample input
}
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
                Tensor out_conv1 = conv1.forward(images[idx]);
                Tensor out_relu1 = relu1.forward(out_conv1);
                Tensor out_pool1 = pool1.forward(out_relu1);

                Tensor out_conv2 = conv2.forward(out_pool1);
                Tensor out_relu2 = relu2.forward(out_conv2);
                Tensor out_pool2 = pool2.forward(out_relu2);

                std::vector<float> out_flatten = flatten.forward(out_pool2);
                std::vector<float> out_fc1 = fc1.forward(out_flatten);
                std::vector<float> out_relu_fc1 = relu_fc1.forward(out_fc1);
                std::vector<float> out_fc2 = fc2.forward(out_relu_fc1);

                // Compute loss and accuracy
                std::vector<float> softmax_output = Loss::softmax(out_fc2);
                float loss = Loss::cross_entropy(out_fc2, labels[idx]);
                batch_loss += loss;

                int predicted_label = std::distance(softmax_output.begin(),
                                        std::max_element(softmax_output.begin(), softmax_output.end()));
                if (predicted_label == labels[idx]) {
                    batch_correct++;
                }

                // Backward pass
                std::vector<float> grad_loss = Loss::cross_entropy_gradient(out_fc2, labels[idx]);
                std::vector<float> grad_fc2 = fc2.backward(grad_loss);
                std::vector<float> grad_relu_fc1 = relu_fc1.backward(grad_fc2);
                std::vector<float> grad_fc1 = fc1.backward(grad_relu_fc1);
                Tensor grad_flatten = flatten.backward(grad_fc1, out_pool2);
                Tensor grad_pool2 = pool2.backward(grad_flatten);
                Tensor grad_relu2 = relu2.backward(grad_pool2);
                Tensor grad_conv2 = conv2.backward(grad_relu2);
                Tensor grad_pool1 = pool1.backward(grad_conv2);
                Tensor grad_relu1 = relu1.backward(grad_pool1);
                conv1.backward(grad_relu1);
            }

            // Update weights after processing the batch

            // Before weight update
float weight_before = conv1.getKernels()[0].at(0, 0, 0);


            conv1.update_weights(learning_rate);
            conv2.update_weights(learning_rate);
            fc1.update_weights(learning_rate);
            fc2.update_weights(learning_rate);

            total_loss += batch_loss;
            total_correct += batch_correct;
// After updating weights
conv1.update_weights(learning_rate);

// After weight update
float weight_after = conv1.getKernels()[0].at(0, 0, 0);

std::cout << "Weight change: " << weight_before - weight_after << std::endl;
            // Print batch progress
            if (((batch_start / BATCH_SIZE) % 10 == 0) || (batch_end == TRAIN_SIZE)) {
                float batch_accuracy = static_cast<float>(batch_correct) / (batch_end - batch_start) * 100.0f;
                std::cout << "Epoch " << epoch + 1
                          << ", Batch " << (batch_start / BATCH_SIZE)
                          << "/" << (TRAIN_SIZE / BATCH_SIZE)
                          << ", Batch Loss: " << batch_loss / (batch_end - batch_start)
                          << ", Batch Accuracy: " << batch_accuracy << "%"
                          << ", LR: " << learning_rate << "\r" << std::flush;
            }
        }

        // Epoch summary
        float epoch_loss = total_loss / TRAIN_SIZE;
        float epoch_accuracy = static_cast<float>(total_correct) / TRAIN_SIZE * 100.0f;
        std::cout << "\nEpoch " << epoch + 1
                  << ", Loss: " << epoch_loss
                  << ", Accuracy: " << epoch_accuracy << "%" << std::endl;

        // Validation loop
        float val_loss = 0.0f;
        int val_correct = 0;
        for (size_t i = TRAIN_SIZE; i < TOTAL_SIZE; ++i) {
            // Forward pass
            Tensor out_conv1 = conv1.forward(images[i]);
            Tensor out_relu1 = relu1.forward(out_conv1);
            Tensor out_pool1 = pool1.forward(out_relu1);

            Tensor out_conv2 = conv2.forward(out_pool1);
            Tensor out_relu2 = relu2.forward(out_conv2);
            Tensor out_pool2 = pool2.forward(out_relu2);

            std::vector<float> out_flatten = flatten.forward(out_pool2);
            std::vector<float> out_fc1 = fc1.forward(out_flatten);
            std::vector<float> out_relu_fc1 = relu_fc1.forward(out_fc1);
            std::vector<float> out_fc2 = fc2.forward(out_relu_fc1);

            // Compute loss and accuracy
            std::vector<float> softmax_output = Loss::softmax(out_fc2);
            float loss = Loss::cross_entropy(out_fc2, labels[i]);
            val_loss += loss;

            int predicted_label = std::distance(softmax_output.begin(),
                                    std::max_element(softmax_output.begin(), softmax_output.end()));
            if (predicted_label == labels[i]) {
                val_correct++;
            }
        }
        val_loss /= VALIDATION_SIZE;
        float val_accuracy = static_cast<float>(val_correct) / VALIDATION_SIZE * 100.0f;
        std::cout << "Validation Loss: " << val_loss << ", Validation Accuracy: " << val_accuracy << "%" << std::endl;
    }

    return 0;
}