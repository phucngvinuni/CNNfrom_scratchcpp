#include "layers/conv_layer.h"
#include "layers/relu_layer.h"
#include "layers/maxpool_layer.h"
#include "layers/flatten_layer.h"
#include "layers/fully_connected_layer.h"
#include "utils/mnist_loader.h"
#include "utils/loss.h"
#include "utils/optimizer.h"
#include <iostream>
#include <algorithm>
#include <vector>
#include <random>       // For random_device and mt19937
#include <numeric>      // For iota
#include <cstddef>     // For size_t

using namespace std;

int main() {
    // Load reduced dataset for faster testing
    const int TRAIN_SIZE = 10000;
    const int BATCH_SIZE = 64;  // Increased batch size
    
    cout << "Loading MNIST dataset...\n";
    auto train_images = MNISTLoader::load_images("train-images-idx3-ubyte", TRAIN_SIZE, 28, 28, 1);
    auto train_labels = MNISTLoader::load_labels("train-labels-idx1-ubyte", TRAIN_SIZE);

    // Normalize images to [0,1] range
    for (size_t i = 0; i < train_images.size(); ++i) {
        for (int h = 0; h < train_images[i].getHeight(); ++h) {
            for (int w = 0; w < train_images[i].getWidth(); ++w) {
                for (int d = 0; d < train_images[i].getDepth(); ++d) {
                    train_images[i].at(h, w, d) /= 255.0f;
                }
            }
        }
    }

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    
     ConvLayer conv1(1, 32, 3, 1, 1);  // Increased filters
    ReLULayer relu1;
    MaxPoolingLayer pool1(2);
    ConvLayer conv2(32, 64, 3, 1, 1);  // Increased filters
    ReLULayer relu2;
    MaxPoolingLayer pool2(2);
    FlattenLayer flatten;
    FullyConnectedLayer fc1(64 * 7 * 7, 128);
    FullyConnectedLayer fc2(128, 10);

    // Modified learning parameters
    float learning_rate = 0.001f;  // Adjusted learning rate
    SGD optimizer(0.9f);

    std::vector<size_t> indices(TRAIN_SIZE);
    std::iota(indices.begin(), indices.end(), 0);
    
    for (int epoch = 0; epoch < 10; ++epoch) {
        // Learning rate decay
        float current_lr = learning_rate / (1.0f + epoch * 0.1f);
        
        // Shuffle the training data
        std::shuffle(indices.begin(), indices.end(), gen);
        
        float total_loss = 0.0f;
        int correct = 0;

        for (size_t batch_start = 0; batch_start < TRAIN_SIZE; batch_start += BATCH_SIZE) {
            size_t batch_end = std::min(batch_start + BATCH_SIZE, (size_t)TRAIN_SIZE);
            float batch_loss = 0.0f;
            int batch_correct = 0;

            // Accumulate gradients over batch
            for (size_t i = batch_start; i < batch_end; ++i) {
                size_t idx = indices[i];
                
                // Forward pass
                Tensor out1 = conv1.forward(train_images[idx]);
                Tensor out2 = relu1.forward(out1);
                Tensor out3 = pool1.forward(out2);
                Tensor out4 = conv2.forward(out3);
                Tensor out5 = relu2.forward(out4);
                Tensor out6 = pool2.forward(out5);
                std::vector<float> flattened = flatten.forward(out6);
                std::vector<float> fc_out1 = fc1.forward(flattened);
                std::vector<float> fc_out2 = fc2.forward(fc_out1);

                // Compute loss and accuracy
                std::vector<float> softmax_output = Loss::softmax(fc_out2);
                float loss = Loss::cross_entropy(fc_out2, train_labels[idx]);
                batch_loss += loss;

                int predicted = std::max_element(softmax_output.begin(), softmax_output.end()) 
                              - softmax_output.begin();
                if (predicted == train_labels[idx]) batch_correct++;

                // Backward pass with scaled gradients
                std::vector<float> grad_output = Loss::cross_entropy_gradient(fc_out2, train_labels[idx]);
                float scale = 1.0f / (batch_end - batch_start);
                for (auto& grad : grad_output) grad *= scale;

                // Backward pass through the network
                std::vector<float> grad_fc2 = fc2.backward(grad_output);
                std::vector<float> grad_fc1 = fc1.backward(grad_fc2);
                Tensor grad_pool2 = flatten.backward(grad_fc1, out6);
                Tensor grad_relu2 = relu2.backward(grad_pool2);
                Tensor grad_conv2 = conv2.backward(out3, grad_relu2);
                Tensor grad_pool1 = pool1.backward(grad_conv2);
                Tensor grad_relu1 = relu1.backward(grad_pool1);
                conv1.backward(train_images[idx], grad_relu1);
            }

            // Update weights with current learning rate
            optimizer.update_conv(conv1.getKernels(), conv1.biases, 
                                conv1.getGradKernels(), conv1.grad_biases, current_lr);
            optimizer.update_conv(conv2.getKernels(), conv2.biases, 
                                conv2.getGradKernels(), conv2.grad_biases, current_lr);
            optimizer.update(fc1.weights, fc1.biases, fc1.grad_weights, fc1.grad_biases, current_lr);
            optimizer.update(fc2.weights, fc2.biases, fc2.grad_weights, fc2.grad_biases, current_lr);

            total_loss += batch_loss;
            correct += batch_correct;

            if ((batch_start/BATCH_SIZE) % 10 == 0) {
                std::cout << "Epoch " << epoch + 1 
                         << ", Batch " << batch_start/BATCH_SIZE 
                         << "/" << TRAIN_SIZE/BATCH_SIZE 
                         << ", Batch Loss: " << batch_loss / (batch_end - batch_start)
                         << ", LR: " << current_lr << "\r" << std::flush;
            }
        }

        float accuracy = static_cast<float>(correct) / TRAIN_SIZE;
        std::cout << "\nEpoch " << epoch + 1 
                 << ", Loss: " << total_loss / TRAIN_SIZE 
                 << ", Accuracy: " << accuracy * 100 << "%" << std::endl;
    }

    return 0;
}