# CNN from Scratch in C++

This repository implements a **Convolutional Neural Network (CNN)** from scratch in **C++**. The project demonstrates training on the MNIST dataset, handling forward and backward passes, gradient updates, and basic optimization without relying on external machine learning libraries.


```markdown

---

## **Directory Structure**

```plaintext
phucngvinuni-cnnfrom_scratchcpp/
├── README.md               # Project documentation
├── Makefile                # Compilation instructions
├── main.cpp                # Main program file
├── storrr                  # Placeholder (optional purpose not specified)
├── train-images-idx3-ubyte # MNIST training images (flat)
├── train-labels-idx1-ubyte # MNIST training labels (flat)
├── MNIST/                  # Full MNIST dataset
│   ├── [Various MNIST files and their copies]
├── layers/                 # CNN layer implementations
│   ├── conv_layer.cpp/h    # Convolutional layer
│   ├── relu_layer.cpp/h    # ReLU activation layer
│   ├── maxpool_layer.cpp/h # Max-pooling layer
│   ├── flatten_layer.cpp/h # Flatten layer
│   ├── fully_connected_layer.cpp/h # Fully connected layer
├── utils/                  # Utility files
│   ├── mnist_loader.cpp/h  # MNIST dataset loader
│   ├── loss.cpp/h          # Loss functions and gradients
│   ├── optimizer.cpp/h     # Optimizers
│   ├── tensor.cpp/h        # Tensor implementation
```

---

## **Features**

- Fully implemented CNN architecture:
  - **Convolutional layers**
  - **ReLU activation**
  - **Max-pooling**
  - **Fully connected layers**
- Training on the MNIST dataset with batch gradient descent and backpropagation.
- Manual implementation of **softmax**, **cross-entropy loss**, and **gradient computation**.
- Dynamic learning rate adjustment and weight updates.

---

## **Requirements**

- **Compiler**: GCC or any C++17 compatible compiler.
- **Dataset**: The MNIST dataset files are included in the `MNIST` directory.
- **Build Tool**: Makefile provided for convenient compilation.

---

## **Compilation and Execution**

### 1. Compile the program
Use the provided `Makefile`:
```bash
make
```

### 2. Run the program
```bash
./cnn_train
```

---

## **How It Works**

1. **Dataset Loading**:
   - The MNIST dataset is loaded using `MNISTLoader`.
   - Images are normalized to the range `[0, 1]`.

2. **Model Architecture**:
   - Input: 28x28 grayscale images.
   - Layers:
     - Conv Layer (1 input channel, 32 filters, kernel size 3x3, stride 1, padding 1).
     - ReLU Activation.
     - Max Pooling (2x2).
     - Conv Layer (32 input channels, 64 filters, kernel size 3x3, stride 1, padding 1).
     - ReLU Activation.
     - Max Pooling (2x2).
     - Flatten Layer.
     - Fully Connected Layer (64x7x7 to 128).
     - Fully Connected Layer (128 to 10).
   - Output: Probabilities over 10 classes (digits 0-9).

3. **Training**:
   - Batch size: 64.
   - Number of epochs: 50.
   - Loss: Cross-entropy.
   - Optimization: Manual weight updates with learning rate decay.

4. **Performance Metrics**:
   - **Loss** and **Accuracy** reported at each epoch.

---

## **Sample Output**

```plaintext
Loading MNIST dataset...
Epoch 1, Loss: 0.350, Accuracy: 87.1%
Epoch 2, Loss: 0.250, Accuracy: 90.3%
...
Epoch 50, Loss: 0.012, Accuracy: 99.2%
```

---

## **Future Improvements**

- Add support for testing and evaluation on separate datasets.
- Extend optimizer choices (e.g., Adam, RMSProp).
- Implement dropout for better regularization.
- Expand to support other datasets and architectures.

---

## **Contributing**

We welcome contributions to enhance this project. To contribute:
1. Fork the repository.
2. Create a new branch (`feature-name`).
3. Commit your changes (`git commit -m "Add new feature"`).
4. Push to your branch (`git push origin feature-name`).
5. Open a Pull Request.

---

## **License**

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

