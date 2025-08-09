# Convolutional Neural Networks in Pure C

A high-performance implementation of Convolutional Neural Networks (CNNs) written entirely in C, featuring SIMD optimizations and support for MNIST and Fashion MNIST datasets.

Note: Almost all of the code was written with minimal help from LLMs. Only help was in writing meanial functions such as completing switch case statements, segfault debugging before I knew what gdb was and some legitimately miniscule errors like reversed arguments to avx fmadd function that was cocking up my training process that even claude could not find and gpt5 in cursor took 15 minutes of deep thinking to find so you can't blame me there. And the documentation in the docs folder was written by AI.

##  Features

### Core Neural Network
- **Convolutional Neural Networks**: Full 2D convolution and pooling operations. Current performance bottleneck.
- **Dense Layers**: Fully connected layers with multiple activation functions. Think there are some small optimizations here.
- **Backpropagation**: Complete Stochastic gradient descent training. Will add Adam later.
- **Multiple Datasets**: Support for MNIST and Fashion MNIST rn but you can train using anything as long as you can load pixel values into the first convolution layer.

### Performance Optimizations
- **AVX SIMD Vectorization**: 2-3x speedup using 256-bit vector instructions
- **Cache-Friendly Memory Layout**: Optimized data structures for better performance
- **Efficient Memory Management**: My laptop had crashed from a memory leak lmao, fixed it so now there's minimal memory allocation during training.

### Activation Functions
- ReLU, Sigmoid, Tanh, LeakyReLU, PReLU
- ELU, SELU, GELU, Swish, Softmax

### Loss Functions
- Cross-Entropy, Mean Squared Error, Mean Absolute Error
- Huber Loss, Binary Cross-Entropy, Categorical Cross-Entropy

## Model Performance

| Dataset | Accuracy | Training Time |
|---------|----------|---------------|
| MNIST | ~96% | ~10 minutes |
| Fashion MNIST | ~90% | ~15 minutes |

MNIST was trained on 10 epochs, ~50s per epoch
*Performance may vary based on hardware specifications*

##  Requirements

- **CPU**: Any processor supporting AVX instructions (released after 2013)
- **OS**: Linux, macOS, or Windows with GCC/Clang
- **Memory**: At least 4GB RAM recommended
- **Storage**: ~50MB for datasets and compiled binary

## Quick Start

### 1. Clone the Repository
```bash
git clone <repository-url>
cd CNNs-in-C
```

### 2. Build the Project
```bash
make
```

### 3. Run Training
```bash
./main
```

The program will automatically:
- Load the specified dataset
- Train the CNN for epochs
- Display training progress and loss
- Calculate and show final accuracy
- For Fmnist train for >4 epochs use He weight init
- For Mnist, train for ~5 epochs, use random weight init

## Project Structure

```
CNNs-in-C/
├── main.c                 # Main training loop and program entry point
├── Conv/
│   ├── Convolution2D.h    # Convolution operations header
│   └── Convolution2D.c    # 2D convolution and pooling implementations
├── NN-funcs/
│   ├── NeuralNetwork.h    # Neural network structures and functions
│   └── NeuralNetwork.c    # Dense layers, activations, and training
├── dataloaders/
│   ├── idx-file-parser.h  # Dataset loading header
│   └── idx-file-parser.c  # MNIST/Fashion MNIST file parser
├── fashion-mnist/         # Fashion MNIST dataset files
├── mnist/                 # MNIST dataset files
├── Docs/                  # Detailed function documentation
└── Makefile              # Build configuration
```

## Configuration

### Training Parameters
You can modify these parameters in `main.c`:

```c
#define BATCH_SIZE 64      // Number of samples per batch
#define NUM_KERNELS 32     // Number of convolution kernels
int epoch = 15;           // Number of training epochs
float learning_rate = 0.0005f;  // Learning rate for gradient descent
```

### Model Architecture
The current architecture for Fmnist:
- **Input**: 28×28 grayscale images
- **Conv Layer**: 32 kernels of size 5×5
- **Max Pooling**: 2×2 with stride 2
- **Dense Layers**: 128 → 64 → 10 neurons
- **Output**: 10 classes (digits 0-9 or fashion categories)

## Documentation

AI generated Comprehensive documentation for all functions is available in the `Docs/` folder:

- [Activation Functions](Docs/Activation%20functions.md) - Detailed guide to all activation functions
- [Loss Functions](Docs/Loss%20functions.md) - Explanation of loss functions and their use cases
- [Weight Initialization](Docs/Weight%20initializer.md) - Different weight initialization strategies
- [Convolution Operations](Docs/Convolution%20Operations.md) - 2D convolution and pooling functions
- [Neural Network Functions](Docs/Neural%20Network%20Functions.md) - Dense layers and training functions
- [Data Loading](Docs/Data%20Loading%20Functions.md) - Dataset parsing and loading functions

## Troubleshooting

### Common Issues

**"Illegal instruction" error**
- If your CPU does not support avx256 then get a new computer buddy.

**"Segmentation fault" during training**
- Should not happen. But submit a pull request or email me or something

**Poor accuracy**
- There is no batchnorm yet
- Try adjusting learning rate or number of epochs
- Check if dataset files are properly downloaded

### Performance Tips
This this was ridicoulously memory hungry because of a memory leak but i fixed it.
- **Batch Size**: Larger batches (64-128) generally train faster
- **Learning Rate**: Start with 0.001 and adjust based on convergence
- **Memory**: Ensure at least 4GB free RAM for optimal performance

## Contributing

Just hit me up and we can talk about you contributing to this.

### Areas for Improvement
- Fucking batchnorm, i know im a bit lazy.
- More optimization techniques, imma vectorize TF outta this.
- Better error handling and validation
- GPU acceleration support, OpenCL
