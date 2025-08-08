# Weight Initialization Documentation

This document explains all the weight initialization methods available in the CNN implementation. Proper weight initialization is crucial for training deep neural networks effectively.

## Table of Contents
- [What is Weight Initialization?](#what-is-weight-initialization)
- [Available Methods](#available-methods)
- [Method Details](#method-details)
- [Usage Examples](#usage-examples)
- [Best Practices](#best-practices)
- [Performance Considerations](#performance-considerations)

## What is Weight Initialization?

Weight initialization sets the initial values of neural network weights before training begins. The choice of initialization method significantly affects:

- **Training speed**: How quickly the network converges
- **Training stability**: Whether gradients flow properly
- **Final performance**: The quality of learned representations

**Goal:** Initialize weights to break symmetry and allow proper gradient flow.

## Available Methods

The implementation supports 15 different initialization methods:

| ID | Method | Type | Best For | Formula |
|----|--------|------|----------|---------|
| 0 | Zero | Constant | Testing only | W = 0 |
| 1 | Random Uniform | Uniform | Simple networks | W ~ U(-0.5, 0.5) |
| 2 | Random Normal | Normal | General purpose | W ~ N(0, 1) |
| 3 | Xavier/Glorot Uniform | Uniform | Tanh/Sigmoid | W ~ U(-√(6/(fan_in+fan_out)), √(6/(fan_in+fan_out))) |
| 4 | Xavier/Glorot Normal | Normal | Tanh/Sigmoid | W ~ N(0, √(2/(fan_in+fan_out))) |
| 5 | He/Kaiming Uniform | Uniform | ReLU | W ~ U(-√(6/fan_in), √(6/fan_in)) |
| 6 | He/Kaiming Normal | Normal | ReLU | W ~ N(0, √(2/fan_in)) |
| 7 | LeCun Uniform | Uniform | SELU | W ~ U(-√(3/fan_in), √(3/fan_in)) |
| 8 | LeCun Normal | Normal | SELU | W ~ N(0, √(1/fan_in)) |
| 9 | Orthogonal | Special | RNNs | W = orthogonal matrix |
| 10 | Identity | Special | Skip connections | W = I (if square) |
| 11 | Variance Scaling Uniform | Uniform | General | W ~ U(-√(3/fan_avg), √(3/fan_avg)) |
| 12 | Variance Scaling Normal | Normal | General | W ~ N(0, √(1/fan_avg)) |
| 13 | Truncated Normal | Normal | Stable training | W ~ N(0, σ) truncated |
| 14 | Small Random | Uniform | Simple | W ~ U(-0.01, 0.01) |

## Method Details

### 0. Zero Initialization
```c
W = 0 for all weights
```

**What it does:** Sets all weights to zero.

**Advantages:**
- Simple and deterministic
- Good for testing

**Disadvantages:**
- Breaks symmetry but causes vanishing gradients
- Neurons learn identical features

**Best for:** Testing and debugging only

**Example:**
```c
DenseLayer* layer = init_DenseLayer(64, 128, 0);  // Zero initialization
```

### 1. Random Uniform
```c
W ~ Uniform(-0.5, 0.5)
```

**What it does:** Initializes weights from uniform distribution.

**Advantages:**
- Simple and intuitive
- Good starting point

**Disadvantages:**
- May not scale well with layer size
- No theoretical justification

**Best for:** Simple networks and experimentation

### 2. Random Normal
```c
W ~ Normal(0, 1)
```

**What it does:** Initializes weights from standard normal distribution.

**Advantages:**
- Standard choice
- Good for many activation functions

**Disadvantages:**
- May cause vanishing/exploding gradients
- No layer size consideration

**Best for:** General purpose, moderate depth networks

### 3. Xavier/Glorot Uniform
```c
W ~ Uniform(-√(6/(fan_in+fan_out)), √(6/(fan_in+fan_out)))
```

**What it does:** Maintains variance of activations and gradients.

**Advantages:**
- Theoretically sound
- Good for sigmoid/tanh activations
- Prevents vanishing/exploding gradients

**Disadvantages:**
- May not be optimal for ReLU
- Requires knowledge of layer sizes

**Best for:** Networks with sigmoid or tanh activations

**Example:**
```c
DenseLayer* layer = init_DenseLayer(64, 128, 3);  // Xavier Uniform
```

### 4. Xavier/Glorot Normal
```c
W ~ Normal(0, √(2/(fan_in+fan_out)))
```

**What it does:** Normal version of Xavier initialization.

**Advantages:**
- Same benefits as Xavier Uniform
- Often preferred over uniform
- Good theoretical properties

**Disadvantages:**
- May not be optimal for ReLU
- Requires layer size information

**Best for:** Networks with sigmoid or tanh activations

### 5. He/Kaiming Uniform
```c
W ~ Uniform(-√(6/fan_in), √(6/fan_in))
```

**What it does:** Designed specifically for ReLU activations.

**Advantages:**
- Optimal for ReLU networks
- Prevents dying ReLU problem
- Good gradient flow

**Disadvantages:**
- Specific to ReLU-like activations
- May not work well with sigmoid/tanh

**Best for:** Networks with ReLU, LeakyReLU, or similar activations

**Example:**
```c
DenseLayer* layer = init_DenseLayer(64, 128, 5);  // He Uniform
```

### 6. He/Kaiming Normal
```c
W ~ Normal(0, √(2/fan_in))
```

**What it does:** Normal version of He initialization.

**Advantages:**
- Optimal for ReLU networks
- Often preferred over uniform
- Excellent for deep networks

**Disadvantages:**
- Specific to ReLU-like activations

**Best for:** Deep networks with ReLU activations

### 7. LeCun Uniform
```c
W ~ Uniform(-√(3/fan_in), √(3/fan_in))
```

**What it does:** Designed for SELU activation function.

**Advantages:**
- Optimal for SELU networks
- Enables self-normalizing properties
- Good for deep networks

**Disadvantages:**
- Specific to SELU activation
- Requires specific architecture

**Best for:** Self-normalizing networks with SELU

### 8. LeCun Normal
```c
W ~ Normal(0, √(1/fan_in))
```

**What it does:** Normal version of LeCun initialization.

**Advantages:**
- Optimal for SELU networks
- Enables self-normalizing properties
- Good theoretical foundation

**Disadvantages:**
- Specific to SELU activation

**Best for:** Self-normalizing networks with SELU

### 9. Orthogonal Initialization
```c
W = orthogonal matrix
```

**What it does:** Initializes weights as orthogonal matrices.

**Advantages:**
- Good for recurrent networks
- Maintains gradient flow
- Prevents vanishing gradients

**Disadvantages:**
- More complex computation
- Not always necessary

**Best for:** Recurrent neural networks (RNNs)

### 10. Identity Initialization
```c
W = I (identity matrix) if square, otherwise W = 0
```

**What it does:** Sets weights to identity matrix for square layers.

**Advantages:**
- Good for skip connections
- Preserves input information
- Stable training

**Disadvantages:**
- Only works for square matrices
- May not be optimal for all cases

**Best for:** Skip connections and residual networks

### 11. Variance Scaling Uniform
```c
W ~ Uniform(-√(3/fan_avg), √(3/fan_avg))
```

**What it does:** Uses average of fan_in and fan_out for scaling.

**Advantages:**
- Good compromise between Xavier and He
- Works well with many activations
- Balanced approach

**Disadvantages:**
- May not be optimal for specific cases

**Best for:** General purpose networks

### 12. Variance Scaling Normal
```c
W ~ Normal(0, √(1/fan_avg))
```

**What it does:** Normal version of variance scaling.

**Advantages:**
- Good compromise approach
- Often works well in practice
- Balanced scaling

**Disadvantages:**
- May not be optimal for specific cases

**Best for:** General purpose networks

### 13. Truncated Normal
```c
W ~ Normal(0, σ) truncated at ±2σ
```

**What it does:** Normal distribution with truncated tails.

**Advantages:**
- Prevents extreme weight values
- More stable training
- Good for deep networks

**Disadvantages:**
- More complex computation
- May limit expressiveness

**Best for:** Deep networks requiring stability

### 14. Small Random Values
```c
W ~ Uniform(-0.01, 0.01)
```

**What it does:** Very small random weights.

**Advantages:**
- Simple and safe
- Good for shallow networks
- Prevents saturation

**Disadvantages:**
- May cause slow learning
- Not optimal for deep networks

**Best for:** Simple networks and experimentation

## Usage Examples

### Basic Usage
```c
// Create layers with different initializations
DenseLayer* layer1 = init_DenseLayer(128, 784, 5);  // He Uniform for ReLU
DenseLayer* layer2 = init_DenseLayer(64, 128, 5);   // He Uniform for ReLU
DenseLayer* layer3 = init_DenseLayer(10, 64, 5);    // He Uniform for ReLU
```

### Different Architectures
```c
// CNN with ReLU activations
DenseLayer* conv_layer = init_DenseLayer(32, 784, 5);  // He Uniform
DenseLayer* hidden1 = init_DenseLayer(128, 32, 5);     // He Uniform
DenseLayer* hidden2 = init_DenseLayer(64, 128, 5);     // He Uniform
DenseLayer* output = init_DenseLayer(10, 64, 5);       // He Uniform

// Network with sigmoid activations
DenseLayer* hidden1 = init_DenseLayer(128, 784, 3);    // Xavier Uniform
DenseLayer* hidden2 = init_DenseLayer(64, 128, 3);     // Xavier Uniform
DenseLayer* output = init_DenseLayer(10, 64, 3);       // Xavier Uniform

// Self-normalizing network with SELU
DenseLayer* hidden1 = init_DenseLayer(128, 784, 8);    // LeCun Normal
DenseLayer* hidden2 = init_DenseLayer(64, 128, 8);     // LeCun Normal
DenseLayer* output = init_DenseLayer(10, 64, 8);       // LeCun Normal
```

### Experimentation
```c
// Try different initializations
DenseLayer* layer1 = init_DenseLayer(128, 784, 5);  // He Uniform
DenseLayer* layer2 = init_DenseLayer(128, 784, 6);  // He Normal
DenseLayer* layer3 = init_DenseLayer(128, 784, 4);  // Xavier Normal

// Compare performance and choose the best
```

## Best Practices

### Choosing Initialization Methods

**For ReLU Networks (Most Common):**
- **Default choice**: He Normal (ID: 6)
- **Alternative**: He Uniform (ID: 5)
- **For very deep networks**: Truncated Normal (ID: 13)

**For Sigmoid/Tanh Networks:**
- **Default choice**: Xavier Normal (ID: 4)
- **Alternative**: Xavier Uniform (ID: 3)

**For SELU Networks:**
- **Default choice**: LeCun Normal (ID: 8)
- **Alternative**: LeCun Uniform (ID: 7)

**For General Purpose:**
- **Default choice**: Variance Scaling Normal (ID: 12)
- **Alternative**: Variance Scaling Uniform (ID: 11)

### Common Patterns

**Modern CNN Architecture:**
```c
// All layers use He initialization for ReLU
DenseLayer* conv1 = init_DenseLayer(32, 784, 5);   // He Uniform
DenseLayer* conv2 = init_DenseLayer(64, 32, 5);    // He Uniform
DenseLayer* dense1 = init_DenseLayer(128, 64, 5);  // He Uniform
DenseLayer* dense2 = init_DenseLayer(64, 128, 5);  // He Uniform
DenseLayer* output = init_DenseLayer(10, 64, 5);   // He Uniform
```

**Mixed Architecture:**
```c
// Hidden layers with ReLU
DenseLayer* hidden1 = init_DenseLayer(128, 784, 6);  // He Normal
DenseLayer* hidden2 = init_DenseLayer(64, 128, 6);   // He Normal

// Output layer with softmax (no specific requirement)
DenseLayer* output = init_DenseLayer(10, 64, 6);     // He Normal
```

### Troubleshooting

**Vanishing Gradients:**
- Use He initialization for ReLU networks
- Use Xavier initialization for sigmoid/tanh networks
- Check activation function compatibility

**Exploding Gradients:**
- Use smaller initialization scales
- Try truncated normal initialization
- Check learning rate

**Poor Convergence:**
- Try different initialization methods
- Experiment with variance scaling
- Check layer sizes and architecture

## Performance Considerations

### Computational Speed
**Fastest to slowest:**
1. Zero, Small Random (fastest)
2. Random Uniform, Random Normal
3. Xavier, He, LeCun methods
4. Variance Scaling methods
5. Truncated Normal (slowest)

### Memory Usage
- All methods use the same memory layout
- No significant memory differences
- Orthogonal initialization may use more temporary memory

### Numerical Stability
- **He/Xavier methods**: Very stable
- **Random methods**: Generally stable
- **Truncated Normal**: Most stable
- **Zero**: May cause issues in deep networks

## Implementation Notes

### Random Number Generation
The implementation uses the standard C `rand()` function with Box-Muller transform for normal distributions.

### Box-Muller Transform
```c
float randn() {
    // Generates normal random numbers using Box-Muller transform
    // Returns two numbers per call, stores one for next call
}
```

### Error Handling
All initialization methods include error checking for memory allocation failures.

### SIMD Optimization
Some initialization methods use SIMD instructions for faster computation on modern CPUs.