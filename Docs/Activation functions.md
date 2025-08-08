# Activation Functions Documentation

This document explains all the activation functions available in the CNN implementation. Activation functions are crucial components that introduce non-linearity into neural networks, allowing them to learn complex patterns.

## Table of Contents
- [What are Activation Functions?](#what-are-activation-functions)
- [Available Functions](#available-functions)
- [Function Details](#function-details)
- [Usage Examples](#usage-examples)
- [Performance Considerations](#performance-considerations)
- [Best Practices](#best-practices)

## What are Activation Functions?

Activation functions transform the weighted sum of inputs to a neuron, introducing non-linearity into the network. They are essential because:

- **Non-linearity**: Without them, the entire network would be linear
- **Gradient flow**: They help with gradient propagation during training
- **Feature learning**: Different functions learn different types of features

**Mathematical form:** `output = f(weighted_sum + bias)`

## Available Functions

The implementation supports 10 different activation functions:

| Function | Type | Range | Use Case |
|----------|------|-------|----------|
| ReLU | Linear | [0, ∞) | Hidden layers |
| Sigmoid | S-shaped | (0, 1) | Binary classification |
| Tanh | S-shaped | (-1, 1) | Hidden layers |
| LeakyReLU | Linear | (-∞, ∞) | Hidden layers |
| PReLU | Linear | (-∞, ∞) | Hidden layers |
| ELU | Exponential | (-α, ∞) | Hidden layers |
| SELU | Scaled Exponential | (-∞, ∞) | Self-normalizing networks |
| GELU | Smooth | (-∞, ∞) | Transformer models |
| Swish | Smooth | (-∞, ∞) | Modern architectures |
| Softmax | Probability | (0, 1) | Multi-class output |

## Function Details

### ReLU (Rectified Linear Unit)
```c
f(x) = max(0, x)
f'(x) = 1 if x > 0, 0 if x ≤ 0
```

**What it does:** Sets negative values to zero, keeps positive values unchanged.

**Advantages:**
- Simple and fast to compute
- Helps with vanishing gradient problem
- Most commonly used in modern networks

**Disadvantages:**
- Can cause "dying ReLU" problem
- Not differentiable at x = 0

**Best for:** Hidden layers in most neural networks

**Example:**
```c
activation_function(layer, ReLU);
```

### Sigmoid
```c
f(x) = 1 / (1 + e^(-x))
f'(x) = f(x) * (1 - f(x))
```

**What it does:** Maps any input to a value between 0 and 1.

**Advantages:**
- Output can be interpreted as probability
- Smooth and differentiable everywhere

**Disadvantages:**
- Suffers from vanishing gradient problem
- Outputs are not zero-centered

**Best for:** Binary classification output layers

**Example:**
```c
activation_function(output_layer, Sigmoid);
```

### Tanh (Hyperbolic Tangent)
```c
f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
f'(x) = 1 - f(x)^2
```

**What it does:** Maps inputs to values between -1 and 1.

**Advantages:**
- Zero-centered output
- Smooth and differentiable
- Better gradient flow than sigmoid

**Disadvantages:**
- Still suffers from vanishing gradients
- Slower than ReLU

**Best for:** Hidden layers, especially in RNNs

**Example:**
```c
activation_function(hidden_layer, Tanh);
```

### LeakyReLU
```c
f(x) = x if x > 0, 0.01x if x ≤ 0
f'(x) = 1 if x > 0, 0.01 if x ≤ 0
```

**What it does:** Like ReLU but allows small negative gradients.

**Advantages:**
- Prevents "dying ReLU" problem
- Fast computation
- Good gradient flow

**Disadvantages:**
- Requires tuning of leak parameter
- Not as widely used as ReLU

**Best for:** Hidden layers when ReLU causes issues

**Example:**
```c
activation_function(layer, LeakyRelu);
```

### PReLU (Parametric ReLU)
```c
f(x) = x if x > 0, αx if x ≤ 0 (α = 0.25)
f'(x) = 1 if x > 0, α if x ≤ 0
```

**What it does:** Like LeakyReLU but with learnable parameter α.

**Advantages:**
- Learnable slope for negative values
- Better than fixed LeakyReLU
- Good performance in practice

**Disadvantages:**
- More parameters to learn
- Slightly more complex

**Best for:** Deep networks where ReLU underperforms

**Example:**
```c
activation_function(layer, PReLU);
```

### ELU (Exponential Linear Unit)
```c
f(x) = x if x > 0, α(e^x - 1) if x ≤ 0 (α = 1.0)
f'(x) = 1 if x > 0, f(x) + α if x ≤ 0
```

**What it does:** Smooth version of ReLU with exponential decay for negatives.

**Advantages:**
- Smooth everywhere
- Better gradient flow than ReLU
- Closer to zero mean outputs

**Disadvantages:**
- Slower computation due to exponential
- Requires tuning of α parameter

**Best for:** Networks where smoothness is important

**Example:**
```c
activation_function(layer, ELU);
```

### SELU (Scaled Exponential Linear Unit)
```c
f(x) = λ * x if x > 0, λ * α(e^x - 1) if x ≤ 0
λ = 1.0507, α = 1.67326
```

**What it does:** Self-normalizing version of ELU with specific scaling.

**Advantages:**
- Self-normalizing properties
- No need for batch normalization
- Good for deep networks

**Disadvantages:**
- Requires specific weight initialization
- Slower than ReLU
- Not widely adopted

**Best for:** Deep networks without batch normalization

**Example:**
```c
activation_function(layer, SELU);
```

### GELU (Gaussian Error Linear Unit)
```c
f(x) = x * Φ(x) where Φ is the CDF of standard normal
```

**What it does:** Smooth activation based on Gaussian distribution.

**Advantages:**
- Smooth and differentiable
- Good performance in practice
- Used in state-of-the-art models

**Disadvantages:**
- More complex computation
- Slower than ReLU

**Best for:** Transformer models and modern architectures

**Example:**
```c
activation_function(layer, GELU);
```

### Swish
```c
f(x) = x * sigmoid(x)
f'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
```

**What it does:** Self-gated activation that smoothly transitions.

**Advantages:**
- Smooth and differentiable
- Better than ReLU in many cases
- Self-gating property

**Disadvantages:**
- More complex computation
- Slower than ReLU

**Best for:** Modern neural network architectures

**Example:**
```c
activation_function(layer, Swish);
```

### Softmax
```c
f(x_i) = e^(x_i) / Σ(e^(x_j))
```

**What it does:** Converts a vector of numbers into a probability distribution.

**Advantages:**
- Outputs sum to 1
- Perfect for multi-class classification
- Interpretable as probabilities

**Disadvantages:**
- Only for output layers
- Can cause numerical instability

**Best for:** Multi-class classification output layers

**Example:**
```c
activation_function(output_layer, Softmax);
```

## Usage Examples

### Basic Usage
```c
// Apply ReLU to a hidden layer
activations* hidden = init_activations(128);
activation_function(hidden, ReLU);

// Apply softmax to output layer
activations* output = init_activations(10);
activation_function(output, Softmax);
```

### Complete Forward Pass
```c
// Input layer (no activation needed)
// hidden1 = ReLU(W1 * input + b1)
forward_prop_step(input, layer1, hidden1);
activation_function(hidden1, ReLU);

// hidden2 = Tanh(W2 * hidden1 + b2)
forward_prop_step(hidden1, layer2, hidden2);
activation_function(hidden2, Tanh);

// output = Softmax(W3 * hidden2 + b3)
forward_prop_step(hidden2, layer3, output);
activation_function(output, Softmax);
```

### Different Architectures
```c
// Modern CNN with ReLU
activation_function(conv_layer, ReLU);
activation_function(hidden1, ReLU);
activation_function(hidden2, ReLU);
activation_function(output, Softmax);

// Transformer-style with GELU
activation_function(attention, GELU);
activation_function(ffn1, GELU);
activation_function(ffn2, GELU);
activation_function(output, Softmax);

// Self-normalizing network with SELU
activation_function(hidden1, SELU);
activation_function(hidden2, SELU);
activation_function(output, Softmax);
```

## Performance Considerations

### Computational Speed
**Fastest to slowest:**
1. ReLU (fastest)
2. LeakyReLU, PReLU
3. Tanh, Sigmoid
4. ELU, SELU
5. GELU, Swish (slowest)

### Memory Usage
- All functions use the same memory layout
- Derivatives are computed in-place
- No additional memory allocation during forward pass

### Numerical Stability
- **ReLU**: Very stable
- **Sigmoid/Tanh**: Can saturate, causing vanishing gradients
- **Softmax**: Can overflow with large inputs (handled with safe_exp)
- **ELU/SELU**: Can underflow with very negative inputs

## Best Practices

### Choosing Activation Functions

**For Hidden Layers:**
- **Default choice**: ReLU
- **If ReLU causes issues**: Try LeakyReLU or ELU
- **For very deep networks**: Consider SELU or GELU
- **For transformers**: Use GELU

**For Output Layers:**
- **Binary classification**: Sigmoid
- **Multi-class classification**: Softmax
- **Regression**: No activation (linear)

### Common Patterns

**CNN Architecture:**
```c
// Convolutional layers
activation_function(conv1, ReLU);
activation_function(conv2, ReLU);

// Dense layers
activation_function(dense1, ReLU);
activation_function(dense2, ReLU);
activation_function(output, Softmax);
```

**Deep Network:**
```c
// Use consistent activation throughout
for (int i = 0; i < num_layers; i++) {
    forward_prop_step(prev_layer, layers[i], curr_layer);
    activation_function(curr_layer, ReLU);
}
```

### Troubleshooting

**Vanishing Gradients:**
- Use ReLU instead of Sigmoid/Tanh
- Try LeakyReLU or ELU
- Check weight initialization

**Dying ReLU:**
- Use LeakyReLU or PReLU
- Adjust learning rate
- Check for proper weight initialization

**Poor Performance:**
- Try different activation functions
- Experiment with GELU or Swish
- Consider SELU for very deep networks

## Implementation Notes

### Safe Exponential
The implementation uses `safe_exp()` to prevent overflow:
```c
static inline float safe_exp(float x) { 
    return (x > 500.0f || x < -500.0f) ? 0.0f : expf(x); 
}
```

### SIMD Optimization
Some functions use SIMD instructions for faster computation on modern CPUs.

### Memory Layout
All activations are stored in contiguous arrays for better cache performance.
