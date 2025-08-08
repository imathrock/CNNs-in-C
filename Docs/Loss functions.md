# Loss Functions Documentation

This document explains all the loss functions available in the CNN implementation. Loss functions measure how well the model's predictions match the true targets and provide gradients for training.

## Table of Contents
- [What are Loss Functions?](#what-are-loss-functions)
- [Available Functions](#available-functions)
- [Function Details](#function-details)
- [Usage Examples](#usage-examples)
- [Performance Considerations](#performance-considerations)
- [Best Practices](#best-practices)

## What are Loss Functions?

Loss functions (also called cost functions or objective functions) measure the difference between predicted and true values. They serve two critical purposes:

- **Training signal**: Provide gradients to update model weights
- **Performance metric**: Quantify how well the model is performing

**Mathematical form:** `Loss = f(predicted, target)`

## Available Functions

The implementation supports 9 different loss functions:

| Function | Type | Use Case | Output Range |
|----------|------|----------|--------------|
| CE | Classification | Multi-class | [0, ∞) |
| MSE | Regression | Continuous values | [0, ∞) |
| MAE | Regression | Robust regression | [0, ∞) |
| HUBER | Regression | Robust regression | [0, ∞) |
| BCE | Classification | Binary classification | [0, ∞) |
| CCE | Classification | Multi-class | [0, ∞) |
| SCE | Classification | Multi-class | [0, ∞) |
| L1loss | Regression | Sparse regression | [0, ∞) |
| L2loss | Regression | Standard regression | [0, ∞) |

## Function Details

### CE (Cross-Entropy)
```c
Loss = -log(predicted[true_class])
Gradient = predicted - target (one-hot)
```

**What it does:** Measures the difference between predicted probabilities and true class.

**Advantages:**
- Standard for classification problems
- Works well with softmax outputs
- Provides good gradients

**Disadvantages:**
- Can be numerically unstable
- Requires softmax activation

**Best for:** Multi-class classification with softmax output

**Example:**
```c
// After softmax activation
activation_function(output, Softmax);
float loss = loss_function(output, CE, true_label);
```

### MSE (Mean Squared Error)
```c
Loss = (1/n) * Σ(predicted[i] - target[i])²
Gradient = 2 * (predicted - target) / n
```

**What it does:** Measures the average squared difference between predictions and targets.

**Advantages:**
- Standard for regression
- Smooth and differentiable
- Penalizes large errors heavily

**Disadvantages:**
- Sensitive to outliers
- May not be optimal for classification

**Best for:** Regression problems, continuous outputs

**Example:**
```c
// For regression output (no activation)
float loss = loss_function(output, MSE, target_index);
```

### MAE (Mean Absolute Error)
```c
Loss = (1/n) * Σ|predicted[i] - target[i]|
Gradient = sign(predicted - target) / n
```

**What it does:** Measures the average absolute difference between predictions and targets.

**Advantages:**
- Robust to outliers
- Linear penalty for errors
- Good for noisy data

**Disadvantages:**
- Less smooth than MSE
- Gradient is constant magnitude

**Best for:** Regression with outliers, robust estimation

**Example:**
```c
float loss = loss_function(output, MAE, target_index);
```

### HUBER (Huber Loss)
```c
Loss = (1/n) * Σ(0.5 * error² if |error| < δ, δ * |error| - 0.5δ² otherwise)
Gradient = error if |error| < δ, δ * sign(error) otherwise
```

**What it does:** Combines benefits of MSE and MAE, robust to outliers.

**Advantages:**
- Best of both MSE and MAE
- Robust to outliers
- Smooth gradients

**Disadvantages:**
- Requires tuning of δ parameter
- More complex computation

**Best for:** Regression with mixed noise characteristics

**Example:**
```c
float loss = loss_function(output, HUBER, target_index);
```

### BCE (Binary Cross-Entropy)
```c
Loss = -(target * log(predicted) + (1-target) * log(1-predicted))
Gradient = predicted - target
```

**What it does:** Measures binary classification loss between predicted and true probabilities.

**Advantages:**
- Standard for binary classification
- Works with sigmoid outputs
- Good theoretical properties

**Disadvantages:**
- Only for binary problems
- Requires sigmoid activation

**Best for:** Binary classification with sigmoid output

**Example:**
```c
// For binary classification (sigmoid output)
activation_function(output, Sigmoid);
float loss = loss_function(output, BCE, target);
```

### CCE (Categorical Cross-Entropy)
```c
Loss = -log(predicted[true_class])
Gradient = predicted - target (one-hot)
```

**What it does:** Same as CE, but explicitly for categorical targets.

**Advantages:**
- Clear semantic meaning
- Standard for classification
- Good gradients

**Disadvantages:**
- Requires one-hot encoded targets
- Same as CE in practice

**Best for:** Multi-class classification with one-hot targets

**Example:**
```c
float loss = loss_function(output, CCE, true_class);
```

### SCE (Sparse Categorical Cross-Entropy)
```c
Loss = -log(predicted[true_class])
Gradient = predicted - target (sparse)
```

**What it does:** Same as CE but for integer class labels instead of one-hot.

**Advantages:**
- Simpler target format
- Memory efficient
- Standard for classification

**Disadvantages:**
- Same as CE in practice
- Requires integer labels

**Best for:** Multi-class classification with integer labels

**Example:**
```c
// true_class is integer (0-9 for MNIST)
float loss = loss_function(output, SCE, true_class);
```

### L1loss (L1 Loss)
```c
Loss = |predicted[true_class] - 1|
Gradient = sign(predicted[true_class] - 1)
```

**What it does:** L1 loss for classification, measures absolute difference from target.

**Advantages:**
- Robust to outliers
- Sparse gradients
- Good for feature selection

**Disadvantages:**
- Less common for classification
- Non-smooth gradients

**Best for:** Sparse classification problems

**Example:**
```c
float loss = loss_function(output, L1loss, true_class);
```

### L2loss (L2 Loss)
```c
Loss = 0.5 * (predicted[true_class] - 1)²
Gradient = predicted[true_class] - 1
```

**What it does:** L2 loss for classification, measures squared difference from target.

**Advantages:**
- Smooth gradients
- Standard choice
- Good theoretical properties

**Disadvantages:**
- Sensitive to outliers
- Less robust than L1

**Best for:** Standard classification problems

**Example:**
```c
float loss = loss_function(output, L2loss, true_class);
```

## Usage Examples

### Basic Usage
```c
// Multi-class classification
activations* output = init_activations(10);
activation_function(output, Softmax);
float loss = loss_function(output, CE, true_label);

// Binary classification
activations* output = init_activations(1);
activation_function(output, Sigmoid);
float loss = loss_function(output, BCE, target);

// Regression
activations* output = init_activations(1);
float loss = loss_function(output, MSE, target_index);
```

### Complete Training Step
```c
// Forward pass
forward_prop_step(input, layer1, hidden1);
activation_function(hidden1, ReLU);

forward_prop_step(hidden1, layer2, output);
activation_function(output, Softmax);

// Loss computation
float loss = loss_function(output, CE, true_label);

// Backward pass (gradients are set up automatically)
back_propogate_step(hidden1, layer2, output);
calc_grad_activation(hidden1, layer2, output);
```

### Different Problem Types
```c
// MNIST digit classification
activation_function(output, Softmax);
float loss = loss_function(output, CE, digit_label);

// Fashion MNIST classification
activation_function(output, Softmax);
float loss = loss_function(output, SCE, fashion_label);

// Regression problem
float loss = loss_function(output, MSE, target_value);

// Robust regression
float loss = loss_function(output, HUBER, target_value);
```

## Performance Considerations

### Computational Speed
**Fastest to slowest:**
1. L1loss, L2loss (fastest)
2. CE, CCE, SCE
3. MSE, MAE
4. HUBER (slowest)

### Memory Usage
- All functions compute gradients in-place
- No additional memory allocation
- Gradients stored in `activations->dZ`

### Numerical Stability
- **CE/CCE/SCE**: Uses safe_exp to prevent overflow
- **MSE**: Very stable
- **MAE**: Stable but non-smooth
- **HUBER**: Good stability with proper δ

## Best Practices

### Choosing Loss Functions

**For Classification:**
- **Multi-class**: CE (Cross-Entropy) with Softmax
- **Binary**: BCE (Binary Cross-Entropy) with Sigmoid
- **Integer labels**: SCE (Sparse Categorical Cross-Entropy)

**For Regression:**
- **Standard**: MSE (Mean Squared Error)
- **With outliers**: MAE (Mean Absolute Error) or HUBER
- **Robust**: HUBER with tuned δ parameter

### Common Patterns

**Classification Network:**
```c
// Hidden layers
activation_function(hidden1, ReLU);
activation_function(hidden2, ReLU);

// Output layer
activation_function(output, Softmax);
float loss = loss_function(output, CE, true_label);
```

**Regression Network:**
```c
// Hidden layers
activation_function(hidden1, ReLU);
activation_function(hidden2, ReLU);

// Output layer (no activation)
float loss = loss_function(output, MSE, target_index);
```

**Binary Classification:**
```c
// Hidden layers
activation_function(hidden1, ReLU);

// Output layer
activation_function(output, Sigmoid);
float loss = loss_function(output, BCE, target);
```

### Loss Function Combinations

**With Activation Functions:**
- **Softmax + CE**: Standard for multi-class
- **Sigmoid + BCE**: Standard for binary
- **Linear + MSE**: Standard for regression
- **Linear + MAE**: Robust regression

**Avoid These Combinations:**
- Softmax + MSE (theoretically incorrect)
- Sigmoid + CE (use BCE instead)
- Linear + CE (use Softmax + CE)

## Troubleshooting

### Common Issues

**Loss Not Decreasing:**
- Check learning rate
- Verify loss function matches activation
- Ensure proper data normalization

**Numerical Instability:**
- Use safe_exp for exponential functions
- Check for very large or small values
- Consider different loss function

**Poor Performance:**
- Try different loss functions
- Check activation function compatibility
- Verify target format

### Debugging Tips

**Monitor Loss Values:**
```c
float loss = loss_function(output, CE, true_label);
printf("Loss: %f\n", loss);

// Check gradients
for (int i = 0; i < output->size; i++) {
    printf("Gradient[%d]: %f\n", i, output->dZ[i]);
}
```

**Verify Target Format:**
```c
// For classification
printf("True label: %d\n", true_label);
printf("Predicted probabilities: ");
for (int i = 0; i < output->size; i++) {
    printf("%f ", output->Z[i]);
}
printf("\n");
```

## Implementation Notes

### Safe Exponential
The implementation uses `safe_exp()` to prevent numerical overflow:
```c
static inline float safe_exp(float x) { 
    return (x > 500.0f || x < -500.0f) ? 0.0f : expf(x); 
}
```

### Gradient Computation
All loss functions automatically compute and store gradients in the `activations->dZ` array for use in backpropagation.

### Memory Layout
Loss functions operate on contiguous arrays for optimal cache performance and SIMD optimization.
