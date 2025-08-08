# Neural Network Functions Documentation

This document explains all the neural network functions including dense layers, activation functions, training operations, and memory management.

## Table of Contents
- [Data Structures](#data-structures)
- [Layer Initialization](#layer-initialization)
- [Activation Functions](#activation-functions)
- [Forward Propagation](#forward-propagation)
- [Loss Functions](#loss-functions)
- [Backpropagation](#backpropagation)
- [Training Functions](#training-functions)
- [Utility Functions](#utility-functions)

## Data Structures

### activations
```c
typedef struct {
    int size;           // Number of neurons in this layer
    float* Z;           // Activation values (outputs)
    float* gprime;      // Activation derivatives
    float* dZ;          // Gradients (error signals)
} activations;
```

**Purpose:** Stores all the data needed for a layer of neurons during forward and backward passes.

### DenseLayer
```c
typedef struct {
    int init_type;      // Weight initialization method
    int rows;           // Number of neurons in this layer
    int cols;           // Number of inputs to this layer
    layer* params;      // Weights and biases
    layer* param_grad;  // Current batch gradients
    layer* param_grad_sum; // Accumulated gradients
} DenseLayer;
```

**Purpose:** Represents a fully connected layer with weights, biases, and gradient buffers.

### layer
```c
typedef struct {
    float* Weights;     // Weight matrix (flattened 1D array)
    float* biases;      // Bias vector
} layer;
```

**Purpose:** Stores the learnable parameters of a layer.

## Layer Initialization

### init_activations
```c
activations* init_activations(int size);
```

**What it does:** Creates a new activations structure for a layer with the specified number of neurons.

**Parameters:**
- `size`: Number of neurons in the layer

**Returns:** Pointer to initialized activations structure

**Memory:** Allocates memory for Z, gprime, and dZ arrays

**Example:**
```c
activations* layer1 = init_activations(128);  // 128 neurons
```

### init_DenseLayer
```c
DenseLayer* init_DenseLayer(int rows, int cols, int init_type);
```

**What it does:** Creates a fully connected layer with specified dimensions and weight initialization.

**Parameters:**
- `rows`: Number of neurons in this layer
- `cols`: Number of inputs to this layer
- `init_type`: Weight initialization method (0-14, see Weight Initializer docs)

**Returns:** Pointer to initialized DenseLayer structure

**Example:**
```c
DenseLayer* hidden = init_DenseLayer(64, 128, 5);  // 64 neurons, 128 inputs, He initialization
```

### init_layer
```c
layer* init_layer(int rows, int cols, int init_type);
```

**What it does:** Creates the weight and bias arrays for a layer.

**Parameters:**
- `rows`: Number of neurons
- `cols`: Number of inputs
- `init_type`: Weight initialization method

**Returns:** Pointer to layer with initialized weights and biases

**Memory Management:** Includes error checking and cleanup if allocation fails

## Activation Functions

### activation_function
```c
void activation_function(activations* A, act_func_t func);
```

**What it does:** Applies an activation function to all neurons in a layer.

**Parameters:**
- `A`: Activations structure containing neuron values
- `func`: Type of activation function to apply

**Available Functions:**
- `ReLU`: Rectified Linear Unit - max(0, x)
- `Sigmoid`: 1 / (1 + e^(-x))
- `Tanh`: Hyperbolic tangent
- `LeakyRelu`: max(0.01x, x)
- `PReLU`: Parametric ReLU with α = 0.25
- `ELU`: Exponential Linear Unit
- `SELU`: Scaled Exponential Linear Unit
- `GELU`: Gaussian Error Linear Unit
- `Swish`: x * sigmoid(x)
- `Softmax`: e^x / sum(e^x)

**How it works:**
1. Applies the function to each neuron value
2. Stores the result back in the Z array
3. Computes and stores the derivative in gprime array

**Example:**
```c
activation_function(layer1, ReLU);  // Apply ReLU to all neurons
```

## Forward Propagation

### forward_prop_step
```c
void forward_prop_step(activations* A1, DenseLayer* L, activations* A2);
```

**What it does:** Performs one step of forward propagation through a dense layer.

**Parameters:**
- `A1`: Input activations (previous layer)
- `L`: Dense layer with weights and biases
- `A2`: Output activations (this layer)

**How it works:**
1. Copies biases to output activations
2. Computes weighted sum: `A2 = W * A1 + b`
3. Uses SIMD instructions for faster computation
4. Stores result in A2->Z

**Formula:** `Z = W * X + b`

**Example:**
```c
forward_prop_step(input_layer, hidden_layer, output_layer);
```

## Loss Functions

### loss_function
```c
float loss_function(activations* A, loss_func_t func, int k);
```

**What it does:** Computes the loss and sets up gradients for backpropagation.

**Parameters:**
- `A`: Final layer activations (should be softmax for classification)
- `func`: Type of loss function
- `k`: True class label (0-9 for MNIST)

**Available Functions:**
- `CE`: Cross-Entropy (for classification)
- `MSE`: Mean Squared Error
- `MAE`: Mean Absolute Error
- `HUBER`: Huber Loss
- `BCE`: Binary Cross-Entropy
- `CCE`: Categorical Cross-Entropy

**Returns:** Loss value (float)

**How it works:**
1. Computes the loss based on predicted vs true class
2. Sets up gradients in A->dZ for backpropagation
3. For classification, expects softmax activations

**Example:**
```c
float loss = loss_function(output_layer, CE, true_label);
```

## Backpropagation

### calc_grad_activation
```c
void calc_grad_activation(activations* A_curr, DenseLayer* L, activations* A_prev);
```

**What it does:** Computes gradients for the current layer during backpropagation.

**Parameters:**
- `A_curr`: Current layer activations
- `L`: Layer between current and previous
- `A_prev`: Previous layer activations (receives gradients)

**How it works:**
1. Computes gradient from next layer: `dZ_prev = W^T * dZ_curr`
2. Multiplies by activation derivative: `dZ_prev *= gprime`
3. Stores result in A_prev->dZ

**Formula:** `dZ_prev = (W^T * dZ_curr) ⊙ gprime_prev`

### back_propogate_step
```c
void back_propogate_step(activations* A1, DenseLayer* L, activations* A2);
```

**What it does:** Computes gradients for layer weights and biases.

**Parameters:**
- `A1`: Input activations (previous layer)
- `L`: Layer to compute gradients for
- `A2`: Output activations (current layer)

**How it works:**
1. Computes bias gradients: `db = dZ`
2. Computes weight gradients: `dW = dZ * A1^T`
3. Uses SIMD instructions for speed
4. Stores gradients in L->param_grad

**Formula:** `dW = dZ * X^T`, `db = dZ`

## Training Functions

### grad_accum
```c
void grad_accum(DenseLayer* L, float LR);
```

**What it does:** Accumulates gradients across a batch of samples.

**Parameters:**
- `L`: Layer to accumulate gradients for
- `LR`: Learning rate multiplier

**How it works:**
1. Multiplies current gradients by learning rate
2. Adds to accumulated gradients
3. Uses SIMD instructions for speed

**Usage:** Called for each sample in a batch

### update_weights
```c
void update_weights(DenseLayer* L, float LR);
```

**What it does:** Updates layer weights and biases using accumulated gradients.

**Parameters:**
- `L`: Layer to update
- `LR`: Learning rate

**How it works:**
1. Subtracts accumulated gradients from current weights
2. Uses SIMD instructions for speed
3. Implements gradient descent: `W = W - LR * dW`

**Formula:** `W = W - LR * dW`, `b = b - LR * db`

### zero_grad
```c
void zero_grad(DenseLayer* L);
```

**What it does:** Resets all gradient buffers to zero.

**Parameters:**
- `L`: Layer to reset gradients for

**Usage:** Called at the beginning of each batch

## Utility Functions

### get_pred_from_softmax
```c
int get_pred_from_softmax(activations* A);
```

**What it does:** Finds the class with highest probability from softmax output.

**Parameters:**
- `A`: Activations structure (should contain softmax outputs)

**Returns:** Index of predicted class (0-9 for MNIST)

**How it works:**
1. Finds the maximum value in the activation array
2. Returns the index of that maximum value

**Example:**
```c
int prediction = get_pred_from_softmax(output_layer);
printf("Predicted class: %d\n", prediction);
```

### StandardizeActivations
```c
void StandardizeActivations(activations* A);
```

**What it does:** Normalizes activations to have zero mean and unit variance.

**Parameters:**
- `A`: Activations structure to normalize

**How it works:**
1. Computes mean of all activations
2. Computes standard deviation
3. Normalizes: `(x - mean) / std`
4. Uses SIMD instructions for speed

**Usage:** Can be used for batch normalization effects

## Memory Management

### Free_activations
```c
void Free_activations(activations* A);
```

**What it does:** Frees memory allocated for an activations structure.

**Parameters:**
- `A`: Activations structure to free

**Usage:** Call when done with activations to prevent memory leaks

### Free_DenseLayer
```c
void Free_DenseLayer(DenseLayer* DL);
```

**What it does:** Frees memory allocated for a DenseLayer structure.

**Parameters:**
- `DL`: DenseLayer structure to free

**Usage:** Call when done with layers to prevent memory leaks

### free_layer
```c
void free_layer(layer* Layer);
```

**What it does:** Frees memory allocated for a layer structure.

**Parameters:**
- `Layer`: Layer structure to free

**Usage:** Internal function called by Free_DenseLayer

## Performance Optimizations

### SIMD Vectorization
- **AVX Instructions:** Most functions use 256-bit vector instructions
- **Speedup:** 2-3x faster on modern CPUs
- **Compatibility:** Requires CPU with AVX support (2013+)

### Memory Layout
- **1D Arrays:** Weights stored as flattened arrays for better cache performance
- **Contiguous Memory:** Related data stored together
- **Alignment:** Optimized for SIMD operations

## Common Usage Patterns

### Creating a Network
```c
// Create layers
activations* input = init_activations(784);      // 28x28 = 784
activations* hidden1 = init_activations(128);
activations* hidden2 = init_activations(64);
activations* output = init_activations(10);

DenseLayer* layer1 = init_DenseLayer(128, 784, 5);  // He initialization
DenseLayer* layer2 = init_DenseLayer(64, 128, 5);
DenseLayer* layer3 = init_DenseLayer(10, 64, 5);
```

### Forward Pass
```c
// Forward propagation
forward_prop_step(input, layer1, hidden1);
activation_function(hidden1, ReLU);

forward_prop_step(hidden1, layer2, hidden2);
activation_function(hidden2, ReLU);

forward_prop_step(hidden2, layer3, output);
activation_function(output, Softmax);
```

### Backward Pass
```c
// Compute loss and gradients
float loss = loss_function(output, CE, true_label);

// Backpropagation
back_propogate_step(hidden2, layer3, output);
calc_grad_activation(hidden2, layer3, output);

back_propogate_step(hidden1, layer2, hidden2);
calc_grad_activation(hidden1, layer2, hidden2);

back_propogate_step(input, layer1, hidden1);
calc_grad_activation(input, layer1, hidden1);
```

### Training Loop
```c
for (int epoch = 0; epoch < num_epochs; epoch++) {
    for (int batch = 0; batch < num_batches; batch++) {
        // Reset gradients
        zero_grad(layer1);
        zero_grad(layer2);
        zero_grad(layer3);
        
        for (int sample = 0; sample < batch_size; sample++) {
            // Forward pass
            // ... (as shown above)
            
            // Backward pass
            // ... (as shown above)
            
            // Accumulate gradients
            grad_accum(layer1, 1.0f);
            grad_accum(layer2, 1.0f);
            grad_accum(layer3, 1.0f);
        }
        
        // Update weights
        update_weights(layer1, learning_rate);
        update_weights(layer2, learning_rate);
        update_weights(layer3, learning_rate);
    }
}
```

## Error Handling

**Important Checks:**
- Always verify that layer dimensions match
- Check that activations have correct sizes
- Ensure proper initialization before use
- Handle memory allocation failures

**Common Errors:**
- Dimension mismatch between layers
- Using uninitialized structures
- Memory leaks from not freeing structures
- Incorrect activation function for loss type
