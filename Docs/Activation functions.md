# Activation Functions

This module provides implementations of standard neural network activation functions, including their derivatives, for use during forward and backward passes.

## Function: `activation_function`

```c
void activation_function(activations* A, act_func_t func);
````

### Parameters

* `activations* A`
  Pointer to an `activations` struct which contains:

  * `A->Z`: array of pre-activation inputs; will be overwritten with activated outputs
  * `A->dZ`: array where derivatives (∂activation/∂input) will be stored
  * `A->size`: number of elements in `Z` and `dZ`

  * `act_func_t func`
  Enum value specifying the activation function to apply. Supported types:

Yep used an enum here, made more sense to do so here than in init cuz you only call it once.
```c
typedef enum {
    ReLU,
    Sigmoid,
    Tanh,
    LeakyRelu,
    PReLU,
    ELU,
    SELU,
    GELU,
    Swish,
    Softmax
} act_func_t;
```

## Behavior

The function performs an in-place activation transformation on `A->Z` and computes the corresponding derivatives in `A->dZ`. These derivatives are intended for use during backpropagation.

### Special Notes
* **PReLU is not supported yet**

* **ReLU, LeakyRelu, PReLU, ELU, SELU**
  All handle negative/positive branches as expected, and write out clean derivatives.

* **GELU**
  Approximated using the tanh-based formulation for performance. Derivative is also approximated.

* **Swish**
  Implements Swish = x \* sigmoid(x), with correct derivative computation.

* **Softmax**
  Applies numerically stable softmax. No derivative is stored in `dZ` because softmax + loss is typically fused in loss backward. Exits on zero sum error.

* **Sigmoid and Tanh**
  Applied elementwise with exact derivatives.

## Helper Functions

### `safe_exp`

```c
static inline float safe_exp(float x);
```

Clamps large values of `x` to avoid `expf` overflow. Returns 0.0f for values outside \[-500, 500].

### `gelu_approx`

```c
static inline float gelu_approx(float x);
```

Returns GELU(x) using a tanh-based approximation:

```
GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715x³)))
```

## Notes

* Activations are modified in place. Make sure `A->Z` is initialized with pre-activation values before calling.
* For `Softmax`, ensure it's only used on the final layer when paired with a cross-entropy loss function.
* All math functions used are from `math.h` and rely on single-precision (float) for performance.

```
```
