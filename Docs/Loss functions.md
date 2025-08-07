# Loss Function 

This file documents the usage of the `loss_function` implementation used for neural network training. It supports a range of standard loss functions used in both classification and regression tasks.

## Function Signature

```c
float loss_function(activations* A, loss_func_t func, int k);
````

### Parameters

* `activations* A`
  Pointer to the activations struct, which must contain the following:

  * `A->Z`: output predictions (typically from softmax or sigmoid)
  * `A->dZ`: array to be filled with the loss gradient with respect to `Z`
  * `A->size`: number of output units

* `loss_func_t func`
  Enum indicating the type of loss function to compute. The available options are:

  ```c
  typedef enum {
      L1,
      L2,
      CE,
      MSE,
      MAE,
      HUBER,
      BCE,
      CCE,
      SCE
  } loss_func_t;
  ```

* `int k`
  Ground truth label. For classification tasks, this is the class index. For binary classification, use `0` or `1`.

## Return Value

Returns a float representing the scalar loss value for the given sample. The gradient `A->dZ` is also populated for use in the backward pass.

## Behavior by Loss Function

| Loss Function | Purpose                          | Behavior                                             |
| ------------- | -------------------------------- | ---------------------------------------------------- |
| `L1`          | Classification                   | Computes absolute error per class, target as one-hot |
| `L2`          | Classification                   | Computes squared error per class, target as one-hot  |
| `CE`          | Multi-class classification       | Cross entropy loss using log(A->Z\[k])               |
| `MSE`         | Regression or classification     | Mean squared error between prediction and one-hot    |
| `MAE`         | Regression or classification     | Mean absolute error between prediction and one-hot   |
| `HUBER`       | Regression                       | Huber loss with delta = 1.0                          |
| `BCE`         | Binary classification            | Binary cross entropy loss for a single output unit   |
| `CCE`         | Multi-class classification       | Categorical cross entropy, same as CE                |
| `SCE`         | Sparse categorical cross entropy | Same as CE, assumes integer labels                   |

## Notes

* The function assumes activations have already passed through appropriate output functions (e.g., softmax for CE, sigmoid for BCE).
* A small epsilon (`1e-9f`) is added inside logarithms to prevent numerical errors.
* The computed gradient in `A->dZ` matches the shape of `A->Z` and should be used for backward propagation.
* Loss is computed per-sample. If working with mini-batches, you should average the result across all samples externally.

```
```
