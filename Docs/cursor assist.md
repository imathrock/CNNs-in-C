I'll scan your code to see how layers and the training loop are structured so I can tailor the batchnorm plan to your conventions.
Search files...

I'll quickly locate how activations and forward/backward are implemented and whether batchnorm hooks already exist, then outline exactly what you need to add and where.
Ran tool
Ran tool
Ran tool
Ran tool
Ran tool

I'll search for the forward, activation, and backward functions so I can tell you exactly where to hook batchnorm in both passes.
Ran tool
Ran tool

Ran tool
Ran tool

Ran tool

### What you already have
- `activations` carries `Z`, `gprime`, `dZ` plus a `norm_type` and `norm_params`.
- `batchnorm_t` exists and allocates per-feature buffers plus a `[batch_size × num_features]` buffer (`x_hat`) and a `count`.
- `activation_function` currently tries to call `batchnorm(A)` first, but `batchnorm` is commented out.
- Your training loop runs sample-by-sample inside the batch and only accumulates gradients, which is incompatible with true BatchNorm that needs all m samples to compute mean/var together.

### What you need to add/change (high-level)
- Implement true mini-batch forward/backward flow per layer (layer-wise, batch-first), not sample-by-sample.
- Complete BatchNorm forward/backward and use it between the linear layer output and the nonlinearity.
- Track running statistics and update BN parameters.
- Separate training vs inference behavior.

### Concretely, step-by-step

1) Extend `batchnorm_t` to support training + inference
- Add:
  - `float *running_mean, *running_var` (size `num_features`)
  - `float momentum` (e.g., 0.1f) and `float eps` (e.g., 1e-5f)
  - Either add a separate buffer for raw pre-normalization activations, e.g. `float *x_batch` [batch_size × num_features], or change how you use `x_hat` so you don’t overwrite the raw inputs you need in backward.
- Initialize `running_mean` to zeros and `running_var` to ones; set `momentum`, `eps` in `init_batchnorm`. Free them in `free_batchnorm`.

2) Move BatchNorm out of `activation_function` in training
- For training, don’t call BN inside `activation_function`. Instead: apply BN first, then call `activation_function` to apply the nonlinearity and fill `gprime`.
- Keep `inference_activation_function` as-is for nonlinearity; add BN to the inference path too, but using running stats.

3) Restructure the training loop to operate per-batch and per-layer
- For each mini-batch:
  - Layer 1:
    - For each sample k in the batch:
      - Compute Conv/Pool and fill `A1->Z` for that sample.
      - Copy that vector into `A1->norm_params.BN->x_batch[row_k, :]`.
    - When the batch is full:
      - BN forward on `A1` across the whole `x_batch` to produce normalized batch `y_batch`.
      - For each sample k, copy normalized row back into `A1->Z`, then call `activation_function(A1, ReLU)` to produce post-activation `A1->Z` and `A1->gprime`.
  - Layer 2:
    - For each sample k, run `forward_prop_step(A1, L1, A2)` to produce the pre-activation for layer 2 and copy to `A2->BN->x_batch[row_k, :]`.
    - BN forward on `A2` batch, then per-sample `activation_function(A2, ReLU)`.
  - Layer 3:
    - Repeat the same pattern (forward to `A3`, BN(A3), then `activation_function(A3, ReLU)`).
  - Output:
    - Per-sample Softmax and loss; accumulate loss.

4) Implement BatchNorm forward (training)
- For each feature j:
  - mean: `μ_j = (1/m) Σ_i x_ij`
  - var: `σ²_j = (1/m) Σ_i (x_ij − μ_j)²`
  - normalize: `x̂_ij = (x_ij − μ_j) / sqrt(σ²_j + eps)`
  - scale/shift: `y_ij = γ_j x̂_ij + β_j`
- Save for backward: `x̂`, `μ`, `σ²`. Update running stats:
  - `running_mean_j = (1−momentum)*running_mean_j + momentum*μ_j`
  - `running_var_j  = (1−momentum)*running_var_j  + momentum*σ²_j`

5) Implement BatchNorm backward (training)
- Input: upstream gradient `dY` of shape `[m × num_features]` (this is the gradient after the nonlinearity is applied; in your flow, call `calc_grad_activation` first, then BN backward).
- For each feature j:
  - `dbeta_j  = Σ_i dY_ij`
  - `dgamma_j = Σ_i dY_ij * x̂_ij`
  - `dX̂_ij = dY_ij * γ_j`
  - `dvar_j  = Σ_i dX̂_ij * (x_ij − μ_j) * (−0.5) * (σ²_j + eps)^(−3/2)`
  - `dmean_j = Σ_i dX̂_ij * (−1)/sqrt(σ²_j + eps) + dvar_j * Σ_i (−2)(x_ij − μ_j)/m`
  - `dX_ij = dX̂_ij / sqrt(σ²_j + eps) + dvar_j * 2(x_ij − μ_j)/m + dmean_j/m`
- Write `dX` back into the batch buffer and then per-sample into `A->dZ` to feed `back_propogate_step` of the previous dense layer.

6) Wire backward order per layer
- For each layer L3 -> L2 -> L1, per batch:
  - You already do `back_propogate_step(prevA, L, currA)` then `calc_grad_activation(prevA, L, currA)`.
  - After `calc_grad_activation(prevA, L, currA)`, if `prevA->norm_type == BatchNorm`, run BN backward on `prevA` across the batch to transform `prevA->dZ` from post-activation to pre-BN input gradient and accumulate `dgamma/dbeta`.
  - Then continue to the previous layer.

7) Update parameters after each batch
- Dense layers: keep your `grad_accum`, `update_weights`, `zero_grad`.
- BatchNorm: SGD update per feature:
  - `gamma -= lr * dgamma / m`
  - `beta  -= lr * dbeta  / m`
- Zero BN grads after update.

8) Inference path
- Before each nonlinearity, apply BN using running stats (not batch stats):
  - `x̂ = (x − running_mean) / sqrt(running_var + eps)`
  - `y = gamma * x̂ + beta`
- Do this per sample; no batching needed for stats in inference.

9) Integrate with your code (where to change)
- `CNNs-in-C/NN-funcs/NeuralNetwork.c`:
  - Add fields and init/free logic in `init_batchnorm` and `free_batchnorm`.
  - Implement `batchnorm_forward_train(activations*)`, `batchnorm_backward(activations*)`, and `batchnorm_inference(activations*)`.
  - In `activation_function`, remove BN call for training, keep nonlinearity only. For inference, you can keep BN before the nonlinearity or make a parallel inference wrapper that applies BN then calls the existing `inference_activation_function`.
- `CNNs-in-C/main.c`:
  - Refactor the inner training loop to batch-and-layer order as described in step 3.
  - Insert BN forward calls before each activation and BN backward calls right after `calc_grad_activation` for BN-enabled activations.
  - Add BN parameter updates alongside `update_weights`.

10) Edge cases and guards
- Handle remainder batches at dataset end (m < BATCH_SIZE).
- If `m < 2`, skip variance normalization or clamp to `eps`.
- Keep `eps` reasonably large (1e-5 to 1e-3) to avoid NaNs.
- Start with training BN without AVX; validate numerics, then vectorize.

11) Minimal test plan
- Turn BN on for a single layer first (e.g., only `A2`) and train a few steps; verify loss decreases and no NaNs.
- Compare with BN off (`NormNone`) to confirm behavior change.
- Print mean/var of a feature during training to confirm they’re reasonable and running stats converge.
- Validate inference path uses running stats and doesn’t depend on `count`.

If you want, I can point to exact spots in your files to adjust the training loop and sketch the function signatures and call order so you can fill in the math.



So I would have to remove the sequential for loop and modify the training loop to keep my original idea the same. I would like to keep the image input idea the same because it makes convolution operations very easy. I will look at the multidimentional images later on, maybe just have a custom function for it. I would have to call batchnorm, accumulate values, compute batch statistics, normalize in place, scale and shift, store in xhat, then call activation function, then forward propogates to the next step then do the same accumulate gradients.   