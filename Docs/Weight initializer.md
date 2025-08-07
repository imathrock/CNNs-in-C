# Weight Initialization Methods Reference
Documentation and menial repetitive code written with assistance of Claude sonnet 4. I know I could have had an enum. 
## Initialization Types (`init_type` parameter)

| `init_type` | Method | Description | Best Use Case |
|-------------|--------|-------------|---------------|
| 0 | **Zero Initialization** | All weights set to 0 | Generally not recommended (symmetry problem) |
| 1 | **Random Uniform [-0.5, 0.5]** | Uniform random values between -0.5 and 0.5 | Simple baseline, not optimal |
| 2 | **Random Normal (0, 1)** | Normal distribution with mean=0, std=1 | Basic Gaussian initialization |
| 3 | **Xavier/Glorot Uniform** | Uniform in [-√(6/(fan_in+fan_out)), √(6/(fan_in+fan_out))] | Sigmoid, Tanh activation functions |
| 4 | **Xavier/Glorot Normal** | Normal with std=√(2/(fan_in+fan_out)) | Sigmoid, Tanh activation functions |
| 5 | **He/Kaiming Uniform** | Uniform in [-√(6/fan_in), √(6/fan_in)] | ReLU, Leaky ReLU activation functions |
| 6 | **He/Kaiming Normal** | Normal with std=√(2/fan_in) | ReLU, Leaky ReLU activation functions |
| 7 | **LeCun Uniform** | Uniform in [-√(3/fan_in), √(3/fan_in)] | SELU activation function |
| 8 | **LeCun Normal** | Normal with std=√(1/fan_in) | SELU activation function |
| 9 | **Orthogonal** | Orthogonal matrix initialization (simplified) | RNNs, deep networks |
| 10 | **Identity** | Identity matrix (square matrices only) | RNNs, residual connections |
| 11 | **Variance Scaling Uniform** | Uniform with variance scaling by fan_avg | General purpose |
| 12 | **Variance Scaling Normal** | Normal with variance scaling by fan_avg | General purpose |
| 13 | **Truncated Normal** | Normal distribution truncated at ±2σ | When you want bounded Gaussian |
| 14 | **Small Random** | Small uniform random values (±0.005) | When you need very small initial weights |
| Default | **Xavier/Glorot Normal** | Same as type 4 | Default fallback |

## Terminology

- **fan_in**: Number of input connections to a neuron (columns in weight matrix)
- **fan_out**: Number of output connections from a neuron (rows in weight matrix)  
- **fan_avg**: Average of fan_in and fan_out

## Recommended Usage

### By Activation Function:
- **ReLU, Leaky ReLU, ELU**: Use He initialization (types 5 or 6)
- **Sigmoid, Tanh**: Use Xavier/Glorot initialization (types 3 or 4)
- **SELU**: Use LeCun initialization (types 7 or 8)
- **Linear**: Use Xavier/Glorot initialization (types 3 or 4)

### By Network Type:
- **Feedforward Networks**: He Normal (type 6) for ReLU, Xavier Normal (type 4) for Sigmoid/Tanh
- **Convolutional Networks**: He initialization (types 5 or 6)
- **Recurrent Networks**: Orthogonal (type 9) or Identity (type 10) for hidden-to-hidden weights
- **Residual Networks**: He initialization with proper scaling

## Notes

1. **Bias Initialization**: The code initializes all biases to 0, which is the most common practice.

2. **Normal vs Uniform**: Normal distributions are generally preferred as they provide smoother gradients.

3. **Orthogonal Initialization**: The implementation provided is simplified. True orthogonal initialization requires SVD decomposition.

4. **Identity Initialization**: Only works properly when the weight matrix is square (rows == cols).

5. **Random Seed**: Remember to call `srand(time(NULL))` before using these functions to ensure different random values across runs.