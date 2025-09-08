# List of tasks for this projects

## Implement Batchnorm
To implement batchnorm what i have currently done is accumulate the pre activation values over a batch, calculated batch statistics, normalized and scaled them all when i call the activation function, checking for normalization type and calling batchnorm. Now after that activation function directly applies the activation function over all the values. 

- Need to implement the training forward propagation function for batchnorm. Check for norm type and then call a different function called Forwardprop batchnorm 
- Need to implement an inference function that uses running mean and variance 
    - Something like inference activation calls batchnorm inference
    - Uses normal forward propagate function for inference because there is no need for batches. 

