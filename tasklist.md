# List of tasks for this projects

## Implement Batchnorm
Batchnorm forward propagation seems to work I just need to test it a bit more. 

- Need to implement an inference function that uses running mean and variance 
    - Something like inference activation calls batchnorm inference
    - Uses normal forward propagate function for inference because there is no need for batches. 

- For backward propagation use the same idea as forward prop. Call a batchnorm backprop after calculation of gradients. 

UI design idea, Have a full NN struct where you are able to sequentially iterate thru the layers.
The counter should be in the NN struct itself and the user can just go nn.counter++ and counter = 0 later. 
If i could load a batch directly it would be great, problem to solve later. 