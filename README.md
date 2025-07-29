# CNNs-in-C
Bulding upon the MNIST from scratch in C project, this project is an upgrade in the capabilities and speed of my codebase. The following are the list of things added:

## Features:
- Convolution 2D functions like conv2d and pooling. 
- new image data struct.
- modified layer weights from being 2D to 1D for better cache friendliness, speed and simplicity of transpose operation.
- AVX SIMD vectorization for faster training.
    - achieved about 2x speedup on AMD cpus and 1.3x on intel from just vectorizing MLP functions.
    - Uses AVX 256, so it supports all computers built after 2013.
    - side note: this code is faster on intel CPUs because either intel's SIMD is better optimized or the compiler is better, haven't figured it out exactly why yet.

## Features to be added:
- multiple convolution layers
- better user interface so that using it is easier.
- A way to deploy the model for inference. (probably in a new project)
- multithreading and openCL to make it faster. (CUDA done on a Nvdia GPU but i dont have access to one)