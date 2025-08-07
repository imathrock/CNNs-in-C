# CNNs-in-C

**CNNs-in-C** builds upon the "MNIST from scratch in C" project, adding significant performance improvements and support for convolutional neural networks. This version focuses on increasing training speed, improving data structures, and incorporating low-level optimizations such as SIMD vectorization and cache-friendly memory layouts.

Refer to the Documentation in the Docs folder provided. All of the code for this project was written without the help of any AI except for the documentation for each function and some menial repetitive switch case statements. 

## Features

### Custom .idx file parser
  Wrote a custom .idx file parser for this network.

### Core Neural Network Enhancements

* **2D Convolution and Pooling**
  Implemented `conv2d` and pooling operations to support convolutional neural networks.

* **Improved Image Data Structure**
  Introduced a new structure for handling image data more efficiently, it now stores the metadata for Unpooling more efficiently.

* **Optimized Weight Storage**
  Changed layer weights from 2D arrays in previous project to flattened 1D arrays for:

  * Improved cache performance
  * Simplified transpose operations
  * Easier integration with SIMD vectorization

### Performance Improvements

* **AVX SIMD Vectorization**
  Vectorized dense layer operations using AVX 256-bit instructions, resulting in:

  * Approximately 2 to 3 times speedup on AMD CPUs
  * Approximately 1.5 to 2 times speedup on Intel CPUs
    This optimization supports all CPUs released after 2013.
    Note: Intel CPUs currently perform better, possibly due to more optimized SIMD support or compiler behavior. I dont know why yet.

* **OpenMP Multithreading (Experimental)**
  Multithreading was explored using OpenMP, but the overhead from thread creation outweighed the benefits for small models. The project currently remains single-threaded.

## Model Accuracy

* Fashion MNIST: 91%
* MNIST: 94%

These results can likely be improved further. There may be some performance degradation due to internal covariate shift.

## Upcoming Features

* SIMD-based batch normalization
* A CIFAR-10 model
* A Galaxy Image classifier (The whole reason I started this whole thing)

## Planned Improvements

* Improved user interface for easier usage
* A dedicated inference-only deployment project
* OpenCL acceleration for matrix multiplication
* Introduction of a tensor abstraction to simplify operations (potentially in a separate project)
* Auto-differentiation support

## Getting Started

To build and run the project:

```bash
make
./main
```

Ensure your system supports AVX256 (any CPU released after 2013 should be compatible). Any improvement suggestions are welcome.