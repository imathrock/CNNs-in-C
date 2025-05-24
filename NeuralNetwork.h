#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

// Forward declaration of the pixel_data struct (assuming it's defined in idx-file-parser.h)
struct pixel_data;

// Structure containing pointers to Weights and biases of the layers.
typedef struct layer {
    float** Weights;
    float* biases;
    int rows;
    int cols;
} layer;

// Simple float array that holds activation values of the layers.
typedef struct activations {
    float* activations;
    int size;
} activations;

// Definition of a kernel that will be slid over the entire image.
typedef struct kernel {
    int size;
    float* weights;
} kernel;

// Initializes a layer struct with guards in place to prevent memory leak.
struct layer* init_layer(int rows, int cols);

// Frees the Layer struct.
void free_layer(struct layer* Layer);

// Initializes and creates a float array to store the activations in.
struct activations* init_activations(int size);

// Frees the activation struct.
void free_activations(struct activations* a);

// Efficient Forward prop function.
void forward_prop_step(struct activations* A1, struct layer* L, struct activations* A2);

// Applies ReLU to the activations.
void ReLU(struct activations* A);

// Takes Derivative of ReLU and puts it in another activation struct.
void ReLU_derivative(struct activations* A, struct activations* B);

// Applies Softmax to the activation layer.
void softmax(struct activations* A);

// One hot encodes the error function.
float* one_hot_encode(int k);

// Loss function that tells us the error values.
void loss_function(struct activations* dZ_loss, struct activations* Fl, int k);

// Computes the cross-entropy loss between predicted activations and the true label.
float compute_loss(struct activations* Fl, int k);

// Calculates Gradient in activations given previous gradient.
void calc_grad_activation(struct activations* dZ_curr, struct layer* L, struct activations* dZ_prev, struct activations* A_curr);

// Conducts 1 step of back propagation and also updates parameters immediately.
void back_propogate_step(struct layer* L, struct layer* dL, struct activations* dZ, struct activations* A);

// Given original weights, biases and gradient, updates all the values accordingly.
void param_update(struct layer* L, struct layer* dL, float Learning_Rate);

// Clears the Given layer.
void Zero_Layer(struct layer* L, float num);

// Inputs image data into activation struct.
void input_data(struct pixel_data* pixel_data, int k, struct activations* A);

// Gets the largest activation value and returns it.
int get_pred_from_softmax(struct activations* A);

// Prints out activation values for debugging.
void print_activations(struct activations* A);

// Prints the contents of a layer struct.
void print_layer(const struct layer* l);

// Shows the image at kth index.
void show_image(struct pixel_data* pixel_data, int k);

#endif // NEURAL_NETWORK_H