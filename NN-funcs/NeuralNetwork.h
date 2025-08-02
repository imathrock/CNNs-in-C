#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "dataloaders/idx-file-parser.h"

// Structure containing pointers to Weights and biases of the layers.
typedef struct layer {
    float* Weights;
    float* biases;
    int rows;
    int cols;
} layer;

// Simple float array that holds activation values of the layers.
typedef struct activation {
    float* activation;
    int size;
} activation;

// bunched activations struct
typedef struct{
    activation*Z;
    activation*del_Z;
    activation*dZ;
}activations;

typedef struct{
    layer*params;
    layer*param_grad;
    layer*param_grad_sum;
}DenseLayer;

// Initializes a layer struct with guards in place to prevent memory leak.
struct layer* init_layer(int rows, int cols);

// Frees the Layer struct.
void free_layer(struct layer* Layer);

// Dense layer init func
DenseLayer*init_DenseLayer(int rows,int cols);

// DenseLayer finalizer
void Free_DenseLayer(DenseLayer*DL);

// Initializes and creates a float array to store the activation in.
struct activation* init_activation(int size);

// Frees the activation struct.
void free_activation(struct activation* a);

// activations init func
activations*init_activations(int size);

// activations finalizer
void Free_activations(activations*A);

// Efficient Forward prop function.
void forward_prop_step(struct activation* A1, struct layer* L, struct activation* A2);

// Applies ReLU to the activation.
void ReLU(struct activation* A);

// Takes Derivative of ReLU and puts it in the other activation struct.
void ReLU_derivative(struct activation*A,struct activation*B);

// Applies Softmax to the activation layer.
void softmax(struct activation* A);

// One hot encodes the error function.
float* one_hot_encode(int k);

// Loss function that tells us the error values.
void loss_function(struct activation* dZ_loss,struct activation* Fl, int k);

// Computes the cross-entropy loss between predicted activation and the true label.
float compute_loss(struct activation* Fl, int k);

// Calculates Gradient in activation given previous gradient.
void calc_grad_activation(struct activation* dZ_curr, struct layer* L, struct activation* dZ_prev, struct activation* A_curr);

// Conducts 1 step of back propagation and also updates parameters immediately.
void back_propogate_step(struct layer* L, struct layer* dL, struct activation* dZ, struct activation* A);

// Given original weights, biases and gradient, updates all the values accordingly.
void param_update(struct layer* L, struct layer* dL, float Learning_Rate);

// Clears the Given layer.
void Zero_Layer(struct layer* L);

// Inputs image data into activation struct.
void input_data(struct pixel_data* pixel_data, int k, struct activation* A);

// Gets the largest activation value and returns it.
int get_pred_from_softmax(struct activation* A);

// Prints out activation values for debugging.
void print_activation(struct activation* A);

// Prints the contents of a layer struct.
void print_layer(const struct layer* l);

// Shows the image at kth index.
void show_image(struct pixel_data* pixel_data, int k);

#endif // NEURAL_NETWORK_H