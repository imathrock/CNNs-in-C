#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdio.h>
#include <stdlib.h>=
#include <stdint.h>
#include <math.h>
#include "dataloaders/idx-file-parser.h"

typedef enum{
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

// Structure containing pointers to Weights and biases of the layers.
typedef struct layer {
    float* Weights;
    float* biases;
} layer;

// bunched activations struct
typedef struct{
    int size;
    float*Z;
    float*dZ; // derivative
    float*del_Z; // gradient
}activations;

typedef struct{
    int init_type;
    int rows;
    int cols;
    layer*params;
    layer*param_grad;
    layer*param_grad_sum;
}DenseLayer;

// Initializes a layer struct with guards in place to prevent memory leak.
layer* init_layer(int rows, int cols);

// Frees the Layer struct.
void free_layer(struct layer* Layer);

// Dense layer init func
DenseLayer*init_DenseLayer(int rows,int cols);

// DenseLayer finalizer
void Free_DenseLayer(DenseLayer*DL);

// activations init func
activations*init_activations(int size);

// activations finalizer
void Free_activations(activations*A);

// Efficient Forward prop function.
void forward_prop_step(struct activation* A1, struct layer* L, struct activation* A2);

/// @brief Applies enum act_func, simultaneously computes derivatives and stores it.
/// @param A Activations struct
/// @param func act func, default identity.
void activation_function(activations*A,act_func_t func)


float loss_function(activations*A, loss_func_t func, int k)


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

// test StandardizeActivations
void StandardizeActivations(activation *A);

// Prints out activation values for debugging.
void print_activation(struct activation* A);

// Prints the contents of a layer struct.
void print_layer(const struct layer* l);

// Shows the image at kth index.
void show_image(struct pixel_data* pixel_data, int k);

#endif // NEURAL_NETWORK_H