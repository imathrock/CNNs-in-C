#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdio.h>
#include <stdlib.h>
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
    L1loss,
    L2loss,
    CE,     
    MSE,    
    MAE,    
    HUBER,  
    BCE,    
    CCE,    
    SCE     
} loss_func_t;

// Enums for initializing the type of normalization. 
typedef enum{
    NormNone,
    BatchNorm,
    LayerNorm,
}norm_type_t;

// Structure containing pointers to Weights and biases of the layers.
typedef struct {
    float* Weights;
    float* Weights_T;
    float* biases;
} layer;

typedef struct{
    int init_type;
    int rows;
    int cols;
    layer*params;
    layer*param_grad;
    layer*param_grad_sum;
}DenseLayer;

// Batchnorm struct
typedef struct{
    int count;
    float*mean; // [features]
    float*var;// [features]
    float*run_mean; // [features]
    float*run_var;// [features]
    float momentum;
    float*gamma;// [features]
    float*beta;// [features]
    float*dgamma;// [features]
    float*dbeta;// [features]
}batchnorm_t;

// Layernorm struct
typedef struct{
    int num_features;
    float*gamma;// [features]
    float*beta;// [features]
    float*dgamma;// [features]
    float*dbeta;// [features]
}layernorm_t;

// bunched activations struct
typedef struct{
    int size;
    int batch_size;
    float*raw;// rawdog values
    float*Z;  // activation values
    float*gprime; // activation derivative
    float*dZ; // gradient
    norm_type_t norm_type;
    union {
        batchnorm_t*BN;
        layernorm_t*LN;
        void*rawdog;
    } norm_params;
}activations;

// Initializes a layer struct with guards in place to prevent memory leak.
layer* init_layer(int rows, int cols,int init_type);

// Frees the Layer struct.
void free_layer(layer* Layer);

// Dense layer init func
DenseLayer*init_DenseLayer(int rows,int cols,int init_type);

// DenseLayer finalizer
void Free_DenseLayer(DenseLayer*DL);

// activations init func
activations*init_activations(int size,int batchsize,norm_type_t norm);

// activations finalizer
void Free_activations(activations*A);

batchnorm_t* init_batchnorm(int num_features);

void free_batchnorm(batchnorm_t* BN);

layernorm_t*init_layernorm(int num_features);

void free_layernorm(layernorm_t*LN);

// Batch normalizer. 
void batchnorm_forward(activations*A);

void BatchNorm_backward(activations*A);

void BatchNorm_inference(activations*A);

void update_batchnorm_params(activations* A, float learning_rate);

// test Layernorm
void layernorm(activations *A);

// Efficient Forward prop function.
void activation_function(activations*A,act_func_t func);

void inference_activation_function(activations* A, act_func_t func);

float loss_function(activations*A, loss_func_t loss ,int k,unsigned char*l);

void forward_prop_step(activations*A1, DenseLayer*L,activations*A2);

void forward_prop_inference(activations*A1, DenseLayer*L,activations*A2);

// Calculates Gradient in activation given previous gradient.
void calc_grad_activation(activations* A_curr,DenseLayer*L,activations* A_prev);

// Conducts 1 step of back propagation and also updates parameters immediately.
void back_propogate_step(activations*A1,DenseLayer*L,activations* A2);

void grad_accum(DenseLayer* L);

void update_weights(DenseLayer*L, float LR);

void zero_grad(DenseLayer*L);

// Gets the largest activation value and returns it.
int get_pred_from_softmax(activations *A);

// Shows the image at kth index.
void show_image(struct pixel_data* pixel_data, int k);

#endif // NEURAL_NETWORK_H