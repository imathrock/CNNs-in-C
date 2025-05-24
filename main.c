#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<math.h>
#include<time.h>
#include"dataloaders/idx-file-parser.h"
#include"Convolution2D.h"
#include"NN-funcs/NeuralNetwork.h"

#define BATCH_SIZE 10


int main(){
    FILE* file = fopen("data/train-images.idx3-ubyte", "rb");
    struct pixel_data* pixel_data = get_image_pixel_data(file);
    fclose(file);
    file = fopen("data/train-labels.idx1-ubyte", "rb");
    unsigned char* lbl_arr = get_image_labels(file);
    fclose(file);
    
    // FCN details
    int inpl = 128;
    int lay2 = 64;
    int oupl = 10;

    // Defining Neural Network
    struct activations*AL1 = init_activations(inpl);// input
    struct layer*L1 = init_layer(lay2,inpl);
    struct activations*AL2 = init_activations(lay2);// hidden
    struct layer*L2 = init_layer(oupl,lay2);
    struct activations*AL3 = init_activations(oupl);// output

    struct activations* dZAL2 = init_activations(lay2);// hidden layer loss

    struct layer*dL1 = init_layer(lay2,inpl);
    struct layer*dL2 = init_layer(oupl,lay2);

    struct layer*sdL1 = init_layer(lay2,inpl);
    struct layer*sdL2 = init_layer(oupl,lay2);



    Image2D test_image = CreateImage(pixel_data->rows,pixel_data->cols,pixel_data->neuron_activation[0]);

    Image2D kernel1 = CreateKernel(12,12);
    for (int i = 0; i < kernel1.rows*kernel1.cols; i++){
        kernel1.Data[i] = ((float)rand()/((float)RAND_MAX) - 0.5);
    }

    Image2D kernel2 = CreateKernel(12,12);
    for (int i = 0; i < kernel2.rows*kernel2.cols; i++){
        kernel2.Data[i] = ((float)rand()/((float)RAND_MAX) - 0.5);
    }



    // unpooling metadata
    int*UPMD1;
    int*UPMD2;

    // convolution with kernel1b
    Image2D convimg1 = Conv2D(kernel1,test_image);
    // mallocing the metadata array
    UPMD1 = malloc(sizeof(int)*convimg1.cols*convimg1.rows);
    // maxpooling
    Image2D retimg1 = POOL(1,convimg1,2,2,UPMD1);

    // same as above
    Image2D convimg2 = Conv2D(kernel2,test_image);
    UPMD2 = malloc(sizeof(int)*convimg2.cols*convimg2.rows);
    Image2D retimg2 = POOL(2,convimg2,2,2,UPMD2);

    // Flatten function
    for (int i = 0; i < AL1->size/2; i++){
        AL1->activations[i] = retimg1.Data[i];
    }
    for (int i = AL1->size/2; i < AL1->size; i++){
        AL1->activations[i] = retimg2.Data[i];
    }

    // Forward prop test
    forward_prop_step(AL1,L1,AL2);
    ReLU(AL2);
    forward_prop_step(AL2,L2,AL3);
    softmax(AL3);
    loss_function(AL3,lbl_arr[0]); // AL3 now has loss 

    back_propogate_step(L2,dL2,AL3,AL2); // (layer, deriv layer, loss, prev activations)
    ReLU_derivative(AL2); // takes ReLU deriv and stored it in AL2 itself
    calc_grad_activation(dZAL2,L2,AL3,AL2);
    back_propogate_step(L1,dL1,dZAL2,AL1);
    param_update(sdL1,dL1,1);
    param_update(sdL2,dL2,1);

    // TODO UNPOOLING IMPLEMENTATION. 

    return 1;
}
