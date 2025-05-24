#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<math.h>
#include<time.h>
#include"idx-file-parser.h"
#include"Convolution2D.h"
#include"NeuralNetwork.h"

#define BATCH_SIZE 10


int main(){
    FILE* file = fopen("data/train-images.idx3-ubyte", "rb");
    struct pixel_data* pixel_data = get_image_pixel_data(file);
    printf("\n");

    Image2D test_image = CreateImage(pixel_data->rows,pixel_data->cols,pixel_data->neuron_activation[0]);

    Image2D kernel1 = CreateKernel(4,4);
    for (int i = 0; i < kernel1.rows*kernel1.cols; i++){
        kernel1.Data[i] = ((float)rand()/((float)RAND_MAX) - 0.5)*100;
    }
    
    Image2D kernel2 = CreateKernel(4,4);
    for (int i = 0; i < kernel2.rows*kernel2.cols; i++){
        kernel2.Data[i] = ((float)rand()/((float)RAND_MAX) - 0.5)*100;
    }


    // unpooling metadata
    int*UPMD1;
    int*UPMD2;

    // convolution with kernel1
    Image2D convimg1 = Conv2D(kernel1,test_image);
    // mallocing the metadata array
    UPMD1 = malloc(sizeof(int)*convimg1.cols*convimg1.rows);
    // maxpooling
    Image2D retimg1 = POOL(1,convimg1,2,2,UPMD1);

    // same as above
    Image2D convimg2 = Conv2D(kernel2,test_image);
    UPMD2 = malloc(sizeof(int)*convimg2.cols*convimg2.rows);
    Image2D retimg2 = POOL(1,convimg2,2,2,UPMD2);


    // Defining 1 layer for forward prop
    struct activations*AL1 = init_activations(retimg1.cols*retimg1.rows*2);
    struct layer*L1 = init_layer(retimg2.rows,retimg1.cols*retimg1.rows*2);
    struct activations*AL2 = init_activations(retimg1.rows);
    
    // Flatten function
    for (int i = 0; i < AL1->size/2; i++){
        retimg1.Data[i] = AL1->activations[i];
    }
    for (int i = AL1->size/2; i < AL1->size; i++){
        retimg2.Data[i] = AL1->activations[i];
    }
    // Forward prop test
    forward_prop_step(AL1,L1,AL2);
    print_activations(AL2);
    
    return 1;
}
