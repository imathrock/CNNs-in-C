#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<math.h>
#include"idx-file-parser.h"
#include"Convolution2D.h"
#include"NeuralNetwork.h"

#define BATCH_SIZE 10

int main(){
    FILE* file = fopen("data/train-images.idx3-ubyte", "rb");
    struct pixel_data* pixel_data = get_image_pixel_data(file);
    printf("\n");

    Image2D test_image;
    test_image.rows = pixel_data->rows;
    test_image.cols = pixel_data->cols;
    test_image.Data = pixel_data->neuron_activation[0];

    Image2D retimg = POOL(2,test_image,1);

    return 1;
}
