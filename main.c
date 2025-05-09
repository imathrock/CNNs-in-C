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

    Image2D test_image;
    test_image.rows = pixel_data->rows;
    test_image.cols = pixel_data->cols;
    test_image.Data = (float*)pixel_data->neuron_activation[0];

    Normalize_Image(test_image);

    Image2D kernel;
    kernel.rows = 2;
    kernel.cols = 2;
    kernel.Data = malloc(sizeof(float)*kernel.rows*kernel.cols);
    for (int i = 0; i < kernel.rows*kernel.cols; i++){
        kernel.Data[i] = 1;
    }
    

    clock_t start = clock();
    Image2D convimg = Conv2D(kernel,test_image);
    Image2D retimg = POOL(1,convimg,2,2);
    clock_t end = clock();
    for (int i = 0; i < retimg.rows; i++) {
        for (int j = 0; j < retimg.cols; j++) {
            int pixel = retimg.Data[i * retimg.cols + j];
            if (pixel > 0) {
                printf("# ");
            } else {
                printf(". ");
            }
        }
        printf("\n");
    }
    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Function took %.6f seconds\n", time_taken);
    
    
    return 1;
}
