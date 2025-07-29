#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<math.h>
#include"NN-funcs/NeuralNetwork.h"

/// @brief Struct to contain images of any shapes and sizes.
typedef struct{
    int rows;
    int cols;
    float *Data;
    int* maxidx;
}Image2D;

Image2D CreateImage(int rows, int cols);

Image2D CreateKernel(int rows, int cols);

void ImageReLU(Image2D image);

Image2D Conv2D(Image2D Kernel, Image2D image);

Image2D POOL(char type, Image2D image, int ker_size,int stride,int*UPMD);

void ImageInput(Image2D image, uint8_t*Data);

void UNPOOL(Image2D unpooled,Image2D pooled, int*upmd);

void backprop_kernel(Image2D delKernel,Image2D Kernel, Image2D Unpooled, Image2D Image);

void kernel_update(Image2D delta_kernel, Image2D Kernel, float learning_rate);

void zero_kernel(Image2D Kernel);