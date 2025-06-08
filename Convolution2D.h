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
}Image2D;

Image2D CreateImage(int rows, int cols, uint8_t*Data);

Image2D CreateKernel(int rows, int cols);

void Normalize_Image(Image2D image);

Image2D Conv2D(Image2D Kernel, Image2D image);

Image2D POOL(char type, Image2D image, int ker_size,int stride,int*UPMD);

void UNPOOL(Image2D unpooled,Image2D pooled, int*upmd);

void backprop_kernel(Image2D Kernel, Image2D Unpooled, Image2D Image, float learning_rate);