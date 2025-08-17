#ifndef CONV2D_H
#define CONV2D_H

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

typedef struct{
    int rows;
    int cols;
    float*K;
    float*del_K;
    float*sum_del_K;
}kernel;

Image2D CreateImage(int rows, int cols);

kernel CreateKernels(int rows, int cols);

Image2D CreateConvImg(Image2D img, kernel K);

Image2D CreatePoolImg(Image2D img, int ker_size, int stride);

void ImageReLU(Image2D image);

void Conv2D(kernel Kernel, Image2D image, Image2D convimg);

void MAXPOOL(Image2D image,Image2D poolimg, int ker_size, int stride);

void ImageInput(Image2D image, uint8_t*Data);

void MAXUNPOOL(Image2D unpooled,Image2D pooled);

void backprop_kernel(kernel Kernel, Image2D Unpooled, Image2D Image);

void kernel_update(kernel Kernel, float learning_rate);

void kernel_accum(kernel Kernel,float batch_size);

void zero_kernel(kernel Kernel);

#endif