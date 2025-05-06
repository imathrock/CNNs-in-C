#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<math.h>

/// @brief Struct to contain images of any shapes and sizes.
typedef struct{
    int rows;
    int cols;
    unsigned char*Data;
    unsigned char type;
}Image2D;

// void Convolution2D()

Image2D POOL(char type, Image2D image, int ker_size);