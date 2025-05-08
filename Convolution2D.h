#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<math.h>

/// @brief Struct to contain images of any shapes and sizes.
typedef struct{
    int rows;
    int cols;
    uint_fast8_t*Data;
}Image2D;

Image2D Conv2D(Image2D Kernel, Image2D image);

Image2D POOL(char type, Image2D image, int ker_size,int stride);