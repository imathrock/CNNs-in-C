#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<math.h>
#include"NeuralNetwork.h"
#include"idx-file-parser.h"


/// @brief Struct to contain images of any shapes and sizes.
typedef struct{
    int rows;
    int cols;
    unsigned char*Data;
    unsigned char type;
}Image2D;

// void Convolution2D(){return;}

/// @brief Operates POOLing function on the image and returns it
/// @param type Type of pooling
/// @param image the image to do pooling on
/// @param ker_size size of the kernel that slides over the image
/// @return Pooled image
Image2D POOL(char type, Image2D image, int ker_size){
    if(type == 1){
        
    }
    return image;
}