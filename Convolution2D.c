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
    uint_fast8_t*Data;
}Image2D;

// void Convolution2D(){return;}

/// @brief Operates POOLing function on the image and returns it
/// @param type Type of pooling
/// @param image the image to do pooling on
/// @param ker_size size of the kernel that slides over the image
/// @return Pooled image
Image2D POOL(char type, Image2D image, int ker_size, int stride){
    Image2D ret_img;
    ret_img.rows = (image.rows-ker_size)/stride+1;
    ret_img.cols = (image.cols-ker_size)/stride+1;
    ret_img.Data = calloc(sizeof(unsigned char),image.cols*image.rows);
    if(type == 1){
        for(int i = 0; i<ret_img.rows;i++){
            for (int j = 0; j < ret_img.cols; j++){
                int max = 0;
                for (int ki = 0; ki < ker_size; ki++){
                    for(int kj = 0; kj < ker_size; kj++){
                        int idx = (((i*stride)+ki)*image.cols)+((j*stride)+kj);
                        if (image.Data[idx] > max) {max = image.Data[idx];}
                    }
                }
                ret_img.Data[i*ret_img.cols+j] = max;
            }
        }
    }
    if(type == 2){
        for(int i = 0; i<ret_img.rows;i++){
            for (int j = 0; j < ret_img.cols; j++){
                int avg = 0;
                for (int ki = 0; ki < ker_size; ki++){
                    for(int kj = 0; kj < ker_size; kj++){
                        int idx = (((i*stride)+ki)*image.cols)+((j*stride)+kj);
                        avg += image.Data[idx];}
                }
                avg /= ker_size*ker_size;
                ret_img.Data[i*ret_img.cols+j] = avg;
            }
        }
    }
    return ret_img;
}
