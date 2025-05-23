#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<math.h>
#include"Convolution2D.h"

Image2D CreateImage(int rows, int cols, uint8_t*Data){
    Image2D image;
    image.rows = rows;
    image.cols = cols;
    image.Data = (float*)malloc(sizeof(float)*rows*cols);
    int k = rows*cols;
    for (int i = 0; i < k; i++) {
        image.Data[i] = (float)(Data[i]) / 255.0f;
    }
    return image;
}

Image2D CreateKernel(int rows, int cols){
    Image2D kernel;
    kernel.rows = rows;
    kernel.cols = cols;
    kernel.Data = (float*)malloc(sizeof(float)*rows*cols);
    return kernel;
}

/// @brief Normalizes the images pixel values
/// @param image 
void Normalize_Image(Image2D image){
    for(int i=0; i<image.cols*image.rows;i++){image.Data[i] /= 255.0f;}
}

/// @brief Conducts Convolution operation with the provided kernel and returns the image
/// @param Kernel slides over the entire image
/// @param image 
/// @return convoluted image
Image2D Conv2D(Image2D Kernel, Image2D image){
    Image2D ret_img;
    ret_img.rows = image.rows-Kernel.rows;
    ret_img.cols = image.cols-Kernel.cols;
    ret_img.Data = calloc(sizeof(float),ret_img.cols*ret_img.rows);
    for(int i = 0; i<ret_img.rows;i++){
        for (int j = 0; j < ret_img.cols; j++){
            float dotprod = 0;
            for (int ki = 0; ki < Kernel.rows; ki++){
                int ridx = ((i+ki)*image.cols);
                for(int kj = 0; kj < Kernel.cols; kj++){
                    int cidx = (j+kj);
                    dotprod += image.Data[ridx+cidx]*Kernel.Data[(ki*Kernel.cols)+kj];
                }
            }
            ret_img.Data[i*ret_img.cols+j] = dotprod;
        }
    }
    return ret_img;
}

/// @brief Operates POOLing function on the image and returns it
/// @param type Type of pooling
/// @param image the image to do pooling on
/// @param ker_size size of the kernel that slides over the image
/// @return Pooled image
Image2D POOL(char type, Image2D image, int ker_size, int stride, int*UPMD){
    Image2D ret_img;
    ret_img.rows = (image.rows-ker_size)/stride+1;
    ret_img.cols = (image.cols-ker_size)/stride+1;
    ret_img.Data = calloc(sizeof(float),ret_img.cols*ret_img.rows);
    if(type == 1){
        for(int i = 0; i<ret_img.rows;i++){
            for (int j = 0; j < ret_img.cols; j++){
                float max = 0.0f;
                for (int ki = 0; ki < ker_size; ki++){
                    int ridx = (((i*stride)+ki)*image.cols);
                    for(int kj = 0; kj < ker_size; kj++){
                        int cidx = ((j*stride)+kj);
                        if(ridx > image.rows || cidx > image.cols){}
                        if (image.Data[ridx+cidx] > max){
                            max = image.Data[ridx+cidx];
                            UPMD[i*ret_img.cols+j] = ridx+cidx;
                        }
                    }
                }
                ret_img.Data[i*ret_img.cols+j] = max;}
        }
    }
    if(type == 2){
        for(int i = 0; i<ret_img.rows;i++){
            for (int j = 0; j < ret_img.cols; j++){
                float avg = 0.0f;
                for (int ki = 0; ki < ker_size; ki++){
                    int ridx = (((i*stride)+ki)*image.cols);
                    for(int kj = 0; kj < ker_size; kj++){
                        int cidx = ((j*stride)+kj);
                        avg += image.Data[ridx+cidx];}
                }
                avg /= ker_size*ker_size;
                ret_img.Data[i*ret_img.cols+j] = avg;}
        }
    }
    return ret_img;
}
