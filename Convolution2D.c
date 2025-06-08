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
void ImageReLU(Image2D image){
    for(int i=0; i<image.cols*image.rows;i++){
        if(image.Data[i] < 0.0){image.Data[i] = 0.0;}
    }
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

    if (type == 1) {
        for (int i = 0; i < ret_img.rows; i++) {
            for (int j = 0; j < ret_img.cols; j++) {
                float max = -INFINITY;
                int max_idx = -1;
                for (int ki = 0; ki < ker_size; ki++) {
                    int in_row = i * stride + ki;
                    if (in_row >= image.rows) continue;
                    for (int kj = 0; kj < ker_size; kj++) {
                        int in_col = j * stride + kj;
                        if (in_col >= image.cols) continue;
                        int flat_idx = in_row * image.cols + in_col;
                        float val = image.Data[flat_idx];
                        if (val > max) {
                            max = val;
                            max_idx = flat_idx;
                            UPMD[i * ret_img.cols + j] = max_idx;
                        }
                    }
                }
                ret_img.Data[i * ret_img.cols + j] = max;
            }
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

/// @brief Unpools the image from a pooled image and with metadata
/// @param unpooled return part
/// @param pooled pooled image
/// @param UPMD metadata
void UNPOOL(Image2D unpooled,Image2D pooled, int*UPMD){
    for(int i = 0; i < unpooled.rows*unpooled.cols; i++){unpooled.Data[i] = 0.0f;}    
    for(int i = 0; i < pooled.rows*pooled.cols; i++){unpooled.Data[UPMD[i]] = pooled.Data[i];}
}

/**
 * @brief Updates the convolution kernel using backpropagation.
 *
 * Computes weight gradients from the input and upstream gradients,
 * then updates the kernel with gradient descent.
 *
 * @param Kernel         Kernel to be updated (in-place).
 * @param Unpooled       Gradient from the next layer.
 * @param Image          Original input image/feature map.
 * @param learning_rate  Learning rate for update.
 */
void backprop_kernel(Image2D delta_kernel,Image2D Kernel, Image2D Unpooled, Image2D Image){
    for(int i = 0; i<Unpooled.rows;i++){
        for (int j = 0; j < Unpooled.cols; j++){
            int curidx = i*Unpooled.cols+j;
            for(int x = 0; x < Kernel.rows; x++){
                int ridx = (i+x)*Image.cols;
                for(int y = 0; y < Kernel.cols; y++){
                    int cidx = j+y;
                    delta_kernel.Data[x*Kernel.cols+y] += Unpooled.Data[curidx]*Image.Data[ridx+cidx];
                }
            }
        }
    }
}

/// @brief Updates the kernel parameters using the computed gradients.
/// @param delta_kernel Gradient of the kernel. 
void kernel_update(Image2D delta_kernel, Image2D Kernel, float learning_rate){
    for(int i = 0; i < Kernel.cols*Kernel.rows; i++){
        Kernel.Data[i] -= learning_rate*delta_kernel.Data[i];
    }
}

/// @brief Sets all the values in the kernel to zero
/// @param Kernel The kernel to be zeroed out.
void zero_kernel(Image2D Kernel){
    for(int i = 0; i < Kernel.cols*Kernel.rows; i++){
        Kernel.Data[i] = 0.0f;
    }
}