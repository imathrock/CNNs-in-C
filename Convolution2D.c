#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<math.h>
#include"Convolution2D.h"

#include "immintrin.h"

#define Img_Data(I,i,j) (I->Data[((i)*(I->cols))+(j)])
#define Img_Data_Obj(I,i,j) ((I).Data[((i)*(I).cols)+(j)])


Image2D CreateImage(int rows, int cols){
    Image2D image;
    image.rows = rows;
    image.cols = cols;
    image.Data = (float*)malloc(sizeof(float)*rows*cols);
    image.maxidx = NULL;
    return image;
}

Image2D CreateKernel(int rows, int cols){
    Image2D kernel;
    kernel.rows = rows;
    kernel.cols = cols;
    kernel.Data = (float*)malloc(sizeof(float)*rows*cols);
    for(int i = 0; i<rows*cols; i++){
        kernel.Data[i] = (float)rand()/((float)RAND_MAX) - 0.5;
    }
    kernel.maxidx = NULL;
    return kernel;
}

void ImageInput(Image2D image, uint8_t*Data){
    int k = image.rows*image.cols;
    for (int i = 0; i< k; i++) {
        image.Data[i] = (float)(Data[i]) / 255.0f;
    }
}

/// @brief Normalizes the images pixel values
/// @param image 
void ImageReLU(Image2D image){
    for(int i=0; i<image.cols*image.rows;i++) image.Data[i] = fmax(0.0f,image.Data[i]);
}

/// @brief Conducts Convolution operation with the provided kernel and returns the image
/// @param Kernel slides over the entire image
/// @param image 
/// @return convoluted image
Image2D Conv2D(Image2D Kernel, Image2D image){
    Image2D ret_img = CreateImage(image.rows-Kernel.rows,image.cols-Kernel.cols);
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
Image2D MAXPOOL(Image2D image, int ker_size, int stride){
    Image2D ret_img = CreateImage((image.rows-ker_size)/stride+1,(image.cols-ker_size)/stride+1);
    ret_img.maxidx = malloc(ret_img.cols*ret_img.rows*sizeof(int));
    memset(&ret_img.maxidx,-1,ret_img.cols*ret_img.rows);
    //looping over return image
    for(int i = 0; i < ret_img.rows; i++){
        for (int j = 0; j < ret_img.cols; j++){

            float max = -INFINITY;
            float idxmax = -1;
            
            // stride indexing
            int strx = i*stride;
            int stry = j*stride;

            //looping over kernel size
            for (int x = 0; x < ker_size; x++){
                if(strx+x>=image.rows) continue; //bounds check
                for(int y = 0; y < ker_size; y++){  
                    if(stry+y>=image.cols) continue; 
                    // max check
                    float curr = Img_Data_Obj(image,strx+x,stry+y); // to prevent double loading
                    if(curr > max){
                        max = curr;
                        idxmax = ((strx+x)*image.cols+(stry+y));
                    }

                }
            }
            Img_Data_Obj(ret_img,i,j) = max;
            ret_img.maxidx[i*ret_img.cols+j] = idxmax;
        }
    }
    return ret_img;
}

/// @brief Unpools the image from a pooled image and with metadata
/// @param unpooled return part
/// @param pooled pooled image
/// @param UPMD metadata
void UNMAXPOOL(Image2D unpooled, Image2D pooled) {
    // Zero out the unpooled image
    memset(unpooled.Data,0.0f,sizeof(float)*unpooled.rows*unpooled.cols);
    // Unpool with bounds checking
    for(int i = 0; i < pooled.rows*pooled.cols; i++) {
        if(pooled.maxidx[i] < 0 || pooled.maxidx[i] > unpooled.rows*unpooled.cols) continue;
        else{
            unpooled.Data[pooled.maxidx[i]] = pooled.Data[i];
        }
    }
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
    __m256 lr_vec = _mm256_set1_ps(learning_rate);
    int i = 0;
    for(; i+7 < Kernel.cols*Kernel.rows; i+=8){
        __m256 v_ker = _mm256_loadu_ps(&Kernel.Data[i]);
        __m256 v_dker = _mm256_loadu_ps(&delta_kernel.Data[i]);
        __m256 mulvec = _mm256_mul_ps(lr_vec,v_dker);
        v_ker = _mm256_sub_ps(mulvec,v_ker);
        _mm256_storeu_ps(&Kernel.Data[i],v_ker);
    }
    for (; i < Kernel.cols*Kernel.rows; i++){
        Kernel.Data[i] -= learning_rate*delta_kernel.Data[i];
    }
    
}

/// @brief Sets all the values in the kernel to zero
/// @param Kernel The kernel to be zeroed out.
void zero_kernel(Image2D Kernel){
    memset(Kernel.Data,0.0f,Kernel.cols*Kernel.rows);
}