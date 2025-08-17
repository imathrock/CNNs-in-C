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
    image.maxidx = (int*)malloc(image.cols*image.rows*sizeof(int));
    return image;
}

/// @brief He initialization of kernel weights, creates a struct that contains all the weights buffers needed for training
/// @param rows 
/// @param cols 
/// @return 
kernel CreateKernels(int rows, int cols){
    kernel Kernel;
    Kernel.rows = rows; Kernel.cols = cols;
    Kernel.K = (float*)malloc(sizeof(float)*rows*cols);
    Kernel.del_K =(float*)calloc(rows*cols,sizeof(float));
    Kernel.sum_del_K = (float*)calloc(rows*cols,sizeof(float));
    float limit = sqrtf(2.0f / rows);
    for (int i = 0; i < rows * cols; i++) {
        Kernel.K[i] = ((float)rand() / RAND_MAX) * limit;
    }
    return Kernel;
}

/// @brief 
/// @param img 
/// @param kernel 
/// @return 
Image2D CreateConvImg(Image2D img, kernel K){
    Image2D image;
    image.rows = img.rows-K.rows;
    image.cols = img.cols-K.cols;
    image.Data = (float*)malloc(sizeof(float)*image.rows*image.cols);
    image.maxidx = (int*)malloc(image.cols*image.rows*sizeof(int));
    return image;
}

/// @brief 
/// @param img 
/// @param ker_size 
/// @param stride 
/// @return 
Image2D CreatePoolImg(Image2D img, int ker_size, int stride){
    Image2D image;
    image.rows = (img.rows-ker_size)/stride+1;
    image.cols = (img.cols-ker_size)/stride+1;
    image.Data = (float*)malloc(sizeof(float)*image.rows*image.cols);
    image.maxidx = (int*)malloc(image.cols*image.rows*sizeof(int));
    return image;
}

/// @brief 
/// @param image 
/// @param Data 
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
void Conv2D(kernel Kernel, Image2D image, Image2D convimg){
    if (convimg.rows != (image.rows-Kernel.rows) || convimg.cols != (image.cols-Kernel.cols)){ 
        perror("Convimg size incorrect"); 
        exit(1); 
    }
    for(int i = 0; i < convimg.rows; i++){
        int j = 0;
        // Vectorized loop - process 8 output pixels at once
        for (; j+7 < convimg.cols; j += 8){
            __m256 dotprod = _mm256_setzero_ps();
            for (int ki = 0; ki < Kernel.rows; ki++){
                int ridx = ((i+ki)*image.cols);
                for(int kj = 0; kj < Kernel.cols; kj++){
                    int cidx = (j+kj);
                    __m256 ker_vec = _mm256_set1_ps(Kernel.K[(ki*Kernel.cols)+kj]);
                    __m256 img_vec = _mm256_loadu_ps(&image.Data[ridx+cidx]);
                    dotprod = _mm256_fmadd_ps(ker_vec, img_vec, dotprod);
                }
            }
            _mm256_storeu_ps(&convimg.Data[i*convimg.cols+j], dotprod);
        }
        for (; j < convimg.cols; j++){
            float dotprod = 0.0f;
            for (int ki = 0; ki < Kernel.rows; ki++){
                int ridx = ((i+ki)*image.cols);
                for(int kj = 0; kj < Kernel.cols; kj++){
                    int cidx = (j+kj);
                    dotprod += image.Data[ridx+cidx] * Kernel.K[(ki*Kernel.cols)+kj];
                }
            }
            convimg.Data[i*convimg.cols+j] = dotprod;
        }
    }
}

/// @brief Operates POOLing function on the image and returns it
/// @param type Type of pooling
/// @param image the image to do pooling on
/// @param ker_size size of the kernel that slides over the image
/// @return Pooled image
void MAXPOOL(Image2D poolimg, Image2D image, int ker_size, int stride){
    if(poolimg.rows != (image.rows-ker_size)/stride+1 || poolimg.cols != (image.cols-ker_size)/stride+1) {
        perror("Pool image is incorrect size"); 
        exit(1); 
    }
    memset(poolimg.maxidx,-1,sizeof(int)*poolimg.cols*poolimg.rows);
    //looping over return image
    for(int i = 0; i < poolimg.rows; i++){
        for (int j = 0; j < poolimg.cols; j++){

            float max = -INFINITY;
            int idxmax = -1;
            
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
            Img_Data_Obj(poolimg,i,j) = max;
            poolimg.maxidx[i*poolimg.cols+j] = idxmax;
        }
    }
}

/// @brief Unpools the image from a pooled image and with metadata
/// @param unpooled return part
/// @param pooled pooled image
/// @param UPMD metadata
void MAXUNPOOL(Image2D unpooled, Image2D pooled) {
    // Zero out the unpooled image
    memset(unpooled.Data,0.0f,sizeof(float)*unpooled.rows*unpooled.cols);
    // Unpool with bounds checking
    for(int i = 0; i < pooled.rows*pooled.cols; i++) {
        if(pooled.maxidx[i] < 0 || pooled.maxidx[i] > unpooled.rows*unpooled.cols) continue;
        else{ unpooled.Data[pooled.maxidx[i]] = pooled.Data[i]; }
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
void backprop_kernel(kernel Kernel, Image2D Unpooled, Image2D Image){
    for(int i = 0; i<Unpooled.rows;i++){
        for (int j = 0; j < Unpooled.cols; j++){
            int curidx = i*Unpooled.cols+j;
            if(Unpooled.Data[curidx] == 0.0f) continue;
            for(int x = 0; x < Kernel.rows; x++){
                int ridx = (i+x)*Image.cols;
                for(int y = 0; y < Kernel.cols; y++){
                    int cidx = j+y;
                    Kernel.del_K[x*Kernel.cols+y] += Unpooled.Data[curidx]*Image.Data[ridx+cidx];
                }
            }
        }
    }
}

/// @brief accumulates and averages.
/// @param Kernel 
/// @param batch_size is actually 1/BATCH_SIZE
void kernel_accum(kernel Kernel, float batch_size){
    int i = 0;
    __m256 bs = _mm256_set1_ps(batch_size);
    for(; i+7 < Kernel.cols*Kernel.rows; i+=8){
        __m256 v_ker = _mm256_loadu_ps(&Kernel.del_K[i]);
        v_ker = _mm256_mul_ps(bs,v_ker);
        __m256 v_dker = _mm256_loadu_ps(&Kernel.sum_del_K[i]);
        v_ker = _mm256_add_ps(v_ker, v_dker);
        _mm256_storeu_ps(&Kernel.sum_del_K[i],v_ker);
    }
    for (; i < Kernel.cols*Kernel.rows; i++){
        Kernel.sum_del_K[i] += Kernel.del_K[i]*batch_size;
    }
}

/// @brief Updates the kernel parameters using the computed gradients.
/// @param delta_kernel Gradient of the kernel. 
void kernel_update(kernel Kernel, float learning_rate){
    __m256 lr_vec = _mm256_set1_ps(learning_rate);
    int i = 0;
    for(; i+7 < Kernel.cols*Kernel.rows; i+=8){
        __m256 v_ker = _mm256_loadu_ps(&Kernel.K[i]);
        __m256 v_dker = _mm256_loadu_ps(&Kernel.sum_del_K[i]);
        __m256 mulvec = _mm256_mul_ps(lr_vec,v_dker);
        v_ker = _mm256_sub_ps(v_ker, mulvec);
        _mm256_storeu_ps(&Kernel.K[i],v_ker);
    }
    for (; i < Kernel.cols*Kernel.rows; i++){
        Kernel.K[i] -= learning_rate*Kernel.sum_del_K[i];
    }
    
}

/// @brief Sets all the values in the kernel to zero
/// @param Kernel The kernel to be zeroed out.
void zero_kernel(kernel Kernel){
    memset(Kernel.del_K, 0, sizeof(float) * Kernel.cols * Kernel.rows);
    memset(Kernel.sum_del_K, 0, sizeof(float) * Kernel.cols * Kernel.rows);
}