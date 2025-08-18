#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<math.h>
#include<time.h>
#include<string.h>
#include"Conv/Convolution2D.h"
#include"NN-funcs/NeuralNetwork.h"

#define BATCH_SIZE 64
#define NUM_KERNELS 2
#define KERNEL_SIZE 5
#define NUM_KERNELS_ 4
#define STRIDE 1

void print_ascii_art(Image2D img) {
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            float val = img.Data[i * img.cols + j];
            putchar(val > 0.0000f ? '#' : '.');
            putchar(' ');
        }
        putchar('\n');
    }
}

int main(){
    // int accuracy[500];
    // int i = 0;
    // for (float learning_rate = 0.00001; learning_rate < 0.005f; learning_rate+=0.00001){
    // printf("\n\n---------------------------------------\n");
    // printf("          Iteration no %i                  ", i);
    // printf("\n---------------------------------------\n\n");
    // preprocess the data. 172029 203264
    FILE* file = fopen("fashion-mnist/train-images-idx3-ubyte", "rb");
    if (!file) {
        printf("Error: Could not open training images file\n");
        return -1;
    }
    struct pixel_data* pixel_data = get_image_pixel_data(file);
    fclose(file);
    
    file = fopen("fashion-mnist/train-labels-idx1-ubyte", "rb");
    if (!file) {
        printf("Error: Could not open training labels file\n");
        return -1;
    }
    unsigned char* lbl_arr = get_image_labels(file);
    fclose(file);
    
    if (!pixel_data || !lbl_arr) {
        printf("Error: Failed to load training data\n");
        return -1;
    }
    
    //10 epochs 0.0005 lr 88.97%
    // Training parameters
    int epoch = 1;
    float learning_rate = 0.000001; // Increased from 0.00001 for better initial learning
    int size = pixel_data->size/BATCH_SIZE; 
    
    // Create Image
    Image2D image = CreateImage(pixel_data->rows, pixel_data->cols);
    
    // Create Kernels
    Image2D kernels[NUM_KERNELS];
    Image2D del_kernel[NUM_KERNELS];
    Image2D sum_del_kernel[NUM_KERNELS];
    for (int i = 0; i < NUM_KERNELS; i++){
        kernels[i] = CreateKernel(KERNEL_SIZE, KERNEL_SIZE);
        del_kernel[i] = CreateKernel(KERNEL_SIZE, KERNEL_SIZE);
        sum_del_kernel[i] = CreateKernel(KERNEL_SIZE, KERNEL_SIZE);
        // Ensure gradient buffers start at zero
        zero_kernel(del_kernel[i]);
        zero_kernel(sum_del_kernel[i]);
    }
    
    // second layer of kernels
    Image2D kernels_[NUM_KERNELS_];
    Image2D del_kernel_[NUM_KERNELS_];
    Image2D sum_del_kernel_[NUM_KERNELS_];
    for (int i = 0; i < NUM_KERNELS_; i++){
        kernels_[i] = CreateKernel(KERNEL_SIZE, KERNEL_SIZE);
        del_kernel_[i] = CreateKernel(KERNEL_SIZE, KERNEL_SIZE);
        sum_del_kernel_[i] = CreateKernel(KERNEL_SIZE, KERNEL_SIZE);
        // Ensure gradient buffers start at zero
        zero_kernel(del_kernel_[i]);
        zero_kernel(sum_del_kernel_[i]);
    }
    
    Image2D convimg[NUM_KERNELS];
    Image2D Poolimg[NUM_KERNELS];
    Image2D convimg_[NUM_KERNELS_];
    Image2D Poolimg_[NUM_KERNELS_];
    
    // Create first layer images
    for (int i = 0; i < NUM_KERNELS; i++){
        convimg[i] = CreateConvImg(image, kernels[i]);
        printf("Conv1[%d] rows: %i, cols: %i\n", i, convimg[i].rows, convimg[i].cols);        
        Poolimg[i] = CreatePoolImg(convimg[i], 2, STRIDE);
        printf("Pool1[%d] rows: %i, cols: %i\n", i, Poolimg[i].rows, Poolimg[i].cols);
    }
    
    // Create second layer images - you need to decide how to connect layers
    for (int i = 0; i < NUM_KERNELS_; i++){
        // Option 1: Each second layer kernel connects to one first layer output (cycling through)
        int input_idx = i % NUM_KERNELS;  // Maps 0->0, 1->1, 2->2, 3->3, 4->0, 5->1, 6->2, 7->3
        convimg_[i] = CreateConvImg(Poolimg[input_idx], kernels_[i]);
        printf("Conv2[%d] rows: %i, cols: %i\n", i, convimg_[i].rows, convimg_[i].cols);  
        Poolimg_[i] = CreatePoolImg(convimg_[i], 2, STRIDE);
        printf("Pool2[%d] rows: %i, cols: %i\n", i, Poolimg_[i].rows, Poolimg_[i].cols);
    }
    
    // Calculate the actual size needed for A1 based on the created pooled images
    int lay1 = 0;
    for (int i = 0; i < NUM_KERNELS_; i++) {
        lay1 += Poolimg_[i].rows * Poolimg_[i].cols;
    }
    int lay2 = 256;
    int lay3 = 64;    
    int lay4 = 10;
    
    printf("Layer sizes: lay1=%d (will be calculated), lay2=%d, lay3=%d, lay4=%d\n", lay1, lay2, lay3, lay4);

    activations* A1 = init_activations(lay1);
    activations* A2 = init_activations(lay2);
    activations* A3 = init_activations(lay3);
    activations* A4 = init_activations(lay4);
    
    if (!A1 || !A2 || !A3 || !A4) {
        printf("Error: Failed to initialize activations\n");
        return -1;
    }
    
    DenseLayer* L1 = init_DenseLayer(lay2, lay1, 5);
    DenseLayer* L2 = init_DenseLayer(lay3, lay2, 5);
    DenseLayer* L3 = init_DenseLayer(lay4, lay3, 5);

    if (!L1 || !L2 || !L3) {
        printf("Error: Failed to initialize dense layers\n");
        return -1;
    }

    while (epoch--) {
        float start = clock();
        for (int j = 0; j < size; j++) {
            float total_loss = 0.0f;
    
            for (int k = BATCH_SIZE * j; k < BATCH_SIZE * (j + 1) && k < pixel_data->size; k++) {
                // Load image
                ImageInput(image, pixel_data->neuron_activation[k]);
                
                // Conv + Pool
                for (int i = 0; i < NUM_KERNELS; i++) {
                    Conv2D(kernels[i], image, convimg[i]);
                    MAXPOOL(Poolimg[i], convimg[i], 2, STRIDE);
                }
                
                for(int i = 0; i < NUM_KERNELS_; i++){
                    int input_idx = i % NUM_KERNELS;
                    Conv2D(kernels_[i], Poolimg[input_idx], convimg_[i]);
                    MAXPOOL(Poolimg_[i], convimg_[i], 2, STRIDE);
                    int imgsize = Poolimg_[i].rows * Poolimg_[i].cols;
                    memcpy(A1->Z + i * imgsize, Poolimg_[i].Data, imgsize * sizeof(float));
                }
                
                // Forward
                activation_function(A1, ReLU);
                forward_prop_step(A1, L1, A2);
                activation_function(A2, ReLU);
                forward_prop_step(A2, L2, A3);
                activation_function(A3, ReLU);
                forward_prop_step(A3, L3, A4);
                activation_function(A4, Softmax);
                
                // Loss
                float sample_loss = loss_function(A4, CE, lbl_arr[k]);
                total_loss += sample_loss / BATCH_SIZE;
    
                // Backward
                back_propogate_step(A3, L3, A4);
                calc_grad_activation(A3, L3, A4);
    
                back_propogate_step(A2, L2, A3);
                calc_grad_activation(A2, L2, A3);
    
                back_propogate_step(A1, L1, A2);
                calc_grad_activation(A1, L1, A2);
    
                // Backprop kernels
                // Backpropagation for second layer (layer 2 -> layer 1)
                for (int i = 0; i < NUM_KERNELS_; i++) {
                    int imgsize = Poolimg_[i].rows * Poolimg_[i].cols;
                    memcpy(Poolimg_[i].Data, A1->dZ + i * imgsize, imgsize * sizeof(float));
                    MAXUNPOOL(convimg_[i], Poolimg_[i]);
                    int input_idx = i % NUM_KERNELS;
                    backprop_kernel(del_kernel_[i], kernels_[i], convimg_[i], Poolimg[input_idx]);
                    kernel_update(del_kernel_[i], sum_del_kernel_[i], 1.0f);
                    zero_kernel(del_kernel_[i]);
                }
                
                for (int i = 0; i < NUM_KERNELS; i++) {
                    MAXUNPOOL(convimg[i], Poolimg[i]); // This needs the gradient from second layer
                    backprop_kernel(del_kernel[i], kernels[i], convimg[i], image);
                    kernel_update(del_kernel[i], sum_del_kernel[i], 1.0f);
                    zero_kernel(del_kernel[i]);
                }
                
                grad_accum(L1, BATCH_SIZE);
                grad_accum(L2, BATCH_SIZE);
                grad_accum(L3, BATCH_SIZE);
            }
            
            // Debug: Print batch loss
            // printf("Batch %d total loss: %f\n", j, total_loss);
            
            // Apply gradient descent update
            update_weights(L1, learning_rate);
            update_weights(L2, learning_rate);
            update_weights(L3, learning_rate);
            zero_grad(L1);
            zero_grad(L2);
            zero_grad(L3);
            
            for (int i = 0; i < NUM_KERNELS; i++) {
                kernel_update(sum_del_kernel[i], kernels[i], learning_rate*0.01);
                zero_kernel(sum_del_kernel[i]);
            }
            for (int i = 0; i < NUM_KERNELS_; i++) {
                kernel_update(sum_del_kernel_[i], kernels_[i], learning_rate*0.01);
                zero_kernel(sum_del_kernel_[i]);
            }
            
            // Debug: Print learning rate and some kernel values
        }
        float end = clock();
        printf("Epoch %i time: %f s\n", 5-epoch, ((end - start) / CLOCKS_PER_SEC));
        
    }

    
    image_data_finalizer(pixel_data);
    image_label_finalizer(lbl_arr);
    
    FILE* test_file = fopen("fashion-mnist/t10k-images-idx3-ubyte", "rb");
    if (!test_file) {
        printf("Error: Could not open test images file\n");
        return -1;
    }
    struct pixel_data* test_pix_data = get_image_pixel_data(test_file);
    fclose(test_file);
    
    test_file = fopen("fashion-mnist/t10k-labels-idx1-ubyte", "rb");
    if (!test_file) {
        printf("Error: Could not open test labels file\n");
        return -1;
    }
    unsigned char* test_lbl_arr = get_image_labels(test_file);
    fclose(test_file);
    
    if (!test_pix_data || !test_lbl_arr) {
        printf("Error: Failed to load test data\n");
        return -1;
    }
    
    float start = clock();
    printf("\n\nCalculating accuracy:-\n\n");
    int correct_pred = 0;
    for (unsigned int k = 0; k < test_pix_data->size; k++){
        ImageInput(image, test_pix_data->neuron_activation[k]);

        // Conv + Pool
        for (int i = 0; i < NUM_KERNELS; i++) {
            Conv2D(kernels[i], image, convimg[i]);
            MAXPOOL(Poolimg[i], convimg[i], 2, STRIDE);
        }
        
        for(int i = 0; i < NUM_KERNELS_; i++){
            int input_idx = i % NUM_KERNELS;
            Conv2D(kernels_[i], Poolimg[input_idx], convimg_[i]);
            MAXPOOL(Poolimg_[i], convimg_[i], 2, STRIDE);
            int imgsize = Poolimg_[i].rows * Poolimg_[i].cols;
            memcpy(A1->Z + i * imgsize, Poolimg_[i].Data, imgsize * sizeof(float));
        }
    
        // Forward
        inference_activation_function(A1, ReLU);
        forward_prop_step(A1, L1, A2);
        inference_activation_function(A2, ReLU);
        forward_prop_step(A2, L2, A3);
        inference_activation_function(A3, ReLU);
        forward_prop_step(A3, L3, A4);
        inference_activation_function(A4, Softmax);

        if(test_lbl_arr[k] == get_pred_from_softmax(A4)){correct_pred++;}
        if (k%1000 == 0){printf(".");}
    }
    float end = clock();
    printf("\n\nTesting 10000 inferences, time: %f ms\n",((end - start) / CLOCKS_PER_SEC) * 1000);
    printf("\n\n Total Correct predictions: %d\n",correct_pred);
    printf("\n\n The Accuracy of the model is: %d/%d\n\n",correct_pred,test_pix_data->size);
    
    // Cleanup
    image_data_finalizer(test_pix_data);
    image_label_finalizer(test_lbl_arr);
    
    // Cleanup activations and layers
    Free_activations(A1);
    Free_activations(A2);
    Free_activations(A3);
    Free_activations(A4);
    Free_DenseLayer(L1);
    Free_DenseLayer(L2);
    Free_DenseLayer(L3);
    
    // Cleanup images
    free(image.Data);
    for (int i = 0; i < NUM_KERNELS; i++) {
        free(convimg[i].Data);
        free(Poolimg[i].Data);
        free(kernels[i].Data);
        free(del_kernel[i].Data);
        free(sum_del_kernel[i].Data);
    }
    for (int i = 0; i < NUM_KERNELS_; i++) {
        free(convimg_[i].Data);
        free(Poolimg_[i].Data);
        free(kernels_[i].Data);
        free(del_kernel_[i].Data);
        free(sum_del_kernel_[i].Data);
    }
    
    return 0;
}
 