#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<math.h>
#include<time.h>
#include"Conv/Convolution2D.h"
#include"NN-funcs/NeuralNetwork.h"

#define BATCH_SIZE 64
#define NUM_KERNELS 32

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

    // preprocess the data.
    FILE* file = fopen("mnist/train-images-idx3-ubyte/train-images-idx3-ubyte", "rb");
    struct pixel_data* pixel_data = get_image_pixel_data(file);
    fclose(file);
    file = fopen("mnist/train-labels-idx1-ubyte/train-labels-idx1-ubyte", "rb");
    unsigned char* lbl_arr = get_image_labels(file);
    fclose(file);
    
    // Training parameters
    float batch_time = 0.0f;
    int epoch = 5;
    float learning_rate = 0.0005f;
    int size = pixel_data->size/BATCH_SIZE; 
    
    // layer sizes
    int lay1 = 121*NUM_KERNELS;
    int lay2 = 128;
    int lay3 = 64;    
    int lay4 = 10;

    activations*A1 = init_activations(lay1);
    activations*A2 = init_activations(lay2);
    activations*A3 = init_activations(lay3);  
    activations*A4 = init_activations(lay4);   
    
    // Layer init
    DenseLayer* L1 = init_DenseLayer(lay2,lay1,1);
    DenseLayer* L2 = init_DenseLayer(lay3,lay2,1);
    DenseLayer* L3 = init_DenseLayer(lay4,lay3,1);
    
    // Create Image
    Image2D image = CreateImage(pixel_data->rows, pixel_data->cols);
    
    // Create Kernels
    Image2D kernels[NUM_KERNELS];
    Image2D del_kernel[NUM_KERNELS];
    Image2D sum_del_kernel[NUM_KERNELS];
    for (int i = 0; i < NUM_KERNELS; i++){
        kernels[i] = CreateKernel(5, 5);
        del_kernel[i] = CreateKernel(5, 5);
        sum_del_kernel[i] = CreateKernel(5, 5);
        // Ensure gradient buffers start at zero
        zero_kernel(del_kernel[i]);
        zero_kernel(sum_del_kernel[i]);
    }
    
    Image2D convimg[NUM_KERNELS];
    Image2D Poolimg[NUM_KERNELS];
    
    for (int i = 0; i < NUM_KERNELS; i++){
        convimg[i] = CreateConvImg(image,kernels[i]);
        Poolimg[i] = CreatePoolImg(convimg[i],2,2);
    }
    

    while (epoch--) {
        for (int j = 0; j < size; j++) {
            float total_loss = 0.0f;
            float start = clock();
    
            for (int k = BATCH_SIZE * j; k < BATCH_SIZE * (j + 1); k++) {
                // Load image
                ImageInput(image, pixel_data->neuron_activation[k]);
                // Conv + Pool
                for (int i = 0; i < NUM_KERNELS; i++) {
                    Conv2D(kernels[i], image, convimg[i]);
                    MAXPOOL(Poolimg[i], convimg[i], 2, 2);

                    int imgsize = Poolimg[i].rows * Poolimg[i].cols;
                    memcpy(A1->Z + i * imgsize, Poolimg[i].Data, imgsize * sizeof(float));
                }
                // print_ascii_art(image); // Should show digit
                // print_ascii_art(Poolimg[0]); // Should still resemble digit-ish blobs

                // Forward
                activation_function(A1, ReLU);
                forward_prop_step(A1, L1, A2);
                activation_function(A2, ReLU);
                forward_prop_step(A2, L2, A3);
                activation_function(A3, ReLU);
                forward_prop_step(A3, L3, A4);
                activation_function(A4, Softmax);
    
                // Loss
                total_loss += loss_function(A4, CE, lbl_arr[k]) / BATCH_SIZE;
    
                // Backward
                back_propogate_step(A3, L3, A4);
                calc_grad_activation(A3, L3, A4);
    
                back_propogate_step(A2, L2, A3);
                calc_grad_activation(A2, L2, A3);
    
                back_propogate_step(A1, L1, A2);
                calc_grad_activation(A1, L1, A2);
    
                // Backprop kernels
                for (int i = 0; i < NUM_KERNELS; i++) {
                    int imgsize = Poolimg[i].rows * Poolimg[i].cols;
                    memcpy(Poolimg[i].Data, A1->dZ + i * imgsize, imgsize * sizeof(float));
                    MAXUNPOOL(convimg[i], Poolimg[i]);
                    backprop_kernel(del_kernel[i], kernels[i], convimg[i], image);
                    // Accumulate gradients: sum_del_kernel += del_kernel
                    kernel_update(del_kernel[i], sum_del_kernel[i], -1.0f);
                    // Reset per-sample gradient buffer
                    zero_kernel(del_kernel[i]);
                }
                grad_accum(L1,1);
                grad_accum(L2,1);
                grad_accum(L3,1);
            }
    
            float end = clock();
    
            // Apply gradient descent update
            update_weights(L1, learning_rate);
            update_weights(L2, learning_rate);
            update_weights(L3, learning_rate);
            zero_grad(L1);
            zero_grad(L2);
            zero_grad(L3);
    
            for (int i = 0; i < NUM_KERNELS; i++) {
                kernel_update(sum_del_kernel[i], kernels[i], learning_rate);
                zero_kernel(sum_del_kernel[i]);
            }
    
            float bt = ((end - start) / CLOCKS_PER_SEC) * 1000;
            printf("\n\nBatch Loss: %f\nEpoch: %d", total_loss, epoch);
            printf("\nBatch process time: %f ms\n", bt);
    
            batch_time += bt;
        }
    
        batch_time /= size;
        printf("Average Batch time: %f ms\nBatch size: 32\n", batch_time);
        
    }

    
    image_data_finalizer(pixel_data);
    image_label_finalizer(lbl_arr);
    
    FILE* test_file = fopen("mnist/t10k-images.idx3-ubyte", "rb");
    struct pixel_data* test_pix_data = get_image_pixel_data(test_file);
    test_file = fopen("mnist/t10k-labels.idx1-ubyte", "rb");
    unsigned char* test_lbl_arr = get_image_labels(test_file);
    
    printf("\n\nCalculating accuracy:-\n\n");
    int correct_pred = 0;
    for (unsigned int k = 0; k < test_pix_data->size; k++){
        ImageInput(image, test_pix_data->neuron_activation[k]);

            // Conv + Pool
            for (int i = 0; i < NUM_KERNELS; i++) {
                Conv2D(kernels[i], image, convimg[i]);
                MAXPOOL(Poolimg[i], convimg[i], 2, 2);
                int imgsize = Poolimg[i].rows * Poolimg[i].cols;
                memcpy(A1->Z + i * imgsize, Poolimg[i].Data, imgsize * sizeof(float));
            }

            // Forward
            activation_function(A1, ReLU);
            forward_prop_step(A1, L1, A2);
            activation_function(A2, ReLU);
            forward_prop_step(A2, L2, A3);
            activation_function(A3, ReLU);
            forward_prop_step(A3, L3, A4);
            activation_function(A4, Softmax);

        if(test_lbl_arr[k] == get_pred_from_softmax(A4)){correct_pred++;}
        if (k%1000 == 0){printf(".");}
    }
    printf("\n\n Total Correct predictions: %d\n",correct_pred);
    printf("\n\n The Accuracy of the model is: %d/%d\n",correct_pred,test_pix_data->size);
    
    return 1;
}
 