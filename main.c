#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<math.h>
#include<time.h>
#include"Conv/Convolution2D.h"
#include"NN-funcs/NeuralNetwork.h"
#include<omp.h>

#define BATCH_SIZE 32
#define NUM_KERNELS 7

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

    FILE* file = fopen("fashion-mnist/train-images-idx3-ubyte", "rb");
    struct pixel_data* pixel_data = get_image_pixel_data(file);
    fclose(file);
    file = fopen("fashion-mnist/train-labels-idx1-ubyte", "rb");
    unsigned char* lbl_arr = get_image_labels(file);
    fclose(file);
    
    // Training parameters
    float batch_time = 0.0f;
    int epoch = 1;
    float learning_rate = 0.0001f;
    int size = pixel_data->size/BATCH_SIZE; 
    
    // layer sizes
    int lay1 = 121*NUM_KERNELS;
    int lay2 = 100;
    int lay3 = 10;
    
    

    // Init Activation buffers
    struct activation* AL1 = init_activation(lay1);
    struct activation* AL2 = init_activation(lay2);
    struct activation* AL3 = init_activation(lay3);

    // init activation error buffers
    struct activation* dZAL1 = init_activation(lay1);
    struct activation* dZAL2 = init_activation(lay2);
    struct activation* dZAL3 = init_activation(lay3);

    // derivative buffers
    struct activation* dZAL1_ReLU = init_activation(lay1);
    struct activation* dZAL2_ReLU = init_activation(lay2);
    
    
    // Layer init
    struct layer* L1 = init_layer(lay2,lay1);
    struct layer* L2 = init_layer(lay3,lay2);

    // Layer error buffers
    struct layer* dL1 = init_layer(lay2,lay1);
    struct layer* dL2 = init_layer(lay3,lay2);
    
    // Layer error sum buffers
    struct layer* sdL1 = init_layer(lay2,lay1);
    struct layer* sdL2 = init_layer(lay3,lay2);
    
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
    }
    
    Image2D convimg[NUM_KERNELS];
    Image2D Poolimg[NUM_KERNELS];
    
    while(epoch--){
        // #pragma omp parallel
        for (int j = 0; j < size; j++){
            float total_loss = 0.0f;
            float start = clock();
            for (int k = (BATCH_SIZE*j); k < (BATCH_SIZE*(j+1)); k++){
                ImageInput(image,pixel_data->neuron_activation[k]);
                for (int i = 0; i < NUM_KERNELS; i++){
                    convimg[i] = Conv2D(kernels[i],image);
                    Poolimg[i] = MAXPOOL(convimg[i],2,2);
                    int imgsize = Poolimg[i].rows*Poolimg[i].cols;
                    memcpy(AL1->activations+i*imgsize, Poolimg[i].Data,imgsize*sizeof(float));
                }
                
                // Forward Propagation
                ReLU(AL1); 
                forward_prop_step(AL1, L1, AL2); 
                ReLU(AL2); 
                forward_prop_step(AL2, L2, AL3); 
                softmax(AL3); 
                loss_function(dZAL3,AL3,lbl_arr[k]); 
                back_propogate_step(L2, dL2, dZAL3, AL2); 
                ReLU_derivative(AL2,dZAL2_ReLU); 
                calc_grad_activation(dZAL2,L2,dZAL3, dZAL2_ReLU); 
                back_propogate_step(L1, dL1, dZAL2, AL1); 
                ReLU_derivative(AL1,dZAL1_ReLU); 
                calc_grad_activation(dZAL1,L1,dZAL2, dZAL1_ReLU); 
                
                for (int i = 0; i < NUM_KERNELS; i++){
                    int imgsize = Poolimg[i].rows*Poolimg[i].cols;
                    memcpy(Poolimg[i].Data,dZAL1->activations+i*imgsize,imgsize*sizeof(float));
                    MAXUNPOOL(convimg[i],Poolimg[i]);
                    backprop_kernel(del_kernel[i],kernels[i],convimg[i],image);
                    kernel_update(del_kernel[i],sum_del_kernel[i],1);
                }
                param_update(sdL1,dL1,1);
                param_update(sdL2,dL2,1);
                
                total_loss += compute_loss(AL3,lbl_arr[k])/BATCH_SIZE;
            }
            float end = clock();
            printf("\n\nBatch Loss:%f\nEpoch no:%i",total_loss,epoch);
            param_update(L1,sdL1,-learning_rate);
            param_update(L2,sdL2,-learning_rate);
            Zero_Layer(sdL1);
            Zero_Layer(sdL2);
            for (int i = 0; i < NUM_KERNELS; i++){
                kernel_update(sum_del_kernel[i],kernels[i],learning_rate);
                zero_kernel(sum_del_kernel[i]);
            }
            float bt = ((end-start)/CLOCKS_PER_SEC)*1000;
            printf("\nBatch process time: %f ms\n",bt);
            batch_time += bt;
        }
        batch_time /= size;
        printf("Average Batch time : %f\n Batch size 32\n", batch_time);
    }
    
    image_data_finalizer(pixel_data);
    image_label_finalizer(lbl_arr);
    
    FILE* test_file = fopen("fashion-mnist/t10k-images-idx3-ubyte", "rb");
    struct pixel_data* test_pix_data = get_image_pixel_data(test_file);
    test_file = fopen("fashion-mnist/t10k-labels-idx1-ubyte", "rb");
    unsigned char* test_lbl_arr = get_image_labels(test_file);
    
    printf("\n\nCalculating accuracy:-\n\n");
    int correct_pred = 0;
    for (unsigned int k = 0; k < test_pix_data->size; k++){
        ImageInput(image,test_pix_data->neuron_activation[k]);
        
        for (int i = 0; i < NUM_KERNELS; i++){
            convimg[i] = Conv2D(kernels[i],image);
            Poolimg[i] = MAXPOOL(convimg[i],2,2);
            int imgsize = Poolimg[i].rows*Poolimg[i].cols;
            memcpy(AL1->activations+i*imgsize, Poolimg[i].Data,imgsize*sizeof(float));
        }

        // Forward Propagation
        ReLU(AL1); // Relu image
        forward_prop_step(AL1, L1, AL2); // forward prop layer 1
        ReLU(AL2); // Relu hidden 1
        forward_prop_step(AL2, L2, AL3); // forward prop layer 2
        softmax(AL3); // Softmax output layer
        if(test_lbl_arr[k] == get_pred_from_softmax(AL3)){correct_pred++;}
        if (k%100 == 0){printf(".");}
    }
    printf("\n\n Total Correct predictions: %d\n",correct_pred);
    printf("\n\n The Accuracy of the model is: %d/%d\n",correct_pred,test_pix_data->size);
    
    return 1;
}
