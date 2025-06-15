#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<math.h>
#include<time.h>
#include"Convolution2D.h"
#include"NN-funcs/NeuralNetwork.h"

#define BATCH_SIZE 32


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

// Add this helper function
void print_pool_indices(int* UPMD, int pooled_rows, int pooled_cols, int unpooled_size) {
    printf("Pool indices validation:\n");
    for(int i = 0; i < pooled_rows; i++) {
        for(int j = 0; j < pooled_cols; j++) {
            int idx = UPMD[i * pooled_cols + j];
            printf("(%d,%d): idx=%d %s\n", 
                   i, j, idx, 
                   (idx >= 0 && idx < unpooled_size) ? "valid" : "INVALID");
        }
    }
}


int main(){
    FILE* file = fopen("mnist/train-images.idx3-ubyte", "rb");
    struct pixel_data* pixel_data = get_image_pixel_data(file);
    fclose(file);
    file = fopen("mnist/train-labels.idx1-ubyte", "rb");
    unsigned char* lbl_arr = get_image_labels(file);
    fclose(file);
    
    // Training parameters
    int epoch = 2;
    float learning_rate = 0.001f;
    int size = pixel_data->size/BATCH_SIZE; 

    // layer sizes
    int lay1 = 1600;
    int lay2 = 100;
    int lay3 = 10;
    
    // Init Activation buffers
    struct activations* AL1 = init_activations(lay1);
    struct activations* AL2 = init_activations(lay2);
    struct activations* AL3 = init_activations(lay3);

    // init activation error buffers
    struct activations* dZAL1 = init_activations(lay1);
    struct activations* dZAL2 = init_activations(lay2);
    struct activations* dZAL3 = init_activations(lay3);

    // derivative buffers
    struct activations* dZAL1_ReLU = init_activations(lay1);
    struct activations* dZAL2_ReLU = init_activations(lay2);

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
    Image2D kernel1 = CreateKernel(8, 8);
    Image2D kernel2 = CreateKernel(8, 8);
    Image2D kernel3 = CreateKernel(8, 8);
    Image2D kernel4 = CreateKernel(8, 8);

    // Create del kernels
    Image2D del_kernel1 = CreateKernel(8, 8);
    Image2D del_kernel2 = CreateKernel(8, 8);
    Image2D del_kernel3 = CreateKernel(8, 8);
    Image2D del_kernel4 = CreateKernel(8, 8);

    // Create sum del kernels
    Image2D sum_del_kernel1 = CreateKernel(8, 8);
    Image2D sum_del_kernel2 = CreateKernel(8, 8);
    Image2D sum_del_kernel3 = CreateKernel(8, 8);
    Image2D sum_del_kernel4 = CreateKernel(8, 8);


    while(epoch--){
        for (int j = 0; j < size; j++){
            float total_loss = 0.0f;
            for (int k = (BATCH_SIZE*j); k < (BATCH_SIZE*(j+1)); k++){
                ImageInput(image,pixel_data->neuron_activation[k]);

                Image2D convimg1 = Conv2D(kernel1,image);
                Image2D convimg2 = Conv2D(kernel2,image);
                Image2D convimg3 = Conv2D(kernel3,image);
                Image2D convimg4 = Conv2D(kernel4,image);

                // Assume AL1->activations is float*, convimgX.Data is float*, all properly sized
                int img_size = AL1->size / 4;

                memcpy(AL1->activations,                    convimg1.Data, img_size * sizeof(float));
                memcpy(AL1->activations + img_size,         convimg2.Data, img_size * sizeof(float));
                memcpy(AL1->activations + 2 * img_size,     convimg3.Data, img_size * sizeof(float));
                memcpy(AL1->activations + 3 * img_size,     convimg4.Data, img_size * sizeof(float));

                // Forward Propagation
                ReLU(AL1); // Relu image
                forward_prop_step(AL1, L1, AL2); // forward prop layer 1
                ReLU(AL2); // Relu hidden 1
                forward_prop_step(AL2, L2, AL3); // forward prop layer 2
                softmax(AL3); // Softmax output layer
                loss_function(dZAL3,AL3,lbl_arr[k]); // Loss function
                back_propogate_step(L2, dL2, dZAL3, AL2); // back prop step 1
                ReLU_derivative(AL2,dZAL2_ReLU); // relu deriv
                calc_grad_activation(dZAL2,L2,dZAL3, dZAL2_ReLU); // gradient calc hidden layer
                back_propogate_step(L1, dL1, dZAL2, AL1); // Back prop step 2
                ReLU_derivative(AL1,dZAL1_ReLU); // relu deriv
                calc_grad_activation(dZAL1,L1,dZAL2, dZAL1_ReLU); // First layer gradient calc
                
                int chunk_size = dZAL1->size / 4;

                memcpy(convimg1.Data, dZAL1->activations,                    chunk_size * sizeof(float));
                memcpy(convimg2.Data, dZAL1->activations + chunk_size,       chunk_size * sizeof(float));
                memcpy(convimg3.Data, dZAL1->activations + 2 * chunk_size,   chunk_size * sizeof(float));
                memcpy(convimg4.Data, dZAL1->activations + 3 * chunk_size,   chunk_size * sizeof(float));


                backprop_kernel(del_kernel1,kernel1,convimg1,image);
                backprop_kernel(del_kernel2,kernel2,convimg2,image);
                backprop_kernel(del_kernel3,kernel3,convimg3,image);
                backprop_kernel(del_kernel4,kernel4,convimg4,image);

                param_update(sdL1,dL1,1);
                param_update(sdL2,dL2,1);

                kernel_update(del_kernel1,sum_del_kernel1,1);
                kernel_update(del_kernel2,sum_del_kernel2,1);
                kernel_update(del_kernel3,sum_del_kernel3,1);
                kernel_update(del_kernel4,sum_del_kernel4,1);
                
                total_loss += compute_loss(AL3,lbl_arr[k])/BATCH_SIZE;
            }
            printf("\n\nBatch Loss:%f\nEpoch no:%i",total_loss,epoch);
            param_update(L1,sdL1,-learning_rate);
            param_update(L2,sdL2,-learning_rate);
            Zero_Layer(sdL1,0);
            Zero_Layer(sdL2,0);
            kernel_update(sum_del_kernel1,kernel1,-learning_rate*0.01f);
            kernel_update(sum_del_kernel2,kernel2,-learning_rate*0.01f);
            kernel_update(sum_del_kernel3,kernel3,-learning_rate*0.01f);
            kernel_update(sum_del_kernel4,kernel4,-learning_rate*0.01f);
            zero_kernel(sum_del_kernel1);
            zero_kernel(sum_del_kernel2);
            zero_kernel(sum_del_kernel3);
            zero_kernel(sum_del_kernel4);
        }
    }
    
    image_data_finalizer(pixel_data);
    image_label_finalizer(lbl_arr);

    FILE* test_file = fopen("mnist/t10k-labels.idx1-ubyte", "r");
    unsigned char* test_lbl_arr = get_image_labels(test_file);
    test_file = fopen("mnist/t10k-images.idx3-ubyte", "rb");
    struct pixel_data* test_pix_data = get_image_pixel_data(test_file);

    printf("\n\nCalculating accuracy:-\n\n");
    int correct_pred = 0;
    for (unsigned int k = 0; k < test_pix_data->size; k++){
        ImageInput(image,test_pix_data->neuron_activation[k]);

                Image2D convimg1 = Conv2D(kernel1,image);
                Image2D convimg2 = Conv2D(kernel2,image);
                Image2D convimg3 = Conv2D(kernel3,image);
                Image2D convimg4 = Conv2D(kernel4,image);

                // Assume AL1->activations is float*, convimgX.Data is float*, all properly sized
                int img_size = AL1->size / 4;

                memcpy(AL1->activations,                    convimg1.Data, img_size * sizeof(float));
                memcpy(AL1->activations + img_size,         convimg2.Data, img_size * sizeof(float));
                memcpy(AL1->activations + 2 * img_size,     convimg3.Data, img_size * sizeof(float));
                memcpy(AL1->activations + 3 * img_size,     convimg4.Data, img_size * sizeof(float));

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
