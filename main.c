#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<math.h>
#include<time.h>
#include"Convolution2D.h"
#include"NN-funcs/NeuralNetwork.h"

#define BATCH_SIZE 16


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
    FILE* file = fopen("data/train-images.idx3-ubyte", "rb");
    struct pixel_data* pixel_data = get_image_pixel_data(file);
    fclose(file);
    file = fopen("data/train-labels.idx1-ubyte", "rb");
    unsigned char* lbl_arr = get_image_labels(file);
    fclose(file);
    
    // FCN details
    int inpl = 200;
    int lay2 = 64;
    int oupl = 10;

    // Defining Neural Network
    // Activation buffers
    struct activations*AL1 = init_activations(inpl);// input
    struct activations*AL2 = init_activations(lay2);// hidden
    struct activations*AL3 = init_activations(oupl);// output

    // Weights
    struct layer*L1 = init_layer(lay2,inpl);
    struct layer*L2 = init_layer(oupl,lay2);

    // Loss activation buffers
    struct activations* dZAL3 = init_activations(oupl);// output layer loss
    struct activations* dZAL2 = init_activations(lay2);// hidden layer loss
    struct activations* dZAL1 = init_activations(inpl);// input layer loss

    // error weight buffers
    struct layer*dL1 = init_layer(lay2,inpl);
    struct layer*dL2 = init_layer(oupl,lay2);

    // // Sum of errors for SGD
    struct layer*sdL1 = init_layer(lay2,inpl);
    struct layer*sdL2 = init_layer(oupl,lay2);

    Image2D kernel1 = CreateKernel(8,8);
    for (int i = 0; i < kernel1.rows*kernel1.cols; i++){
        kernel1.Data[i] = ((float)rand()/((float)RAND_MAX) - 0.5);
    }

    Image2D kernel2 = CreateKernel(8,8);
    for (int i = 0; i < kernel2.rows*kernel2.cols; i++){
        kernel2.Data[i] = ((float)rand()/((float)RAND_MAX) - 0.5);
    }

    Image2D del_kernel1 = CreateKernel(8,8);
    Image2D del_kernel2 = CreateKernel(8,8);


    // unpooling metadata
    int*UPMD1;
    int*UPMD2;

    float LR = 0.0001;
    int epoch = 5;
    int size = pixel_data->size/BATCH_SIZE;
// Looping thru the entire dataset.
while(epoch--){
    for(int i = 0; i < size; i++){
        float total_loss = 0.0;
        for (int k = (BATCH_SIZE*i); k < (BATCH_SIZE*(i+1)); k++){
            // Input image data
            Image2D image = CreateImage(pixel_data->rows,pixel_data->cols,pixel_data->neuron_activation[k]);
    
            // convolution with kernel1
            Image2D convimg1 = Conv2D(kernel1,image);
            // mallocing the metadata array
            UPMD1 = (int*)malloc(sizeof(int)*convimg1.cols*convimg1.rows);
            // maxpooling
            Image2D retimg1 = POOL(1,convimg1,2,2,UPMD1);
            ImageReLU(retimg1);
            // same as above
            Image2D convimg2 = Conv2D(kernel2,image);

            UPMD2 = (int*)malloc(sizeof(int)*convimg2.cols*convimg2.rows);
            Image2D retimg2 = POOL(1,convimg2,2,2,UPMD2);
            ImageReLU(retimg2);

            // Flatten function
            for (int i = 0; i < AL1->size/2; i++){
                AL1->activations[i] = retimg1.Data[i];
            }
            for (int i = AL1->size/2; i < AL1->size; i++){
                AL1->activations[i] = retimg2.Data[i-AL1->size/2];
            }
    
            // Fully Connected network
            forward_prop_step(AL1,L1,AL2);
            ReLU(AL2);
            forward_prop_step(AL2,L2,AL3);
            softmax(AL3);
            loss_function(dZAL3,AL3,lbl_arr[k]); // AL3 now has loss 
            for (int z = 0; z < AL3->size; z++) {
                if (isnan(AL3->activations[z])) {
                    printf("NaN detected in AL3 at index %d\n", z);
                    exit(1);
                }
            }

            
            back_propogate_step(L2,dL2,dZAL3,AL2); // (layer, deriv layer, loss, prev activations)
            ReLU_derivative(AL2,dZAL2); 
            calc_grad_activation(dZAL2,L2,dZAL3,AL2);
            back_propogate_step(L1,dL1,dZAL2,AL1);
            ReLU_derivative(AL1,dZAL1); // takes ReLU deriv and stored it in dZAL1 itself
            calc_grad_activation(dZAL1,L1,dZAL2,AL1);
            
            param_update(sdL1,dL1,1);
            param_update(sdL2,dL2,1);
            
            UNPOOL(convimg1,retimg1,UPMD1);
            UNPOOL(convimg2,retimg2,UPMD2);
            backprop_kernel(del_kernel1,kernel1,convimg1,image);
            backprop_kernel(del_kernel2,kernel2,convimg2,image);
            free(UPMD1);
            free(UPMD2);
            total_loss += compute_loss(AL3,lbl_arr[k])/BATCH_SIZE;
        }
        printf("\n\nBatch Loss:%f\n",total_loss);
        kernel_update(del_kernel1,kernel1,LR/BATCH_SIZE);
        kernel_update(del_kernel2,kernel2,LR/BATCH_SIZE);
        param_update(L1,sdL1,-LR/BATCH_SIZE);
        param_update(L2,sdL2,-LR/BATCH_SIZE);
        zero_kernel(del_kernel1);
        zero_kernel(del_kernel2);
        Zero_Layer(sdL1,0);
        Zero_Layer(sdL2,0);
        // print_activations(AL3);
    }
}
    
    return 1;
}
