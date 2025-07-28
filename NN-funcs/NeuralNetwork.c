#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<math.h>
#include"NeuralNetwork.h"

#include <immintrin.h> // Header for AVX intrinsics

#define L_WEIGHT(L, i, j)  (L->Weights[((i) * (L)->cols) + (j)])


/// @brief Initializes a layer struct with guards in place to prevent memory leak in case malloc fails.
/// @param rows number of rows in both bias and weight matrix.
/// @param cols number of columns in weight matrix.
struct layer*init_layer(int rows, int cols){
    int r = rows; int c = cols;
    struct layer* Layer = (struct layer*)malloc(sizeof(layer));
    if (Layer == NULL) {
        perror("Failed to allocate memory for Layer");
        return NULL;
    }
    Layer->cols = cols; Layer->rows = rows;
    Layer->biases = malloc(r*sizeof(float));
    if (Layer->biases == NULL) {
        perror("Failed to allocate memory for Layer->biases");
        free(Layer);
        return NULL;
    }
    Layer->Weights = (float*)malloc(rows*cols*sizeof(float));
    if (Layer->Weights == NULL) {
        perror("Failed to allocate memory for Layer->Weights");
        free(Layer->biases);
        free(Layer);
        return NULL;
    }
    for (int i = 0; i < r; i++){ Layer->biases[i] = (float)rand()/((float)RAND_MAX) - 0.5; }
    for(int i = 0; i < r*c; i++){
            float scale = sqrt(2.0 / (float)Layer->cols);
            Layer->Weights[i] = ((float)rand() / (float)RAND_MAX) * 2 * scale - scale;
    }
    return Layer;
}

/// @brief Frees the Layer struct.
/// @param Layer 
void free_layer(struct layer*Layer){
    if (Layer == NULL) {return;}
    if (Layer->biases != NULL) {
        free(Layer->biases);
        Layer->biases = NULL; 
    }
    if (Layer->Weights != NULL) {
        free(Layer->Weights);
        Layer->Weights = NULL;
    }
    free(Layer);
}

/// @brief initializes and creates a float array to store the activations in.
/// @param size of array
struct activations*init_activations(int size){
    struct activations*activation_layer = (struct activations*)malloc(sizeof(struct activations));
    if (activation_layer == NULL) {
        perror("Failed to allocate memory for Layer");
        return NULL;
    }
    activation_layer->size = size;
    activation_layer->activations = malloc(size*sizeof(float));
    for (int i = 0; i < size; i++){
        activation_layer->activations[i] = (float)rand()/((float)RAND_MAX) - 0.5;
    }
    return activation_layer;
}

/// @brief Frees the activation struct
/// @param a 
void free_activations(struct activations*a){
    free(a->activations);
    free(a);
}


/// @brief Efficient Forward prop function, Does both 
/// @param A1 Previous activations
/// @param L Layer with weights and biases
/// @param A2 Next activations
void forward_prop_step(struct activations*A1,struct layer*L,struct activations*A2){
    if(A1->size != L->cols){perror("A1's size and L's weight's cols do not match"); exit(1);}
    if(A2->size != L->rows){perror("A2's size and L's weight's rows do not match"); exit(1);}
    memcpy(A2->activations,L->biases,sizeof(float)*A2->size);
    // vectorize the multiplications, then vector add the summations.
    //multiplication vectorization
    float buf[A2->size];
    memset(buf, 0, sizeof(buf));  
    for(int i = 0; i < L->rows; i++){
        __m256 sum_vec = _mm256_setzero_ps();
        int j;
        for(j = 0; j+7<L->cols; j+=8){
            __m256 weightvec = _mm256_loadu_ps(&L_WEIGHT(L,i,j));
            __m256 act_vec = _mm256_loadu_ps(&A1->activations[j]);
            __m256 mulvec = _mm256_mul_ps(weightvec,act_vec);
            sum_vec = _mm256_add_ps(mulvec,sum_vec);
        }
        __m128 bottom = _mm256_castps256_ps128(sum_vec);
        __m128 top = _mm256_extractf128_ps(sum_vec,1);
        bottom = _mm_add_ps(top,bottom);
        bottom = _mm_hadd_ps(bottom,bottom);
        bottom = _mm_hadd_ps(bottom,bottom);
        buf[i] = _mm_cvtss_f32(bottom);
        for(; j < L->cols; j++){ buf[i] += L_WEIGHT(L,i,j)*A1->activations[j]; }
    }
    int i = 0;
    for(; i+7 < L->rows; i+=8){
        __m256 buf_vec = _mm256_loadu_ps(&buf[i]);
        __m256 act_vec = _mm256_loadu_ps(&A2->activations[i]);
        act_vec = _mm256_add_ps(buf_vec,act_vec);
        _mm256_storeu_ps(&A2->activations[i],act_vec);
    }
    for (; i < L->rows; i++){
        A2->activations[i] += buf[i];
    }   
}

/// @brief Applies ReLU to the activations
/// @param A 
void ReLU(struct activations*A){
    for (int i = 0; i < A->size; i++)
    {if(A->activations[i] < 0.0){A->activations[i] = 0.0;}}
}

/// @brief Takes Derivative of ReLU and puts it other activation struct
/// @param A 
/// @param Z_ReLU
void ReLU_derivative(struct activations*A,struct activations*B){
    for (int i = 0; i < A->size; i++) 
    {B->activations[i] = (A->activations[i] <= 0.0) ? 0.0 : 1.0;}
}

/// @brief Applies Softmax to the activation layer
/// @param A 
void softmax(struct activations* A) {
    int k = A->size;
    float max_activation = A->activations[0];
    for (int i = 1; i < k; i++) {
        if (A->activations[i] > max_activation) {
            max_activation = A->activations[i];
        }
    }
    float expsum = 0.0f;
    for (int i = 0; i < k; i++) {
        A->activations[i] = exp(A->activations[i] - max_activation); // Numerical stability
        expsum += A->activations[i];
    }
    if (expsum == 0.0f) {
        perror("Softmax error: expsum is zero");
        exit(1);
    }
    for (int i = 0; i < k; i++) {
        A->activations[i] /= expsum;
    }
}

/// @brief One hot encodes the error function
/// @param k 
/// @return float array with null except kth element as 1
float* one_hot_encode(int k){
    float* final = malloc(sizeof(float)*10);
    for(int i = 0; i < 10; i++){
        if(i == k){final[i] = (float)1;}
        else{final[i] = (float)0;}
    }
    return final;
}

/// @brief Loss function that tells us the error values
/// @param final_layer 
/// @param k size
void loss_function(struct activations* dZ_loss,struct activations* Fl, int k) {
    float* j = one_hot_encode(k);
    if (j == NULL) {
        printf("Failed to allocate memory for one-hot encoding\n");
        exit(1);
    }
    for (int i = 0; i < Fl->size; i++) {
        dZ_loss->activations[i] = Fl->activations[i] - j[i];
    }
    free(j);
}

/// @brief Computes the cross-entropy loss between predicted activations and the true label.
/// @param Fl Predicted activations (output of the softmax layer).
/// @param k True label (integer between 0 and 9).
/// @return Cross-entropy loss value.
float compute_loss(struct activations* Fl, int k) {
    if (k < 0 || k >= Fl->size) {
        perror("Invalid label index");
        exit(1);
    }
    
    float predicted_prob = Fl->activations[k];      
    if (predicted_prob <= 0.0f) {
        predicted_prob = 1e-15; 
    }
    return -log(predicted_prob);
}

/// @brief Calcualtes Gradient in activations given previous gradient.
/// @param dZ_curr The gradient to be calculated
/// @param L Weights and biases of the layer in front
/// @param dZ_prev loss function of layer in front
/// @param A_curr ReLU derivative of the current layer
void calc_grad_activation(struct activations* dZ_curr,struct layer*L,struct activations* dZ_prev,struct activations*A_curr){
    if(dZ_curr->size != A_curr->size){perror("The ReLU deriv and n-1 grad activation matricies do not match");exit(1);}
    if(L->rows != dZ_prev->size){perror("The Layer matricies and gradient layer matricies do not match");exit(1);}
    if(L->cols != dZ_curr->size){perror("The Layer matricies and curr_grad layer matricies do not match");exit(1);}
    // memcpy(dZ_curr->activations,A_curr->activations,sizeof(float)*dZ_curr->size);
    for (int i = 0; i < L->cols; i++) {
        float sum = 0.0f;
        for (int j = 0; j < L->rows; j++) {
            sum += L_WEIGHT(L, j, i) * dZ_prev->activations[j];  // Column-wise sum
        }
        dZ_curr->activations[i] = sum * A_curr->activations[i];
    }
}


/// @brief Conducts 1 step of back propogation
/// @param L Layer's weights and biases
/// @param dL Gradient layer
/// @param dZ Loss function or activation gradient
/// @param A n-1th layer
void back_propogate_step(struct layer*L,struct layer*dL,struct activations* dZ,struct activations* A){
    if(dL->rows != L->rows || dL->cols != L->cols){perror("The Gradient and Layer matrices do not match");exit(1);}
    if(dZ->size != dL->rows){perror("Gradient activation and gradient layer matricies do not match");exit(1);}
    if(A->size != dL->cols){perror("activation and GradientLayer matrices do not match");exit(1);}
    memcpy(dL->biases,dZ->activations,sizeof(float)*dZ->size);
    for (int i = 0; i < dL->rows; i++){
        __m256 dZ_vec = _mm256_set1_ps(dZ->activations[i]);
        int j;
        for (j = 0; j+7 < dL->cols; j+=8){
            __m256 A_vec = _mm256_loadu_ps(&A->activations[j]);
            __m256 weight_vec = _mm256_mul_ps(dZ_vec,A_vec);
            _mm256_storeu_ps(&L_WEIGHT(dL,i,j),weight_vec);
        }
        for (; j<dL->cols; j++) L_WEIGHT(dL,i,j) = dZ->activations[i]*A->activations[j];
    }
}

// void param_update(struct layer* L, struct layer* dL, float LR) {
//     static int avx512_supported = -1;
//     if (avx512_supported == -1) {
//         avx512_supported = __builtin_cpu_supports("avx512f");
//     }

//     if (avx512_supported) {
//         param_update_avx512(L, dL, LR);
//     } else {
//         param_update_scalar(L, dL, LR);
//     }
// }

/// @brief Given original weights, biases and gradient, updates all the values accordingly
/// @param L Layer
/// @param dL Gradient
void param_update(struct layer*L,struct layer*dL, float LR){
    if(dL->rows != L->rows || dL->cols != L->cols){perror("The Gradient and Layer matrices do not match");exit(1);}
    __m256 LR_vec = _mm256_set1_ps(LR);
    int i = 0;
    for(; i+7 < L->rows; i+=8){
        __m256 v_bias = _mm256_loadu_ps(&L->biases[i]);
        __m256 v_dbias = _mm256_loadu_ps(&dL->biases[i]);
        __m256 mulvec = _mm256_mul_ps(LR_vec,v_dbias);
        v_bias = _mm256_sub_ps(v_bias,mulvec);
        _mm256_storeu_ps(&L->biases[i],v_bias);
    }
    for (; i < L->rows; i++) {
        L->biases[i] -= LR * dL->biases[i];
    }
    for (int i = 0; i < dL->rows; i++){
        int j = 0;
        for (; j+7 < dL->cols; j+=8){
            __m256 v_w = _mm256_loadu_ps(&L_WEIGHT(L,i,j));
            __m256 vdW = _mm256_loadu_ps(&L_WEIGHT(dL,i,j));
            __m256 mulvec = _mm256_mul_ps(LR_vec,vdW);
            v_w = _mm256_sub_ps(v_w,mulvec);
            _mm256_storeu_ps(&L_WEIGHT(L,i,j),v_w);
        }
        for (; j < L->cols; j++) {
             L_WEIGHT(L,i,j) += LR*L_WEIGHT(dL,i,j);
        }
    }
    
}

/// @brief Clears the Given layer
/// @param L Layer
void Zero_Layer(struct layer*L){
    if(L->biases) memset(L->biases,0.0f,L->rows*sizeof(float)); //memset to 0;
    if(L->Weights) memset(L->Weights,0.0f,L->rows*L->cols*sizeof(float));
}

/// @brief Inputs image data into activation struct
/// @param pixel_data 
/// @param k index of image
/// @param A 
void input_data(struct pixel_data* pixel_data,int k,struct activations*A){
    int numpx = pixel_data->rows*pixel_data->cols;
    if (A->size != numpx){perror("Wrong layer passed to input");exit(1);}
    for (int i = 0; i < numpx; i++){
        A->activations[i] = pixel_data->neuron_activation[k][i]/255.0;
    }
}

/// @brief Gets the largest activation value and returns it
/// @param A 
/// @return index of highest activation
int get_pred_from_softmax(struct activations *A) {
    int max_index = 0;
    float max_value = A->activations[0];
    for (int i = 1; i < A->size; i++) {
        if (A->activations[i] > max_value) {
            max_value = A->activations[i];
            max_index = i;}
    }
    return max_index;
}


/// @brief Prints out activation values for debugging
/// @param A 
void print_activations(struct activations*A){
    for(int i = 0; i < A->size; i++){printf("%f\n",A->activations[i]);}    
}

/// @brief Prints the contents of a layer struct.
/// @param l Pointer to the layer struct to be printed.
void print_layer(const struct layer* l) {
    if (l == NULL) {
        printf("Layer is NULL.\n");
        return;
    }
    printf("Layer dimensions: rows = %d, cols = %d\n", l->rows, l->cols);
    printf("Weights:\n");
    for (int i = 0; i < l->rows; i++) {
        for (int j = 0; j < l->cols; j++) {
            printf("%8.4f ", L_WEIGHT(l,i,j)); // Format weights for readability
        }
        printf("\n");
    }
    printf("Biases:\n");
    for (int i = 0; i < l->rows; i++) {
        printf("%8.4f ", l->biases[i]); // Format biases for readability
    }
    printf("\n");
}

