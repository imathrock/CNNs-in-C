#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<math.h>
#include"NeuralNetwork.h"

#include <immintrin.h> // Header for AVX intrinsics


#define L_WEIGHT(L, i, j)  (L->Weights[((i) * (L)->cols) + (j)])

/// @brief Box Muller transform, computes random number on mean=0 and var=1 spectrum. Generates 2 numbers, returns spare on second call.
/// @return random number
float randn() {
    static int has_spare = 0;
    static float spare;
    if (has_spare) { 
        has_spare = 0;
        return spare;}
    has_spare = 1;
    static float u, v, mag;
    do {
        u = (float)rand() / RAND_MAX * 2.0 - 1.0;
        v = (float)rand() / RAND_MAX * 2.0 - 1.0;
        mag = u * u + v * v;
    } while (mag >= 1.0 || mag == 0.0);
    
    mag = sqrt(-2.0 * log(mag) / mag);
    spare = v * mag;
    return u * mag;
}

/// @brief Initializes a layer struct with guards in place to prevent memory leak in case malloc fails.
/// @param rows number of rows in both bias and weight matrix.
/// @param cols number of columns in weight matrix.
/// @param init_type initialization type (see documentation for options)
layer* init_layer(int rows, int cols, int init_type) {
    struct layer* Layer = (struct layer*)malloc(sizeof(struct layer));
    if (Layer == NULL) {
        perror("Failed to allocate memory for Layer");
        return NULL;
    }
    
    Layer->biases = malloc(rows * sizeof(float));
    if (Layer->biases == NULL) {
        perror("Failed to allocate memory for Layer->biases");
        free(Layer);
        return NULL;
    }
    
    Layer->Weights = (float*)malloc(rows * cols * sizeof(float));
    if (Layer->Weights == NULL) {
        perror("Failed to allocate memory for Layer->Weights");
        free(Layer->biases);
        free(Layer);
        return NULL;
    }

    // Set biases to 0
    memset(Layer->biases,0.0f,sizeof(float)*rows);

    // Weight initialization based on type
    float limit, std_dev, fan_in, fan_out, fan_avg;
    fan_in = (float)cols;
    fan_out = (float)rows;
    fan_avg = (fan_in + fan_out) / 2.0f;
    
    switch (init_type) {
        case 0: // Zero initialization
            memset(Layer->Weights,0.0f,sizeof(float)*rows*cols);
            break;

        case 1: // Random uniform [-0.5, 0.5]
            for (int i = 0; i < rows * cols; i++) {
                Layer->Weights[i] = ((float)rand() / RAND_MAX) - 0.5f;
            }
            break;

        case 2: // Random normal (0, 1)
            for (int i = 0; i < rows * cols; i++) {
                Layer->Weights[i] = randn();
            }
            break;

        case 3: // Xavier/Glorot Uniform
            limit = sqrtf(6.0f / (fan_in + fan_out));
            for (int i = 0; i < rows * cols; i++) {
                Layer->Weights[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * limit;
            }
            break;
            
        case 4: // Xavier/Glorot Normal
            std_dev = sqrtf(2.0f / (fan_in + fan_out));
            for (int i = 0; i < rows * cols; i++) {
                Layer->Weights[i] = randn() * std_dev;
            }
            break;
            
        case 5: // He/Kaiming Uniform (for ReLU)
            limit = sqrtf(6.0f / fan_in);
            for (int i = 0; i < rows * cols; i++) {
                Layer->Weights[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * limit;
            }
            break;
            
        case 6: // He/Kaiming Normal (for ReLU)
            std_dev = sqrtf(2.0f / fan_in);
            for (int i = 0; i < rows * cols; i++) {
                Layer->Weights[i] = randn() * std_dev;
            }
            break;
            
        case 7: // LeCun Uniform (for SELU)
            limit = sqrtf(3.0f / fan_in);
            for (int i = 0; i < rows * cols; i++) {
                Layer->Weights[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * limit;
            }
            break;
            
        case 8: // LeCun Normal (for SELU)
            std_dev = sqrtf(1.0f / fan_in);
            for (int i = 0; i < rows * cols; i++) {
                Layer->Weights[i] = randn() * std_dev;
            }
            break;
            
        case 9: // Orthogonal initialization
            // Note: This is a simplified version. True orthogonal init requires SVD
            std_dev = 1.0f;
            for (int i = 0; i < rows * cols; i++) {
                Layer->Weights[i] = randn() * std_dev;
            }
            break;
            
        case 10: // Identity initialization (only works for square matrices)
            for (int i = 0; i < rows * cols; i++) {
                Layer->Weights[i] = 0.0f;
            }
            if (rows == cols) {
                for (int i = 0; i < rows; i++) {
                    Layer->Weights[i * cols + i] = 1.0f;
                }
            }
            break;
            
        case 11: // Variance Scaling Uniform
            limit = sqrtf(3.0f / fan_avg);
            for (int i = 0; i < rows * cols; i++) {
                Layer->Weights[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * limit;
            }
            break;
            
        case 12: // Variance Scaling Normal
            std_dev = sqrtf(1.0f / fan_avg);
            for (int i = 0; i < rows * cols; i++) {
                Layer->Weights[i] = randn() * std_dev;
            }
            break;
            
        case 13: // Truncated Normal (approximated)
            std_dev = sqrtf(2.0f / (fan_in + fan_out));
            for (int i = 0; i < rows * cols; i++) {
                float val;
                do {
                    val = randn() * std_dev;
                } while (fabsf(val) > 2.0f * std_dev); // Truncate at 2 standard deviations
                Layer->Weights[i] = val;
            }
            break;
            
        case 14: // Small random values
            for (int i = 0; i < rows * cols; i++) {
                Layer->Weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
            }
            break;
            
        default: // Default to Xavier/Glorot Normal
            std_dev = sqrtf(2.0f / (fan_in + fan_out));
            for (int i = 0; i < rows * cols; i++) {
                Layer->Weights[i] = randn() * std_dev;
            }
            break;
    }
    
    return Layer;
}

/// @brief Inits the full dense layer struct with the grad buffers
/// @param rows 
/// @param cols 
/// @return pointer to the struct
DenseLayer*init_DenseLayer(int rows,int cols,int init_type){
    DenseLayer*layers = malloc(sizeof(DenseLayer));
    layers->cols = cols; layers->rows = rows; layers->init_type = init_type;
    layers->params = init_layer(rows,cols,init_type);
    layers->param_grad = init_layer(rows,cols,0);
    layers->param_grad_sum = init_layer(rows,cols,0);
    return layers;
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

/// @brief Finalizer for Dense Layer
/// @param DL 
void Free_DenseLayer(DenseLayer*DL){
    free_layer(DL->params);
    free_layer(DL->param_grad);
    free_layer(DL->param_grad_sum);
    free(DL);
}


/// @brief init the activations struct that bundles required buffers
/// @param size 
/// @return 
activations*init_activations(int size){
    activations*act = malloc(sizeof(activations));
    act->size = size;
    act->Z = calloc(size,sizeof(float));
    act->del_Z = calloc(size,sizeof(float));
    act->dZ = calloc(size,sizeof(float));
    return act;
}

/// @brief finalizer for activations
/// @param A 
void Free_activations(activations*A){
    free_activation(A->Z);
    free_activation(A->del_Z);
    free_activation(A->dZ);
    free(A);
}

/// @brief Efficient Forward prop function, Does both 
/// @param A1 Previous activation
/// @param L Layer with weights and biases
/// @param A2 Next activation
void forward_prop_step(struct activation*A1,struct layer*L,struct activation*A2){ 
    if(A1->size != L->cols){perror("A1's size and L's weight's cols do not match"); exit(1);}
    if(A2->size != L->rows){perror("A2's size and L's weight's rows do not match"); exit(1);}
    memcpy(A2->activation,L->biases,sizeof(float)*A2->size);
    // vectorize the multiplications, then vector add the summations.
    //multiplication vectorization
    float buf[A2->size];
    memset(buf, 0, sizeof(buf));  
    for(int i = 0; i < L->rows; i++){
        __m256 sum_vec = _mm256_setzero_ps();
        int j;
        for(j = 0; j+7<L->cols; j+=8){
            __m256 weightvec = _mm256_loadu_ps(&L_WEIGHT(L,i,j));
            __m256 act_vec = _mm256_loadu_ps(&A1->activation[j]);
            __m256 mulvec = _mm256_mul_ps(weightvec,act_vec);
            sum_vec = _mm256_add_ps(mulvec,sum_vec);
        }
        __m128 bottom = _mm256_castps256_ps128(sum_vec);
        __m128 top = _mm256_extractf128_ps(sum_vec,1);
        bottom = _mm_add_ps(top,bottom);
        bottom = _mm_hadd_ps(bottom,bottom);
        bottom = _mm_hadd_ps(bottom,bottom);
        buf[i] = _mm_cvtss_f32(bottom);
        for(; j < L->cols; j++){ buf[i] += L_WEIGHT(L,i,j)*A1->activation[j]; }
    }
    int i = 0;
    for(; i+7 < L->rows; i+=8){
        __m256 buf_vec = _mm256_loadu_ps(&buf[i]);
        __m256 act_vec = _mm256_loadu_ps(&A2->activation[i]);
        act_vec = _mm256_add_ps(buf_vec,act_vec);
        _mm256_storeu_ps(&A2->activation[i],act_vec);
    }
    for (; i < L->rows; i++){
        A2->activation[i] += buf[i];
    }   
}

/// @brief inline function that prevents overflow in exp
/// @param x input value
/// @return safe exponential or clamped value
static inline float safe_exp(float x){ 
    return (x > 500.0f || x < -500.0f) ? 0.0f : expf(x); 
}

/// @brief GELU approximation using tanh
/// @param x input value  
/// @return GELU approximation
static inline float gelu_approx(float x){ 
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x))); 
}

/// @brief Applies enum act_func, simultaneously computes derivatives and stores it.
/// @param A Activations struct
/// @param func act func, default identity.
void activation_function(activations*A,act_func_t func){
    switch (func){
    case ReLU:
        for (int i = 0; i < A->size; i++){
            if(A->Z[i] <= 0){
                A->Z[i] = 0; A->dZ[i] = 0;}
            else{ A->dZ[i] = 1;}
        }
        break;

    case Sigmoid:
        for(int i = 0; i < A->size; i++){
            A->Z[i] = 1.0f/(1.0f+safe_exp(-A->Z[i])); // Fixed: should be -A->Z[i]
            A->dZ[i] = A->Z[i] * (1.0f-A->Z[i]);
        }
        break;

    case Tanh:
        for(int i = 0; i < A->size; i++){
            A->Z[i] = tanhf(A->Z[i]);
            A->dZ[i] = 1.0f - A->Z[i] * A->Z[i]; // Avoid powf, use multiplication
        }        
        break;

    case LeakyRelu:
        {
        const float alpha = 0.01f; 
        for (int i = 0; i < A->size; i++) {
            if (A->Z[i] <= 0) {
                A->Z[i] *= alpha;
                A->dZ[i] = alpha;
            } 
            else {A->dZ[i] = 1.0f;}
        }
        }
        break;

    case PReLU:
        {
        const float alpha = 0.25f; // Should be learnable parameter
        for (int i = 0; i < A->size; i++) {
            if (A->Z[i] <= 0) {
                A->Z[i] *= alpha;
                A->dZ[i] = alpha;
            } 
            else {A->dZ[i] = 1.0f;}
        }
        }
        break;

    case ELU:
        {
        const float alpha = 1.0f;
        for (int i = 0; i < A->size; i++) {
            if (A->Z[i] <= 0) {
                A->Z[i] = alpha * (safe_exp(A->Z[i]) - 1.0f);
                A->dZ[i] = A->Z[i] + alpha; // ELU derivative: f(x) + alpha when x <= 0
            } 
            else {A->dZ[i] = 1.0f;}
        }
        }
        break;

    case SELU:
        {
        const float alpha = 1.6732632423543772f;
        const float scale = 1.0507009873554805f;
        for (int i = 0; i < A->size; i++) {
            if (A->Z[i] <= 0) {
                float exp_z = safe_exp(A->Z[i]);
                A->Z[i] = scale * alpha * (exp_z - 1.0f);
                A->dZ[i] = scale * alpha * exp_z;
            } 
            else {
                A->Z[i] *= scale;
                A->dZ[i] = scale;
            }
        }
        }
        break;

    case GELU:
        for (int i = 0; i < A->size; i++) {
            float x = A->Z[i];
            A->Z[i] = gelu_approx(x);
            // GELU derivative approximation
            float tanh_arg = 0.7978845608f * (x + 0.044715f * x * x * x);
            float tanh_val = tanhf(tanh_arg);
            float sech_sq = 1.0f - tanh_val * tanh_val;
            A->dZ[i] = 0.5f * (1.0f + tanh_val) + 0.5f * x * sech_sq * 0.7978845608f * (1.0f + 0.134145f * x * x);
        }
        break;

    case Swish:
        for (int i = 0; i < A->size; i++) {
            float x = A->Z[i];
            float sigmoid = 1.0f/(1.0f+safe_exp(-x));
            A->Z[i] = x * sigmoid;
            A->dZ[i] = sigmoid + x * sigmoid * (1.0f - sigmoid); // Swish derivative
        }
        break;

    case Softmax:
        {
        int k = A->size;
        float max = A->Z[0];
        for (int i = 1; i < k; i++) {if (A->Z[i] > max) max = A->Z[i];} //find max
        float expsum = 0.0f;
        for (int i = 0; i < k; i++) {
            A->Z[i] = safe_exp(A->Z[i] - max); // Use safe_exp for consistency
            expsum += A->Z[i];
        }
        if (expsum == 0.0f) {
            perror("Softmax error: expsum is zero");
            exit(1);}
        for (int i = 0; i < k; i++) A->Z[i] /= expsum;
        }
        break;
        
    default:
        for (int i = 0; i < A->size; i++) A->dZ[i] = 1.0f; 
        break;
    }
}

/// @brief Applies loss function to final layer
/// @param A activation buffer
/// @param func loss function
/// @param k 
/// @return 
float loss_function(activations*A, loss_func_t func, int k){
    float loss = 0.0f;
    switch (func){
    case L1:{
        for(int i = 0; i < A->size; i++){
            if(i == k) A->dZ[i] = A->Z[i] - 1.0f;
            else A->dZ[i] = A->Z[i];
        }
        loss = fabsf(A->dZ[k]);
        }
        break;

    case L2:{
        for(int i = 0; i < A->size; i++){
            if(i == k) A->dZ[i] = A->Z[i] - 1.0f;
            else A->dZ[i] = A->Z[i];
            loss += A->dZ[i] * A->dZ[i];
        }
        loss *= 0.5f; // standard L2 scaling
        }
        break;

    case CE:{ // assumes Z already softmaxed
        for(int i = 0; i < A->size; i++){
            A->dZ[i] = A->Z[i];
        }
        loss = -logf(A->Z[k] + 1e-9f); 
        A->dZ[k] -= 1.0f; 
        }
        break;

    case MSE:{
        for(int i = 0; i < A->size; i++){
            float y = (i == k) ? 1.0f : 0.0f;
            A->dZ[i] = A->Z[i] - y;
            loss += A->dZ[i] * A->dZ[i];
        }
        loss /= A->size;
        }
        break;

    case MAE:{
        for(int i = 0; i < A->size; i++){
            float y = (i == k) ? 1.0f : 0.0f;
            A->dZ[i] = A->Z[i] - y;
            loss += fabsf(A->dZ[i]);
        }
        loss /= A->size;
        }
        break;

    case HUBER:{
        float delta = 1.0f;
        for(int i = 0; i < A->size; i++){
            float y = (i == k) ? 1.0f : 0.0f;
            A->dZ[i] = A->Z[i] - y;
            float abs_error = fabsf(A->dZ[i]);
            if(abs_error < delta){
                loss += 0.5f * abs_error * abs_error;
            } else {
                loss += delta * (abs_error - 0.5f * delta);
            }
        }
        loss /= A->size;
        }
        break;

    case BCE:{ // binary classification only, assumes A->size == 1
        float y = (k == 1) ? 1.0f : 0.0f;
        float z = A->Z[0];
        A->dZ[0] = z - y;
        loss = -(y * logf(z + 1e-9f) + (1.0f - y) * logf(1.0f - z + 1e-9f));
        }
        break;

    case CCE:{ // multi-class categorical CE (one-hot target)
        for(int i = 0; i < A->size; i++){
            A->dZ[i] = A->Z[i];
        }
        loss = -logf(A->Z[k] + 1e-9f);
        A->dZ[k] -= 1.0f;
        }
        break;

    case SCE:{ // sparse categorical CE, same as CE for int labels
        for(int i = 0; i < A->size; i++){
            A->dZ[i] = A->Z[i];
        }
        loss = -logf(A->Z[k] + 1e-9f);
        A->dZ[k] -= 1.0f;
        }
        break;

    default:
        break;
    }
    return loss;
}

/// @brief Calcualtes Gradient in activation given previous gradient.
/// @param dZ_curr The gradient to be calculated
/// @param L Weights and biases of the layer in front
/// @param dZ_prev loss function of layer in front
/// @param A_curr ReLU derivative of the current layer
void calc_grad_activation(struct activation* dZ_curr,struct layer*L,struct activation* dZ_prev,struct activation*A_curr){
    if(dZ_curr->size != A_curr->size){perror("The ReLU deriv and n-1 grad activation matricies do not match");exit(1);}
    if(L->rows != dZ_prev->size){perror("The Layer matricies and gradient layer matricies do not match");exit(1);}
    if(L->cols != dZ_curr->size){perror("The Layer matricies and curr_grad layer matricies do not match");exit(1);}
    // memcpy(dZ_curr->activation,A_curr->activation,sizeof(float)*dZ_curr->size);
    for (int i = 0; i < L->cols; i++) {
        float sum = 0.0f;
        for (int j = 0; j < L->rows; j++) {
            sum += L_WEIGHT(L, j, i) * dZ_prev->activation[j]; 
        }
        dZ_curr->activation[i] = sum * A_curr->activation[i];
    }
}


/// @brief Conducts 1 step of back propogation
/// @param L Layer's weights and biases
/// @param dL Gradient layer
/// @param dZ Loss function or activation gradient
/// @param A n-1th layer

void back_propogate_step(struct layer*L,struct layer*dL,struct activation* dZ,struct activation* A){
    if(dL->rows != L->rows || dL->cols != L->cols){perror("The Gradient and Layer matrices do not match");exit(1);}
    if(dZ->size != dL->rows){perror("Gradient activation and gradient layer matricies do not match");exit(1);}
    if(A->size != dL->cols){perror("activation and GradientLayer matrices do not match");exit(1);}
    memcpy(dL->biases,dZ->activation,sizeof(float)*dZ->size);
    for (int i = 0; i < dL->rows; i++){
        __m256 dZ_vec = _mm256_set1_ps(dZ->activation[i]);
        int j;
        for (j = 0; j+7 < dL->cols; j+=8){
            __m256 A_vec = _mm256_loadu_ps(&A->activation[j]);
            __m256 weight_vec = _mm256_mul_ps(dZ_vec,A_vec);
            _mm256_storeu_ps(&L_WEIGHT(dL,i,j),weight_vec);
        }
        for (; j<dL->cols; j++) L_WEIGHT(dL,i,j) = dZ->activation[i]*A->activation[j];
    }
}

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
void input_data(struct pixel_data* pixel_data,int k,struct activation*A){
    int numpx = pixel_data->rows*pixel_data->cols;
    if (A->size != numpx){perror("Wrong layer passed to input");exit(1);}
    for (int i = 0; i < numpx; i++){
        A->activation[i] = pixel_data->neuron_activation[k][i]/255.0;
    }
}

/// @brief Gets the largest activation value and returns it
/// @param A 
/// @return index of highest activation
int get_pred_from_softmax(struct activation *A) {
    int max_index = 0;
    float max_value = A->activation[0];
    for (int i = 1; i < A->size; i++) {
        if (A->activation[i] > max_value) {
            max_value = A->activation[i];
            max_index = i;}
    }
    return max_index;
}

/// @brief Standardizes the activations
/// @param A 
void StandardizeActivations(activation *A){
    float sum = 0.0f;
    __m256 sumvec = _mm256_setzero_ps();
    int i;
    for (i = 0; i+7 < A->size; i+=8){
        __m256 actvec = _mm256_loadu_ps(&A->activation[i]);
        sumvec = _mm256_add_ps(sumvec,actvec);
    }
    __m128 bottom = _mm256_castps256_ps128(sumvec);
    __m128 top = _mm256_extractf128_ps(sumvec,1);
    bottom = _mm_add_ps(top,bottom);
    bottom = _mm_hadd_ps(bottom,bottom);
    bottom = _mm_hadd_ps(bottom,bottom);
    sum = _mm_cvtss_f32(bottom);
    for (; i < A->size; i++) sum += A->activation[i];
    float mean = sum/A->size;
    float var = 0.0f;
    __m256 meanvec = _mm256_set1_ps(mean);
    __m256 varvec = _mm256_setzero_ps();
    i = 0;
    for(; i+7 < A->size; i+=8){
        __m256 actvec = _mm256_loadu_ps(&A->activation[i]);
        varvec = _mm256_sub_ps(actvec,meanvec);
        varvec = _mm256_mul_ps(varvec,varvec);
    }
    bottom = _mm256_castps256_ps128(varvec);
    top = _mm256_extractf128_ps(varvec,1);
    bottom = _mm_add_ps(top,bottom);
    bottom = _mm_hadd_ps(bottom,bottom);
    bottom = _mm_hadd_ps(bottom,bottom);
    var = _mm_cvtss_f32(bottom);
    for(; i < A->size; i++) var += pow(A->activation[i]-mean,2);
    var /= A->size;
    var = sqrt(var) + 0.00000001;
    varvec = _mm256_set1_ps(var);
    i = 0;
    for(; i+7 < A->size; i+=8){
        __m256 actvec = _mm256_loadu_ps(&A->activation[i]);
        actvec = _mm256_sub_ps(actvec, meanvec);
        actvec = _mm256_div_ps(actvec,varvec);
        _mm256_storeu_ps(&A->activation[i],actvec);
    }
    for(;i < A->size; i++) A->activation[i] /= var;
}

/// @brief Prints out activation values for debugging
/// @param A 
void print_activation(struct activation*A){
    for(int i = 0; i < A->size; i++){printf("%f\n",A->activation[i]);}    
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

