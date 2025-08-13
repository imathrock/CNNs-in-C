#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<math.h>

#include"NeuralNetwork.h"

#include <immintrin.h> // Header for AVX intrinsics


#define L_WEIGHT(L, i, j, cols)  (L->Weights[((i) * cols) + (j)])
#define L_WEIGHT_T(L, i, j, rows)  (L->Weights_T[((i) * rows) + (j)])

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
layer* init_layer(int rows, int cols, int init_type){
    layer* Layer = (layer*)malloc(sizeof(layer));
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
    Layer->Weights_T = (float*)malloc(rows * cols * sizeof(float));
    if (Layer->Weights_T == NULL) {
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
    memcpy(Layer->Weights_T,Layer->Weights,sizeof(float)*rows*cols);
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
void free_layer(layer*Layer){
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
    act->gprime = calloc(size,sizeof(float));
    act->dZ = calloc(size,sizeof(float));
    return act;
}

/// @brief finalizer for activations
/// @param A 
void Free_activations(activations*A){
    free(A->Z);
    free(A->gprime);
    free(A->dZ);
    free(A);
}

/// @brief inline function that prevents overflow in exp
/// @param x input value
/// @return safe exponential or clamped value
static inline float safe_exp(float x){ 
    return (x > 500.0f) ? 0.0f : (x < -500.0f ? 0.0f : expf(x));
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
    case ReLU: {
        __m256 zeros = _mm256_setzero_ps();
        __m256 ones  = _mm256_set1_ps(1.0f);
        int i =0;
        for (; i+7 < A->size; i += 8) {
            __m256 data = _mm256_loadu_ps(&A->Z[i]);
            __m256 mask = _mm256_cmp_ps(data, zeros, _CMP_GT_OS);
            __m256 gprime = _mm256_and_ps(mask, ones);
            _mm256_storeu_ps(&A->gprime[i], gprime);
            __m256 result = _mm256_max_ps(data, zeros);
            _mm256_storeu_ps(&A->Z[i], result);
        }
        for (; i < A->size; i++){
            if (A->Z[i] <= 0) { A->Z[i] = 0.0f; A->gprime[i] = 0.0f;}
            else {A->gprime[i] = 1;}
        }
        
    }
    break;
        break;

    case Sigmoid:
        for(int i = 0; i < A->size; i++){
            A->Z[i] = 1.0f/(1.0f+safe_exp(-A->Z[i])); // Fixed: should be -A->Z[i]
            A->gprime[i] = A->Z[i] * (1.0f-A->Z[i]);
        }
        break;

    case Tanh:
        for(int i = 0; i < A->size; i++){
            A->Z[i] = tanhf(A->Z[i]);
            A->gprime[i] = 1.0f - A->Z[i] * A->Z[i]; // Avoid powf, use multiplication
        }        
        break;

    case LeakyRelu:
        {
        const float alpha = 0.01f; 
        for (int i = 0; i < A->size; i++) {
            if (A->Z[i] <= 0) {
                A->Z[i] *= alpha;
                A->gprime[i] = alpha;
            } 
            else {A->gprime[i] = 1.0f;}
        }
        }
        break;

    case PReLU:
        {
        const float alpha = 0.25f; // Should be learnable parameter
        for (int i = 0; i < A->size; i++) {
            if (A->Z[i] <= 0) {
                A->Z[i] *= alpha;
                A->gprime[i] = alpha;
            } 
            else {A->gprime[i] = 1.0f;}
        }
        }
        break;

    case ELU:
        {
        const float alpha = 1.0f;
        for (int i = 0; i < A->size; i++) {
            if (A->Z[i] <= 0) {
                A->Z[i] = alpha * (safe_exp(A->Z[i]) - 1.0f);
                A->gprime[i] = A->Z[i] + alpha; // ELU derivative: f(x) + alpha when x <= 0
            } 
            else {A->gprime[i] = 1.0f;}
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
                A->gprime[i] = scale * alpha * exp_z;
            } 
            else {
                A->Z[i] *= scale;
                A->gprime[i] = scale;
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
            A->gprime[i] = 0.5f * (1.0f + tanh_val) + 0.5f * x * sech_sq * 0.7978845608f * (1.0f + 0.134145f * x * x);
        }
        break;

    case Swish:
        for (int i = 0; i < A->size; i++) {
            float x = A->Z[i];
            float sigmoid = 1.0f/(1.0f+safe_exp(-x));
            A->Z[i] = x * sigmoid;
            A->gprime[i] = sigmoid + x * sigmoid * (1.0f - sigmoid); // Swish derivative
        }
        break;

    case Softmax:
        {
        int k = A->size;
        // Numerically stable softmax with robust fallbacks
        float max_val = -INFINITY;
        for (int i = 0; i < k; i++) {
            if (isfinite(A->Z[i]) && A->Z[i] > max_val) max_val = A->Z[i];
        }
        if (!isfinite(max_val)) max_val = 0.0f; // fallback if all are non-finite

        float expsum = 0.0f;
        for (int i = 0; i < k; i++) {
            float e = expf(A->Z[i] - max_val);
            if (!isfinite(e)) e = 0.0f; // guard
            A->Z[i] = e;
            expsum += e;
        }

        if (!(expsum > 0.0f) || !isfinite(expsum)) {
            // Fallback: compute without max shift but clamp inputs to avoid overflow/underflow
            expsum = 0.0f;
            for (int i = 0; i < k; i++) {
                float z = A->Z[i];
                // Recover original logits approximately by treating current A->Z as logits in failure case
                // Clamp to [-80, 80] to keep expf well-defined in float
                float clamped = fmaxf(fminf(z, 80.0f), -80.0f);
                float e = expf(clamped);
                if (!isfinite(e)) e = 0.0f;
                A->Z[i] = e;
                expsum += e;
            }
            if (!(expsum > 0.0f) || !isfinite(expsum)) {
                // As a last resort, output uniform distribution
                float invk = 1.0f / (float)k;
                for (int i = 0; i < k; i++) A->Z[i] = invk;
                break;
            }
        }

        float invsum = 1.0f / expsum;
        for (int i = 0; i < k; i++) A->Z[i] *= invsum;
        }
        break;
        
    default:
        for (int i = 0; i < A->size; i++) A->gprime[i] = 1.0f; 
        break;
    }
}

/// @brief Activation function for inference
/// @param A Activations struct
/// @param func act func, default identity.
void inference_activation_function(activations* A, act_func_t func) {
    switch (func) {
    case ReLU:
    {
        __m256 zeros = _mm256_setzero_ps();
        int i = 0;
        for (; i + 7 < A->size; i += 8) {
            __m256 data = _mm256_loadu_ps(&A->Z[i]);
            __m256 result = _mm256_max_ps(data, zeros);
            _mm256_storeu_ps(&A->Z[i], result);
        }
        for (; i < A->size; i++) 
        {if (A->Z[i] < 0.0f) {A->Z[i] = 0.0f;}}
    }
    break;

    case Sigmoid:
        for (int i = 0; i < A->size; i++) {
            A->Z[i] = 1.0f / (1.0f + safe_exp(-A->Z[i])); 
        }
        break;

    case Tanh:
        for (int i = 0; i < A->size; i++) {
            A->Z[i] = tanhf(A->Z[i]);
        }
        break;

    case LeakyRelu: {
        const float alpha = 0.01f; 
        for (int i = 0; i < A->size; i++) {
            if (A->Z[i] <= 0) { A->Z[i] *= alpha; }
        }
        break;
    }

    case PReLU: {
        const float alpha = 0.25f; // Should be learnable parameter
        for (int i = 0; i < A->size; i++) {
            if (A->Z[i] <= 0) { A->Z[i] *= alpha; }
        }
        break;
    }

    case ELU: {
        const float alpha = 1.0f;
        for (int i = 0; i < A->size; i++) {
            if (A->Z[i] <= 0) {
                A->Z[i] = alpha * (safe_exp(A->Z[i]) - 1.0f);
            }
        }
        break;
    }

    case SELU: {
        const float alpha = 1.6732632423543772f;
        const float scale = 1.0507009873554805f;
        for (int i = 0; i < A->size; i++) {
            if (A->Z[i] <= 0) {
                float exp_z = safe_exp(A->Z[i]);
                A->Z[i] = scale * alpha * (exp_z - 1.0f);
            } else {
                A->Z[i] *= scale;
            }
        }
        break;
    }

    case GELU:
        for (int i = 0; i < A->size; i++) {
            float x = A->Z[i];
            A->Z[i] = gelu_approx(x);
        }
        break;

    case Swish:
        for (int i = 0; i < A->size; i++) {
            float x = A->Z[i];
            float sigmoid = 1.0f / (1.0f + safe_exp(-x));
            A->Z[i] = x * sigmoid;
        }
        break;

    case Softmax: {
        int k = A->size;
        // Numerically stable softmax with robust fallbacks
        float max_val = -INFINITY;
        for (int i = 0; i < k; i++) {
            if (isfinite(A->Z[i]) && A->Z[i] > max_val) max_val = A->Z[i];
        }
        if (!isfinite(max_val)) max_val = 0.0f; // fallback if all are non-finite

        float expsum = 0.0f;
        for (int i = 0; i < k; i++) {
            float e = expf(A->Z[i] - max_val);
            if (!isfinite(e)) e = 0.0f; // guard
            A->Z[i] = e;
            expsum += e;
        }

        if (!(expsum > 0.0f) || !isfinite(expsum)) {
            // Fallback: compute without max shift but clamp inputs to avoid overflow/underflow
            expsum = 0.0f;
            for (int i = 0; i < k; i++) {
                float z = A->Z[i];
                // Clamp to [-80, 80] to keep expf well-defined in float
                float clamped = fmaxf(fminf(z, 80.0f), -80.0f);
                float e = expf(clamped);
                if (!isfinite(e)) e = 0.0f;
                A->Z[i] = e;
                expsum += e;
            }
            if (!(expsum > 0.0f) || !isfinite(expsum)) {
                // As a last resort, output uniform distribution
                float invk = 1.0f / (float)k;
                for (int i = 0; i < k; i++) A->Z[i] = invk;
                break;
            }
        }

        float invsum = 1.0f / expsum;
        for (int i = 0; i < k; i++) {
            A->Z[i] *= invsum;
        }
        break;
    }

    default:
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
    case L1loss:{
        for(int i = 0; i < A->size; i++){
            if(i == k) A->gprime[i] = A->Z[i] - 1.0f;
            else A->gprime[i] = A->Z[i];
            A->dZ[i] = A->gprime[i]; // Set dZ for backprop
        }
        loss = fabsf(A->gprime[k]);
        }
        break;

    case L2loss:{
        for(int i = 0; i < A->size; i++){
            if(i == k) A->gprime[i] = A->Z[i] - 1.0f;
            else A->gprime[i] = A->Z[i];
            A->dZ[i] = A->gprime[i]; // Set dZ for backprop
            loss += A->gprime[i] * A->gprime[i];
        }
        loss *= 0.5f; // standard L2 scaling
        }
        break;

    case CE:{ // assumes Z already softmaxed
        for(int i = 0; i < A->size; i++){
            A->gprime[i] = A->Z[i];
            A->dZ[i] = A->gprime[i]; // For softmax+CE, dZ = y_hat - y
        }
        loss = -logf(A->Z[k] + 1e-9f);
        A->gprime[k] -= 1.0f;
        A->dZ[k] -= 1.0f;
        }
        break;

    case MSE:{
        for(int i = 0; i < A->size; i++){
            float y = (i == k) ? 1.0f : 0.0f;
            A->gprime[i] = A->Z[i] - y;
            A->dZ[i] = A->gprime[i];
            loss += A->gprime[i] * A->gprime[i];
        }
        loss /= A->size;
        }
        break;

    case MAE:{
        for(int i = 0; i < A->size; i++){
            float y = (i == k) ? 1.0f : 0.0f;
            A->gprime[i] = A->Z[i] - y;
            A->dZ[i] = A->gprime[i];
            loss += fabsf(A->gprime[i]);
        }
        loss /= A->size;
        }
        break;

    case HUBER:{
        float delta = 1.0f;
        for(int i = 0; i < A->size; i++){
            float y = (i == k) ? 1.0f : 0.0f;
            A->gprime[i] = A->Z[i] - y;
            A->dZ[i] = A->gprime[i];
            float abs_error = fabsf(A->gprime[i]);
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
        A->gprime[0] = z - y;
        A->dZ[0] = A->gprime[0];
        loss = -(y * logf(z + 1e-9f) + (1.0f - y) * logf(1.0f - z + 1e-9f));
        }
        break;

    case CCE:{ // multi-class categorical CE (one-hot target)
        for(int i = 0; i < A->size; i++){
            A->gprime[i] = A->Z[i];
            A->dZ[i] = A->gprime[i];
        }
        loss = -logf(A->Z[k] + 1e-9f);
        A->gprime[k] -= 1.0f;
        A->dZ[k] -= 1.0f;
        }
        break;

    case SCE:{ // sparse categorical CE, same as CE for int labels
        for(int i = 0; i < A->size; i++){
            A->gprime[i] = A->Z[i];
            A->dZ[i] = A->gprime[i];
        }
        loss = -logf(A->Z[k] + 1e-9f);
        A->gprime[k] -= 1.0f;
        A->dZ[k] -= 1.0f;
        }
        break;

    default:
        break;
    }
    return loss;
}

/// @brief Efficient Forward prop function with AVX2 intrinsics.
/// @param A1 Previous activation
/// @param L Layer with weights and biases
/// @param A2 Next activation
void forward_prop_step(activations*A1, DenseLayer*L,activations*A2){ 
    if(A1->size != L->cols){perror("A1's size and L's weight's cols do not match"); exit(1);}
    if(A2->size != L->rows){perror("A2's size and L's weight's rows do not match"); exit(1);}
    memcpy(A2->Z,L->params->biases,sizeof(float)*A2->size);
    for(int i = 0; i < L->rows; i++){
        __m256 sum_vec = _mm256_setzero_ps();
        int j;
        for(j = 0; j+7<L->cols; j+=8){
            __m256 weightvec = _mm256_loadu_ps(&L_WEIGHT(L->params,i,j,L->cols));
            __m256 act_vec = _mm256_loadu_ps(&A1->Z[j]);
            sum_vec = _mm256_fmadd_ps(weightvec,act_vec,sum_vec);
        }
        __m128 bottom = _mm256_castps256_ps128(sum_vec);
        __m128 top = _mm256_extractf128_ps(sum_vec,1);
        bottom = _mm_add_ps(top,bottom);
        bottom = _mm_hadd_ps(bottom,bottom);
        bottom = _mm_hadd_ps(bottom,bottom);
        float dot = _mm_cvtss_f32(bottom);
        for(; j < L->cols; j++){ dot += L_WEIGHT(L->params,i,j,L->cols)*A1->Z[j]; }
        A2->Z[i] += dot;
    }
}

/// @brief Calcualtes Gradient in activation given previous gradient.
/// @param A1 The gradient to be calculated
/// @param L Weights and biases of the layer in front
/// @param dZ_prev loss function 
void calc_grad_activation(activations* A1,DenseLayer*L,activations* A2){
    if(L->rows != A2->size){perror("The Layer matricies and gradient layer matricies do not match");exit(1);}
    if(L->cols != A1->size){perror("The Layer matricies and curr_grad layer matricies do not match");exit(1);}
    for (int i = 0; i < L->cols; i++) {
        float sum = 0.0f;
        __m256 sumvec = _mm256_setzero_ps();
        int j = 0;
        const float*weights = &L_WEIGHT_T(L->params, i, 0, L->rows);
        for (; j+7 < L->rows; j+=8) {
            __m256 A2vec = _mm256_load_ps(&A2->dZ[j]);
            __m256 weightvec = _mm256_load_ps(&weights[j]);
            sumvec = _mm256_fmadd_ps(weightvec,A2vec,sumvec);
        }
        __m128 bottom = _mm256_castps256_ps128(sumvec);
        __m128 top = _mm256_extractf128_ps(sumvec,1);
        bottom = _mm_add_ps(top,bottom);
        bottom = _mm_hadd_ps(bottom,bottom);
        bottom = _mm_hadd_ps(bottom,bottom);
        sum = _mm_cvtss_f32(bottom);
        for (; j < L->rows; j++) {
            sum += weights[j] * A2->dZ[j]; 
        }
        A1->dZ[i] = sum * A1->gprime[i];
    }
}


/// @brief Conducts 1 step of back propogation
/// @param L Layer's weights and biases
/// @param dL Gradient layer
/// @param dZ activation gradient
/// @param A n-1th layer
void back_propogate_step(activations*A1,DenseLayer*L,activations* A2){
    if(A2->size != L->rows){perror("Gradient activation and gradient layer matricies do not match");exit(1);}
    if(A1->size != L->cols){perror("activation and GradientLayer matrices do not match");exit(1);}
    memcpy(L->param_grad->biases,A2->dZ,sizeof(float)*A2->size);
    for (int i = 0; i < L->rows; i++){
        __m256 dZ_vec = _mm256_set1_ps(A2->dZ[i]);
        int j;
        for (j = 0; j+7 < L->cols; j+=8){
            __m256 A_vec = _mm256_loadu_ps(&A1->Z[j]);
            __m256 weight_vec = _mm256_mul_ps(dZ_vec,A_vec);
            _mm256_storeu_ps(&L_WEIGHT(L->param_grad,i,j,L->cols),weight_vec);
        }
        for (; j<L->cols; j++) L_WEIGHT(L->param_grad,i,j,L->cols) = A2->dZ[i]*A1->Z[j];
    }
}

/// @brief Gradient Accumulator
/// @param L 
/// @param LR 
void grad_accum(DenseLayer* L, float LR) {
    __m256 LR_vec = _mm256_set1_ps(LR);

    // Accumulate biases
    int i = 0;
    for (; i + 7 < L->rows; i += 8) {
        __m256 grad = _mm256_loadu_ps(&L->param_grad->biases[i]);
        __m256 acc = _mm256_loadu_ps(&L->param_grad_sum->biases[i]);
        __m256 scaled = _mm256_mul_ps(LR_vec, grad);
        acc = _mm256_add_ps(acc, scaled);
        _mm256_storeu_ps(&L->param_grad_sum->biases[i], acc);
    }
    for (; i < L->rows; i++) {
        L->param_grad_sum->biases[i] += LR * L->param_grad->biases[i];
    }

    // Accumulate weights
    for (int i = 0; i < L->rows; i++) {
        int j = 0;
        for (; j + 7 < L->cols; j += 8) {
            __m256 grad = _mm256_loadu_ps(&L_WEIGHT(L->param_grad, i, j, L->cols));
            __m256 acc = _mm256_loadu_ps(&L_WEIGHT(L->param_grad_sum, i, j, L->cols));
            __m256 scaled = _mm256_mul_ps(LR_vec, grad);
            acc = _mm256_add_ps(acc, scaled);
            _mm256_storeu_ps(&L_WEIGHT(L->param_grad_sum, i, j, L->cols), acc);
        }
        for (; j < L->cols; j++) {
            L_WEIGHT(L->param_grad_sum, i, j, L->cols) += LR * L_WEIGHT(L->param_grad, i, j, L->cols);
        }
    }
}


/// @brief Given original weights, biases and gradient, updates all the values accordingly
/// @param DenseLayer
/// @param LR learning rate
void update_weights(DenseLayer*L, float LR){
    __m256 LR_vec = _mm256_set1_ps(LR);
    int i = 0;
    for(; i+7 < L->rows; i+=8){
        __m256 v_bias = _mm256_loadu_ps(&L->params->biases[i]);
        __m256 v_dbias = _mm256_loadu_ps(&L->param_grad_sum->biases[i]);
        __m256 mulvec = _mm256_mul_ps(LR_vec,v_dbias);
        v_bias = _mm256_sub_ps(v_bias,mulvec);
        _mm256_storeu_ps(&L->params->biases[i],v_bias);
    }
    for (; i < L->rows; i++) {
        L->params->biases[i] -= LR * L->param_grad_sum->biases[i];
    }
    for (int i = 0; i < L->rows; i++){
        int j = 0;
        for (; j+7 < L->cols; j+=8){
            __m256 v_w = _mm256_loadu_ps(&L_WEIGHT(L->params,i,j,L->cols));
            __m256 vdW = _mm256_loadu_ps(&L_WEIGHT(L->param_grad_sum,i,j,L->cols));
            __m256 mulvec = _mm256_mul_ps(LR_vec,vdW);
            v_w = _mm256_sub_ps(v_w,mulvec);
            _mm256_storeu_ps(&L_WEIGHT(L->params,i,j,L->cols),v_w);
            for (int k = 0; k < 8; k++) {
                L_WEIGHT_T(L->params, j + k, i, L->rows) = L_WEIGHT(L->params, i, j + k, L->cols);
            }
        }
        for (; j < L->cols; j++) {
            float new_w = L_WEIGHT(L->params, i, j, L->cols) - LR * L_WEIGHT(L->param_grad_sum, i, j, L->cols);
            L_WEIGHT(L->params, i, j, L->cols) = new_w;
            L_WEIGHT_T(L->params, j, i, L->rows) = new_w;
        }
    }
}

/// @brief Zeroes out gradients
/// @param L 
void zero_grad(DenseLayer*L){
    memset(L->param_grad->biases,0.0f,L->rows*sizeof(float));
    memset(L->param_grad->Weights,0.0f,L->rows*L->cols*sizeof(float));
    memset(L->param_grad_sum->biases,0.0f,L->rows*sizeof(float));
    memset(L->param_grad_sum->Weights,0.0f,L->rows*L->cols*sizeof(float));
}

/// @brief Inputs image data into activation struct
/// @param pixel_data 
/// @param k index of image
/// @param A 
void input_data(struct pixel_data* pixel_data,int k,activations*A){
    int numpx = pixel_data->rows*pixel_data->cols;
    if (A->size != numpx){perror("Wrong layer passed to input");exit(1);}
    memcpy(A->Z,pixel_data->neuron_activation[k],sizeof(float)*numpx);
}

/// @brief Gets the largest activation value and returns it
/// @param A 
/// @return index of highest activation
int get_pred_from_softmax(activations *A) {
    int max_index = 0;
    float max_value = A->Z[0];
    for (int i = 1; i < A->size; i++) {
        if (A->Z[i] > max_value) {
            max_value = A->Z[i];
            max_index = i;}
    }
    return max_index;
}

/// @brief Standardizes the activations
/// @param A 
void StandardizeActivations(activations *A){
    float sum = 0.0f;
    __m256 sumvec = _mm256_setzero_ps();
    int i;
    for (i = 0; i+7 < A->size; i+=8){
        __m256 actvec = _mm256_loadu_ps(&A->Z[i]);
        sumvec = _mm256_add_ps(sumvec,actvec);
    }
    __m128 bottom = _mm256_castps256_ps128(sumvec);
    __m128 top = _mm256_extractf128_ps(sumvec,1);
    bottom = _mm_add_ps(top,bottom);
    bottom = _mm_hadd_ps(bottom,bottom);
    bottom = _mm_hadd_ps(bottom,bottom);
    sum = _mm_cvtss_f32(bottom);
    for (; i < A->size; i++) sum += A->Z[i];
    float mean = sum/A->size;
    float var = 0.0f;
    __m256 meanvec = _mm256_set1_ps(mean);
    __m256 varvec = _mm256_setzero_ps();
    i = 0;
    for(; i+7 < A->size; i+=8){
        __m256 actvec = _mm256_loadu_ps(&A->Z[i]);
        varvec = _mm256_sub_ps(actvec,meanvec);
        varvec = _mm256_mul_ps(varvec,varvec);
    }
    bottom = _mm256_castps256_ps128(varvec);
    top = _mm256_extractf128_ps(varvec,1);
    bottom = _mm_add_ps(top,bottom);
    bottom = _mm_hadd_ps(bottom,bottom);
    bottom = _mm_hadd_ps(bottom,bottom);
    var = _mm_cvtss_f32(bottom);
    for(; i < A->size; i++) var += pow(A->Z[i]-mean,2);
    var /= A->size;
    var = sqrt(var) + 0.00000001;
    varvec = _mm256_set1_ps(var);
    i = 0;
    for(; i+7 < A->size; i+=8){
        __m256 actvec = _mm256_loadu_ps(&A->Z[i]);
        actvec = _mm256_sub_ps(actvec, meanvec);
        actvec = _mm256_div_ps(actvec,varvec);
        _mm256_storeu_ps(&A->Z[i],actvec);
    }
    for(;i < A->size; i++) A->Z[i] /= var;
}

// /// @brief Prints out activation values for debugging
// /// @param A 
// void print_activation(struct activation*A){
//     for(int i = 0; i < A->size; i++){printf("%f\n",A->activation[i]);}    
// }

// /// @brief Prints the contents of a layer struct.
// /// @param l Pointer to the layer struct to be printed.
// void print_layer(const struct layer* l) {
//     if (l == NULL) {
//         printf("Layer is NULL.\n");
//         return;
//     }
//     printf("Layer dimensions: rows = %d, cols = %d\n", l->rows, l->cols);
//     printf("Weights:\n");
//     for (int i = 0; i < l->rows; i++) {
//         for (int j = 0; j < l->cols; j++) {
//             printf("%8.4f ", L_WEIGHT(l,i,j)); // Format weights for readability
//         }
//         printf("\n");
//     }
//     printf("Biases:\n");
//     for (int i = 0; i < l->rows; i++) {
//         printf("%8.4f ", l->biases[i]); // Format biases for readability
//     }
//     printf("\n");
// }

