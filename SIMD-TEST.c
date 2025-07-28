// #include <stdio.h>   // For input/output functions
// #include <xmmintrin.h> // Header for SSE intrinsics

// void addVectors_SSE(const float* a, const float* b, float* c, int n) {
//     // Process vectors in chunks of 4 floats (128 bits)
//     int i = 0;
//     for (; i + 3 < n; i += 4) {
//         // Load 4 floats from 'a' into an SSE register
//         __m128 va = _mm_loadu_ps(&a[i]); // Use _mm_loadu_ps for unaligned data, _mm_load_ps for aligned data
//         // Load 4 floats from 'b' into an SSE register
//         __m128 vb = _mm_loadu_ps(&b[i]);
//         // Add the corresponding elements in the two SSE registers
//         __m128 vc = _mm_add_ps(va, vb); //
//         // Store the result back into 'c'
//         _mm_storeu_ps(&c[i], vc); //
//     }

//     // Handle any remaining elements (if n is not a multiple of 4)
//     for (; i < n; ++i) {
//         c[i] = a[i] + b[i];
//     }
// }

// int main() {
//     // Example usage
//     const int N = 10; // Number of elements in the vectors
//     float a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
//     float b[] = {10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
//     float c[N]; // Result vector

//     addVectors_SSE(a, b, c, N);

//     printf("Vector A: ");
//     for (int i = 0; i < N; i++) {
//         printf("%.1f ", a[i]);
//     }
//     printf("\n");

//     printf("Vector B: ");
//     for (int i = 0; i < N; i++) {
//         printf("%.1f ", b[i]);
//     }
//     printf("\n");

//     printf("Result Vector C (A + B): ");
//     for (int i = 0; i < N; i++) {
//         printf("%.1f ", c[i]);
//     }
//     printf("\n");

//     return 0;
// }

#include <stdio.h>    // For input/output functions
#include <immintrin.h> // Header for AVX intrinsics

void addVectors_AVX(const float* a, const float* b, float* c, int n) {
    // Process vectors in chunks of 8 floats (256 bits)
    int i = 0;
    for (; i + 7 < n; i += 8) {
        // Load 8 floats from 'a' into an AVX register
        __m256 va = _mm256_loadu_ps(&a[i]); // Use _mm256_loadu_ps for unaligned data
        // Load 8 floats from 'b' into an AVX register
        __m256 vb = _mm256_loadu_ps(&b[i]);
        // Add the corresponding elements in the two AVX registers
        __m256 vc = _mm256_add_ps(va, vb);
        // Store the result back into 'c'
        _mm256_storeu_ps(&c[i], vc);
    }

    // Handle any remaining elements (if n is not a multiple of 8)
    for (; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    // Example usage
    const int N = 10; // Number of elements in the vectors
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    float b[] = {10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    float c[N]; // Result vector

    addVectors_AVX(a, b, c, N);

    printf("Vector A: ");
    for (int i = 0; i < N; i++) {
        printf("%.1f ", a[i]);
    }
    printf("\n");

    printf("Vector B: ");
    for (int i = 0; i < N; i++) {
        printf("%.1f ", b[i]);
    }
    printf("\n");

    printf("Result Vector C (A + B): ");
    for (int i = 0; i < N; i++) {
        printf("%.1f ", c[i]);
    }
    printf("\n");

    return 0;
}
