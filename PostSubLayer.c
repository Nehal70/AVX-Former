#include <malloc.h>
#include "helpers.c"

void layerNorm(float* input, float* dest, float* X, float* gamma, float* beta) {
    const float eps = 1e-5f;
    float* temp = (float*)alloc32(sizeof(float) * 8 * 512);
    float* centered = (float*)alloc32(sizeof(float) * 512);
    float* squared = (float*)alloc32(sizeof(float) * 512);

    // Residual: temp = input + X
    add_2_arrays(input, X, 8 * 512, 8 * 512, temp);

    // Loop through each row
    for (int t = 0; t < 8; t++) {
        // Get the row
        float* row = temp + t * 512;
        // Get mean of the row
        float mu = row_mean(row, 512);
        // broadcast mean to centered
        sub_broadcast(row, 512, mu, centered);
        // square centered
        square(centered, 512, squared);
        // 
        float var = row_mean(squared, 512);
        float inv_std = inv_sqrt(var + eps);
        scale_mul_add(centered, 512, inv_std, gamma, beta, dest + t * 512);
    }

    _aligned_free(temp);
    _aligned_free(centered);
    _aligned_free(squared);
}
