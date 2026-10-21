#include <string.h>
#include "helpers.c"

// Use for Input & Output Embeddings Lookup
void lookup (int N, int* tokens, float* embeddings, float* out) {
    // For each token, copy its 512-float row into the matching output row
    for (int i = 0; i < N; i++) {
        memcpy(out + i * 512, embeddings + tokens[i] * 512, 512 * sizeof(float));
    }
}

// Use to add Position Matrix with Input eltwise
void addPosition (float* position_embeddings, float* input, int N, float* output) {
    // Just Add Both
    add_2_arrays(position_embeddings, input, N * 512, N * 512, output);
}