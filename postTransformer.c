#include "helpers.c"

void get_out_prob(float* vocabulary, float* input, float* dest, int VOCAB_SIZE) {

    // input @ vocabulary
    float* tmp = (float*)alloc32(sizeof(float) * 8 * VOCAB_SIZE);
    matmul(input, vocabulary, 8, 512, 512, VOCAB_SIZE, tmp);

    // Softmax to get probability
    softmax(tmp, 8, VOCAB_SIZE, dest);

    _aligned_free(tmp);
}
