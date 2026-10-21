#include <malloc.h>
#include "helpers.c"

void FeedForwardNetwork(float* in, float* W1, float* b1, float* W2, float* b2, float* dest) {
    //ffn (w1, b1, w2, b2) :: (512, 2048), (1, 2048), (2048, 512), (1, 512)
    float* hidden = (float*)alloc32(8 * 2048 * 4);

    //first linear
    matmul(in, W1, 8, 512, 512, 2048, hidden);
    add_bias(hidden, b1, 8, 2048);
    relu(hidden, 8 * 2048);

    //second linear
    matmul(hidden, W2, 8, 2048, 2048, 512, dest);
    add_bias(dest, b2, 8, 512);

    _aligned_free(hidden);
}
