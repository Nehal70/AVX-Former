#include <malloc.h>
#include <string.h>
#include "helpers.c"

void SingleHeadAttention(float* X, float* mem, float* Wk, float* Wq, float* Wv, int dk, int t, float* O, int causal) {

    if (mem == 0) mem = X;

    // Get K, Q, V
    float* K = (float*) alloc32(t*dk*4); matmul(mem, Wk, t, 512, 512, dk, K);
    float* Q = (float*) alloc32(t*dk*4); matmul(X, Wq, t, 512, 512, dk, Q);
    float* V = (float*) alloc32(t*dk*4); matmul(mem, Wv, t, 512, 512, dk, V);

    // Get S
    float* S = (float*) alloc32(t*t*4); transpose_matmul(Q, K, t, dk, t, dk, S);
    //scale by 1/sqrt(dk)
    sdot_product(inv_sqrt((float)dk), S, t * t, S);

    if (causal) {
        //mask future tokens
        for (int i = 0; i < t; i++) {
            for (int j = i + 1; j < t; j++) {
                S[i * t + j] = -1e9f;
            }
        }
    }

    // Get P
    exp_vec(S, t * t); 
    float* P = (float*) alloc32(t*t*4); mean(S, P, t, t);

    // Get output
    matmul(P, V, t, t, t, dk, O);

    _aligned_free(K);
    _aligned_free(Q);
    _aligned_free(V);
    _aligned_free(S);
    _aligned_free(P);
}

void MultiHeadAttention(float* dest, float* in, float* mem, float* Wk, float* Wq, float* Wv, float* Wo, int N, int causal) {
    //8 heads, each dk=64, concat back to 512 then wo
    float* concat = (float*)alloc32(N * 512 * 4);
    float* head = (float*)alloc32(N * 64 * 4);

    for (int h = 0; h < 8; h++) {
        SingleHeadAttention(in, mem, Wk + h * 512 * 64, Wq + h * 512 * 64, Wv + h * 512 * 64, 64, N, head, causal);
        for (int tok = 0; tok < N; tok++) {
            memcpy(concat + tok * 512 + h * 64, head + tok * 64, 64 * sizeof(float));
        }
    }

    matmul(concat, Wo, N, 512, 512, 512, dest);

    _aligned_free(concat);
    _aligned_free(head);
}
