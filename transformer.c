#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include "pretransformer.c"
#include "multiHeadAttention.c"
#include "FeedForwardNetwork.c"
#include "PostSubLayer.c"
#include "postTransformer.c"

#define SEQ 8
#define DMODEL 512
#define DFF 2048
#define NHEADS 8
#define DK 64
#define VOCAB_SIZE 1024

// Encoder
void encoder (int* tokens, int N, float* tok_emb, float* pos_emb,
              float* Wk, float* Wq, float* Wv, float* Wo,
              float* g1, float* b1,
              float* W1, float* ff_b1, float* W2, float* ff_b2,
              float* g2, float* b2,
              float* dest) {

    // N Input (Token IDs) --> (8, 1)
    // Input Embedding LOOKUP (VOCAB_SIZE, 512) --> Each ID gets its own Row --> (8, 512)
    float* x = (float*)alloc32(N * DMODEL * 4);
    lookup(N, tokens, tok_emb, x);

    // Positional Embeddings --> (8, 512) Position Embedding --> (8, 512) + (8, 512) Eltwise --> (8, 512)
    addPosition(pos_emb, x, N, x);

    // Prefill Multi-Head Attention --> 8 Heads, 8 * SingleHeadAttention
    // SingleHead Attention (Wk, Wv, Wq) :: (512, 64) --> Q, K, V :: (8, 64) 
    float* attn = (float*)alloc32(N * DMODEL * 4);
    MultiHeadAttention(attn, x, 0, Wk, Wq, Wv, Wo, N, 0);

    // LayerNorm + Add
    float* n1 = (float*)alloc32(N * DMODEL * 4);
    layerNorm(attn, n1, x, g1, b1);

    // FFN (W1, b1, W2, b2) :: (512, 2048), (1, 2048), (2048, 512), (1, 512)
    float* ff = (float*)alloc32(N * DMODEL * 4);
    FeedForwardNetwork(n1, W1, ff_b1, W2, ff_b2, ff);

    // LayerNorm + Add
    layerNorm(ff, dest, n1, g2, b2);

    _aligned_free(x);
    _aligned_free(attn);
    _aligned_free(n1);
    _aligned_free(ff);
}

// Decoder
void decoder (int* tokens, int N, float* enc_out, float* tok_emb, float* pos_emb,
              float* Wk_m, float* Wq_m, float* Wv_m, float* Wo_m,
              float* g1, float* b1,
              float* Wk_c, float* Wq_c, float* Wv_c, float* Wo_c,
              float* g2, float* b2,
              float* W1, float* ff_b1, float* W2, float* ff_b2,
              float* g3, float* b3,
              float* dest) {

    // N * 512 Input
    // Output Embeddings
    float* x = (float*)alloc32(N * DMODEL * 4);
    lookup(N, tokens, tok_emb, x);

    // Positional Embeddings
    addPosition(pos_emb, x, N, x);

    // Masked Multi-Head Attention
    float* attn1 = (float*)alloc32(N * DMODEL * 4);
    MultiHeadAttention(attn1, x, 0, Wk_m, Wq_m, Wv_m, Wo_m, N, 1);

    // LayerNorm + Add
    float* n1 = (float*)alloc32(N * DMODEL * 4);
    layerNorm(attn1, n1, x, g1, b1);

    // Multi-Head Attention w/ Encoder Outputs as K, V
    float* attn2 = (float*)alloc32(N * DMODEL * 4);
    MultiHeadAttention(attn2, n1, enc_out, Wk_c, Wq_c, Wv_c, Wo_c, N, 0);

    // LayerNorm + Add
    float* n2 = (float*)alloc32(N * DMODEL * 4);
    layerNorm(attn2, n2, n1, g2, b2);

    // FFN 
    float* ff = (float*)alloc32(N * DMODEL * 4);
    FeedForwardNetwork(n2, W1, ff_b1, W2, ff_b2, ff);

    // LayerNorm + Add
    layerNorm(ff, dest, n2, g3, b3);

    _aligned_free(x);
    _aligned_free(attn1);
    _aligned_free(n1);
    _aligned_free(attn2);
    _aligned_free(n2);
    _aligned_free(ff);
}

static float* zalloc(int n) {
    float* p = (float*)alloc32(n * 4);
    memset(p, 0, n * 4);
    return p;
}

static void fill1(float* p, int n) {
    for (int i = 0; i < n; i++) p[i] = 1.0f;
}

int main(void) {
    int tokens[SEQ] = {1, 2, 3, 4, 5, 6, 7, 8};
    int out_tokens[SEQ] = {0, 1, 2, 3, 4, 5, 6, 7};

    float* tok_emb = zalloc(VOCAB_SIZE * DMODEL);
    float* pos_emb = zalloc(SEQ * DMODEL);
    //tiny values so lookup isnt all zeros
    for (int i = 0; i < VOCAB_SIZE * DMODEL; i++) tok_emb[i] = 0.01f * (float)((i % 17) - 8);
    for (int i = 0; i < SEQ * DMODEL; i++) pos_emb[i] = 0.001f * (float)(i % 13);

    //encoder weights
    float* e_Wk = zalloc(NHEADS * DMODEL * DK);
    float* e_Wq = zalloc(NHEADS * DMODEL * DK);
    float* e_Wv = zalloc(NHEADS * DMODEL * DK);
    float* e_Wo = zalloc(DMODEL * DMODEL);
    float* e_g1 = zalloc(DMODEL); fill1(e_g1, DMODEL);
    float* e_b1 = zalloc(DMODEL);
    float* e_W1 = zalloc(DMODEL * DFF);
    float* e_ffb1 = zalloc(DFF);
    float* e_W2 = zalloc(DFF * DMODEL);
    float* e_ffb2 = zalloc(DMODEL);
    float* e_g2 = zalloc(DMODEL); fill1(e_g2, DMODEL);
    float* e_b2 = zalloc(DMODEL);

    //decoder weights
    float* d_Wk_m = zalloc(NHEADS * DMODEL * DK);
    float* d_Wq_m = zalloc(NHEADS * DMODEL * DK);
    float* d_Wv_m = zalloc(NHEADS * DMODEL * DK);
    float* d_Wo_m = zalloc(DMODEL * DMODEL);
    float* d_g1 = zalloc(DMODEL); fill1(d_g1, DMODEL);
    float* d_b1 = zalloc(DMODEL);
    float* d_Wk_c = zalloc(NHEADS * DMODEL * DK);
    float* d_Wq_c = zalloc(NHEADS * DMODEL * DK);
    float* d_Wv_c = zalloc(NHEADS * DMODEL * DK);
    float* d_Wo_c = zalloc(DMODEL * DMODEL);
    float* d_g2 = zalloc(DMODEL); fill1(d_g2, DMODEL);
    float* d_b2 = zalloc(DMODEL);
    float* d_W1 = zalloc(DMODEL * DFF);
    float* d_ffb1 = zalloc(DFF);
    float* d_W2 = zalloc(DFF * DMODEL);
    float* d_ffb2 = zalloc(DMODEL);
    float* d_g3 = zalloc(DMODEL); fill1(d_g3, DMODEL);
    float* d_b3 = zalloc(DMODEL);
    float* vocab = zalloc(DMODEL * VOCAB_SIZE);

    float* enc_out = (float*)alloc32(SEQ * DMODEL * 4);
    float* dec_out = (float*)alloc32(SEQ * DMODEL * 4);
    float* probs = (float*)alloc32(SEQ * VOCAB_SIZE * 4);

    encoder(tokens, SEQ, tok_emb, pos_emb, e_Wk, e_Wq, e_Wv, e_Wo, e_g1, e_b1, e_W1, e_ffb1, e_W2, e_ffb2, e_g2, e_b2, enc_out);
    decoder(out_tokens, SEQ, enc_out, tok_emb, pos_emb, d_Wk_m, d_Wq_m, d_Wv_m, d_Wo_m, d_g1, d_b1, d_Wk_c, d_Wq_c, d_Wv_c, d_Wo_c, d_g2, d_b2, d_W1, d_ffb1, d_W2, d_ffb2, d_g3, d_b3, dec_out);

    // Linearize (8, 512) : (8, 512) @ (512, VOCAB_SIZE) --> (8, VOCAB_SIZE)
    // Softmax (8, VOCAB_SIZE)
    get_out_prob(vocab, dec_out, probs, VOCAB_SIZE);

    printf("p[0][0]=%f p[0][1]=%f p[0][2]=%f\n", probs[0], probs[1], probs[2]);

    _aligned_free(tok_emb); _aligned_free(pos_emb);
    _aligned_free(e_Wk); _aligned_free(e_Wq); _aligned_free(e_Wv); _aligned_free(e_Wo);
    _aligned_free(e_g1); _aligned_free(e_b1); _aligned_free(e_W1); _aligned_free(e_ffb1); _aligned_free(e_W2); _aligned_free(e_ffb2); _aligned_free(e_g2); _aligned_free(e_b2);
    _aligned_free(d_Wk_m); _aligned_free(d_Wq_m); _aligned_free(d_Wv_m); _aligned_free(d_Wo_m);
    _aligned_free(d_g1); _aligned_free(d_b1); _aligned_free(d_Wk_c); _aligned_free(d_Wq_c); _aligned_free(d_Wv_c); _aligned_free(d_Wo_c);
    _aligned_free(d_g2); _aligned_free(d_b2); _aligned_free(d_W1); _aligned_free(d_ffb1); _aligned_free(d_W2); _aligned_free(d_ffb2); _aligned_free(d_g3); _aligned_free(d_b3);
    _aligned_free(vocab); _aligned_free(enc_out); _aligned_free(dec_out); _aligned_free(probs);
    return 0;
}
