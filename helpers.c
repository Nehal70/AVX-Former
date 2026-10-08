#include <stdalign.h>
#include <stdio.h> // Standard C Library
#include <immintrin.h> // Provides access to compiler intrinsics and vector types for x86 SIMD SSE, AVX, AVX2

// Let's declare 2 Vectors for all ops
float A[8] = {1, 2, 3, 4, 5, 6, 7, 8};
float B[8] = {1, 2, 3, 4, 5, 6, 7, 8};

// Helper to construct an 8-lane mask for 'count' active elements (0 to 8)
static inline __m256i make_mask(int count) {
    __m256i indices = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    __m256i count_vec = _mm256_set1_epi32(count);
    // Lanes where index < count get 0xFFFFFFFF (MSB=1), others get 0
    return _mm256_cmpgt_epi32(count_vec, indices);
}

// Load up to 8 floats; disabled lanes are zeroed
static inline __m256 ld_simd_reg(const float* in, int count) {
    if (count <= 0) return _mm256_setzero_ps();
    if (count >= 8) return _mm256_loadu_ps(in); // Full fast load if 8 elements remain
    __m256i mask = make_mask(count);
    return _mm256_maskload_ps(in, mask);
}

// Store up to 8 floats safely to memory without overwriting past the buffer
static inline void store_simd_reg(float* addr, __m256 val, int count) {
    if (count <= 0) return;
    if (count >= 8) {
        _mm256_storeu_ps(addr, val);
    } else {
        __m256i mask = make_mask(count);
        _mm256_maskstore_ps(addr, mask, val);
    }
}

// Add 2 Vectors - 2 full registers
void add_basic_2(__m256 v1, __m256 v2, float* addr, int max) {
    __m256i indices = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    __m256i count_vec = _mm256_set1_epi32(max);
    __m256i mask = _mm256_cmpgt_epi32(count_vec, indices);
    _mm256_maskstore_ps(addr, mask, _mm256_add_ps(v1, v2));
}

// Add 2 Arrays - Same Size
// v1, v2, addr - All Aligned To 32
void add_2_arrays(float* v1, float* v2, int len1, int len2, float* addr) {
    // Store counter for number of elements
    int elems = 0;
    // Have a default 0
    __m256 idx1 = _mm256_setzero_ps();
    __m256 idx2 = _mm256_setzero_ps();
    // Loop through
    while (elems <= len1 && elems <= len2) {
        // Atleast v1 or v2 has some elements left
        if (elems < len1) { idx1 = ld_simd_reg(v1 + elems, (len1 - elems) % 8); } else { idx1 = _mm256_setzero_ps(); }  
        if (elems < len2) { idx2 = ld_simd_reg(v2 + elems, (len2 - elems) % 8); } else { idx2 = _mm256_setzero_ps(); }
        int max = (len1 - elems) % 8 > (len2 - elems) % 8 ? (len1 - elems) % 8 : (len2 - elems) % 8;
        add_basic_2(idx1, idx2, addr + elems, max);
        // update elems
        elems += 8;
    }
}

// Multiply 2 Vectors
void fmadd_basic_2(__m256 v1, __m256 v2, __m256 currmax, float* addr, int max) {
    __m256i indices = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    __m256i count_vec = _mm256_set1_epi32(max);
    __m256i mask = _mm256_cmpgt_epi32(count_vec, indices);
    currmax = _mm256_fmadd_ps(v1, v2, currmax);
}

// Horizontal Sum
__m256 hsum256_ps(__m256 v) {
    // 1. Fold 8 -> 4
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    __m128 sum4  = _mm_add_ps(vlow, vhigh);

    // 2. Fold 4 -> 2
    __m128 vhigh2 = _mm_movehl_ps(sum4, sum4);
    __m128 sum2   = _mm_add_ps(sum4, vhigh2);

    // 3. Fold 2 -> 1 (sum is in lowest 32-bit float of sum1)
    __m128 vhigh1 = _mm_shuffle_ps(sum2, sum2, 0x55);
    __m128 sum1   = _mm_add_ps(sum2, vhigh1);

    // 4. Broadcast the scalar float to all 8 lanes of __m256
    return _mm256_broadcastss_ps(sum1);
}

// Multiply 2 Vectors (Dot)
void dot_product(float* v1, float* v2, float* dest, int len1, int len2, int dlen) {
    // Store counter for number of elements
    int elems = 0;
    // Have a default 0
    __m256 idx1 = _mm256_setzero_ps();
    __m256 idx2 = _mm256_setzero_ps();
    __m256 cmax = _mm256_setzero_ps();
    // Loop through
    while (elems <= len1 && elems <= len2) {
        // Atleast v1 or v2 has some elements left
        if (elems < len1) { idx1 = ld_simd_reg(v1 + elems, (len1 - elems) % 8); } else { idx1 = _mm256_setzero_ps(); }  
        if (elems < len2) { idx2 = ld_simd_reg(v2 + elems, (len2 - elems) % 8); } else { idx2 = _mm256_setzero_ps(); }
        int max = (len1 - elems) % 8 > (len2 - elems) % 8 ? (len1 - elems) % 8 : (len2 - elems) % 8;
        fmadd_basic_2(idx1, idx2, cmax, dest + elems, max);
        // update elems
        elems += 8;
    }
    // Now horizontally add up cmax
    __m256 res = hsum256_ps(cmax);
    store_simd_reg(dest, res, 1);
}

void sdot_product(float scalar, float* vector, int vlen, float* dest) {
    // Scaled will always be of this value
    __m256 scaled = _mm256_set1_ps(scalar);
    // Counter for how many done in vector
    int elems = 0;
    // Now we need to read 'vector' 
    while (vlen > 0) {
        // Get vector stored in simd reg
        __m256 vec = ld_simd_reg(vector + elems, vlen - elems);
        // Multiply this with 'scaled'

    }

}

void matmul (float* m1, float* m2, int row1, int col1, int row2, int col2, float* dest) {

    // We know starting point of both matrices in memory, how many rows and columns each of them have,
    // and where to store the output in memory.

    // Constraints, assume B's row_size is a multiple of 32

    // Loop through each output row, then each tile
    for (int row = 0; row < row1; row++) {
        int elems = 0;
        // Get 4 SIMD Reg, each with 8 Elems from 'B' Matrix
        while (elems < col2) {
            // Make SIMD Register for each tile in this section of the output row
            __m256 tile1 = _mm256_setzero_ps();
            __m256 tile2 = _mm256_setzero_ps();
            __m256 tile3 = _mm256_setzero_ps();
            __m256 tile4 = _mm256_setzero_ps();
            // Output Tiles are fixed, now let's iterate through all computations we need to do. 
            for (int spatial = 0; spatial < row2; spatial++) {
                // Get Scalar from A
                __m256 scaled = _mm256_broadcast_ss(m1 + (row*col1) + spatial);
                // Get B values
                __m256 b1 = _mm256_load_ps((m2 + (spatial*col2+elems) + 0));
                __m256 b2 = _mm256_load_ps((m2 + (spatial*col2+elems) + 8));
                __m256 b3 = _mm256_load_ps((m2 + (spatial*col2+elems) + 16));
                __m256 b4 = _mm256_load_ps((m2 + (spatial*col2+elems) + 24));
                // Fmadd A, B onto tiles
                tile1 = _mm256_fmadd_ps(scaled, b1, tile1);
                tile2 = _mm256_fmadd_ps(scaled, b2, tile2);
                tile3 = _mm256_fmadd_ps(scaled, b3, tile3);
                tile4 = _mm256_fmadd_ps(scaled, b4, tile4);
            }
             // 4 accumulators have 32 real values.
            _mm256_store_ps(dest+(row*col2)+elems, tile1);
            _mm256_store_ps(dest+(row*col2)+elems+8, tile2);
            _mm256_store_ps(dest+(row*col2)+elems+16, tile3);
            _mm256_store_ps(dest+(row*col2)+elems+24, tile4);
            // Update Elems
            elems += 32;
        }
    }
}

void transpose_matmul (float* m1, float* m2, int row1, int col1, int row2, int col2, float* dest) {

    // We know starting point of both matrices in memory, how many rows and columns each of them have,
    // and where to store the output in memory.

    // Constraints, assume B's row_size is a multiple of 32

    // Loop through each output row, then each tile
    for (int row = 0; row < row1; row++) {
        for (int krow = 0; krow < row2; krow++) {
            // Track how many elements done in dotproduct
            int elems = 0;
            // Create 4 accs for dot product
            __m256 tile1 = _mm256_setzero_ps();
            __m256 tile2 = _mm256_setzero_ps();
            __m256 tile3 = _mm256_setzero_ps();
            __m256 tile4 = _mm256_setzero_ps();
            // Get 4 SIMD Reg, each with 8 Elems from 'B' Matrix
            while (elems < col2) {
                // Get 4 vectors from A
                __m256 a1 = _mm256_load_ps((m1 + (row*col2+elems) + 0));
                __m256 a2 = _mm256_load_ps((m1 + (row*col2+elems) + 8));
                __m256 a3 = _mm256_load_ps((m1 + (row*col2+elems) + 16));
                __m256 a4 = _mm256_load_ps((m1 + (row*col2+elems) + 24));
                // Get 4 vectors from B
                __m256 b1 = _mm256_load_ps((m2 + (krow*col2+elems) + 0));
                __m256 b2 = _mm256_load_ps((m2 + (krow*col2+elems) + 8));
                __m256 b3 = _mm256_load_ps((m2 + (krow*col2+elems) + 16));
                __m256 b4 = _mm256_load_ps((m2 + (krow*col2+elems) + 24));
                // Fmadd A, B onto tiles
                tile1 = _mm256_fmadd_ps(a1, b1, tile1);
                tile2 = _mm256_fmadd_ps(a2, b2, tile2);
                tile3 = _mm256_fmadd_ps(a3, b3, tile3);
                tile4 = _mm256_fmadd_ps(a4, b4, tile4);
                // Update Elems
                elems += 32;
            }
            // Sum up 4 accs + final horizontal reduction
            tile1 = _mm256_add_ps(tile1, tile2);
            tile3 = _mm256_add_ps(tile3, tile4);
            tile2 = _mm256_add_ps(tile1, tile3);
            tile4 = hsum256_ps(tile2);
            // Store this
            __m256i mas = make_mask(1);
            _mm256_maskstore_ps(dest + (row*row2) + krow, mas, tile4); 
        }
    }
}

void exp (float* in) {
    // Loop through, potentially loop '4' SIMD Reg at once. 

    // Multiply Each SIMD Reg by a SIMD Reg that has log2e broadcasted | Eg. 3.2 --> 4.61
    // Create 2 SIMD reg per SIMD reg --> Whole Number, Remainder | Eg. 4.61 --> 5
    // For Remainder | Eg. 3.2 - (5*Ln2) = -0.2657
    // Thus e^3.2 = 2^5*e^-0.2657
    // Evaluate e^-0.2657 --> 0.7666 using fmadds
    // FMADDS: Essentially, make it a polynomial and factor into c + r*(c1+r(c2+r)) and so on until 5. 
    // Because each nesting level matches the exact form $A \cdot B + C$, every single step maps directly to
    // one _mm256_fmadd_ps(prev_accumulator, r, next_coefficient)

}

void mean (float* in, float* out) {
    // Loop through, potentially with 4 SIMD reg
    // Once you get all 4's sum, do reduction on the 4 ACCs into 1
    // Do horizontal reduction into 1 number. 
    // Broadcast this into a SIMD vector. Then, loop through all 4 SIMD vectors from beginning again and divide them
    // Do this for all rows. 
}

void SingleHeadAttention(float* X, float* Wk, float* Wq, float* Wv, int dk, int t, float* O) {

    // Get K, Q, V
    float* K = (float*) aligned_alloc(32, t*dk*4); matmul(X, Wk, t, dk, dk, dk, K);
    float* Q = (float*) aligned_alloc(32, t*dk*4); matmul(X, Wq, t, dk, dk, dk, Q);
    float* V = (float*) aligned_alloc(32, t*dk*4); matmul(X, Wv, t, dk, dk, dk, V);

    // Get S
    float* S = (float*) aligned_alloc(32, t*t*4); transpose_matmul(Q, K, t, dk, t, dk, S);

    // Get P
    exp(S); 
    float* P = (float*) aligned_alloc(32, t*t*4); mean(S, P);

    // Get output
    matmul(P, V, t, t, t, dk, O);
}


// Compare 2 Vectors (Bitwise Grfeater Than)