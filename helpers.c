#ifndef HELPERS_C
#define HELPERS_C

#include <stdalign.h>
#include <stdio.h> // Standard C Library
#include <malloc.h>
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
    int len = len1 < len2 ? len1 : len2;
    // Loop through
    while (elems < len) {
        //atleast v1 or v2 has some elements left
        int left1 = len1 - elems;
        int left2 = len2 - elems;
        if (elems < len1) { idx1 = ld_simd_reg(v1 + elems, left1); } else { idx1 = _mm256_setzero_ps(); }
        if (elems < len2) { idx2 = ld_simd_reg(v2 + elems, left2); } else { idx2 = _mm256_setzero_ps(); }
        int max = left1 < left2 ? left1 : left2;
        if (max > 8) max = 8;
        add_basic_2(idx1, idx2, addr + elems, max);
        // update elems
        elems += 8;
    }
}

// Multiply 2 Vectors
__m256 fmadd_basic_2(__m256 v1, __m256 v2, __m256 currmax, float* addr, int max) {
    __m256i indices = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    __m256i count_vec = _mm256_set1_epi32(max);
    __m256i mask = _mm256_cmpgt_epi32(count_vec, indices);
    (void)addr; (void)mask;
    currmax = _mm256_fmadd_ps(v1, v2, currmax);
    return currmax;
}

// return 32-byte aligned address 
static void *alloc32(size_t bytes) {
    return _aligned_malloc(bytes, 32);
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
    (void)dlen;
    // Loop through
    while (elems < len1 && elems < len2) {
        // Atleast v1 or v2 has some elements left
        int left1 = len1 - elems;
        int left2 = len2 - elems;
        if (elems < len1) { idx1 = ld_simd_reg(v1 + elems, left1); } else { idx1 = _mm256_setzero_ps(); }
        if (elems < len2) { idx2 = ld_simd_reg(v2 + elems, left2); } else { idx2 = _mm256_setzero_ps(); }
        int max = left1 < left2 ? left1 : left2;
        if (max > 8) max = 8;
        cmax = fmadd_basic_2(idx1, idx2, cmax, dest + elems, max);
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
    while (elems < vlen) {
        // Get vector stored in simd reg
        int left = vlen - elems;
        __m256 vec = ld_simd_reg(vector + elems, left);
        // Multiply this with 'scaled'
        __m256 out = _mm256_mul_ps(vec, scaled);
        store_simd_reg(dest + elems, out, left);
        elems += 8;
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

    (void)col1;

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

//one register: e^x = 2^n * e^r
static inline __m256 exp_ps(__m256 x) {
    __m256 log2e = _mm256_set1_ps(1.4426950408889634f);
    __m256 ln2 = _mm256_set1_ps(0.6931471805599453f);
    //c0 + r*(c1 + r*(c2 + r*(c3 + r*(c4 + r*c5))))
    __m256 c0 = _mm256_set1_ps(1.0f);
    __m256 c1 = _mm256_set1_ps(1.0f);
    __m256 c2 = _mm256_set1_ps(0.5f);
    __m256 c3 = _mm256_set1_ps(0.1666666716f);
    __m256 c4 = _mm256_set1_ps(0.04166666791f);
    __m256 c5 = _mm256_set1_ps(0.00833333377f);

    //clamp so 2^n stays in float range
    x = _mm256_min_ps(x, _mm256_set1_ps(88.0f));
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.0f));

    //multiply by log2e | eg. 3.2 --> 4.61
    __m256 n = _mm256_mul_ps(x, log2e);
    n = _mm256_round_ps(n, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    //remainder | eg. 3.2 - (5*ln2) = -0.2657
    __m256 r = _mm256_fnmadd_ps(n, ln2, x);

    //evaluate e^r with nested fmadds
    __m256 p = c5;
    p = _mm256_fmadd_ps(p, r, c4);
    p = _mm256_fmadd_ps(p, r, c3);
    p = _mm256_fmadd_ps(p, r, c2);
    p = _mm256_fmadd_ps(p, r, c1);
    p = _mm256_fmadd_ps(p, r, c0);

    //2^n in the exponent bits
    __m256i ni = _mm256_cvtps_epi32(n);
    __m256i pow2n = _mm256_slli_epi32(_mm256_add_epi32(ni, _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(pow2n));
}

void exp_vec (float* in, int n) {
    //loop through, potentially loop '4' simd reg at once.
    int elems = 0;
    while (elems < n) {
        if (n - elems >= 32) {
            __m256 x1 = _mm256_load_ps(in + elems);
            __m256 x2 = _mm256_load_ps(in + elems + 8);
            __m256 x3 = _mm256_load_ps(in + elems + 16);
            __m256 x4 = _mm256_load_ps(in + elems + 24);
            _mm256_store_ps(in + elems, exp_ps(x1));
            _mm256_store_ps(in + elems + 8, exp_ps(x2));
            _mm256_store_ps(in + elems + 16, exp_ps(x3));
            _mm256_store_ps(in + elems + 24, exp_ps(x4));
            elems += 32;
        } else {
            int left = n - elems;
            __m256 x = ld_simd_reg(in + elems, left);
            store_simd_reg(in + elems, exp_ps(x), left);
            elems += 8;
        }
    }
}

void mean (float* in, float* out, int rows, int cols) {
    //loop through, potentially with 4 simd reg
    for (int r = 0; r < rows; r++) {
        float* row = in + r * cols;
        float* dest = out + r * cols;
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        __m256 acc4 = _mm256_setzero_ps();
        int elems = 0;
        while (elems + 32 <= cols) {
            acc1 = _mm256_add_ps(acc1, _mm256_load_ps(row + elems));
            acc2 = _mm256_add_ps(acc2, _mm256_load_ps(row + elems + 8));
            acc3 = _mm256_add_ps(acc3, _mm256_load_ps(row + elems + 16));
            acc4 = _mm256_add_ps(acc4, _mm256_load_ps(row + elems + 24));
            elems += 32;
        }
        while (elems < cols) {
            acc1 = _mm256_add_ps(acc1, ld_simd_reg(row + elems, cols - elems));
            elems += 8;
        }
        //once you get all 4's sum, do reduction on the 4 accs into 1
        acc1 = _mm256_add_ps(acc1, acc2);
        acc3 = _mm256_add_ps(acc3, acc4);
        acc1 = _mm256_add_ps(acc1, acc3);
        //do horizontal reduction into 1 number.
        __m256 sum = hsum256_ps(acc1);
        sum = _mm256_max_ps(sum, _mm256_set1_ps(1e-12f));
        //broadcast this into a simd vector. then divide
        elems = 0;
        while (elems < cols) {
            int left = cols - elems;
            __m256 v = ld_simd_reg(row + elems, left);
            store_simd_reg(dest + elems, _mm256_div_ps(v, sum), left);
            elems += 8;
        }
    }
}

float row_mean(float* row, int n) {
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    __m256 acc4 = _mm256_setzero_ps();
    int elems = 0;
    while (elems + 32 <= n) {
        acc1 = _mm256_add_ps(acc1, _mm256_load_ps(row + elems));
        acc2 = _mm256_add_ps(acc2, _mm256_load_ps(row + elems + 8));
        acc3 = _mm256_add_ps(acc3, _mm256_load_ps(row + elems + 16));
        acc4 = _mm256_add_ps(acc4, _mm256_load_ps(row + elems + 24));
        elems += 32;
    }
    while (elems < n) {
        acc1 = _mm256_add_ps(acc1, ld_simd_reg(row + elems, n - elems));
        elems += 8;
    }
    acc1 = _mm256_add_ps(acc1, acc2);
    acc3 = _mm256_add_ps(acc3, acc4);
    acc1 = _mm256_add_ps(acc1, acc3);
    __m256 sum = hsum256_ps(acc1);
    float tmp[8];
    _mm256_storeu_ps(tmp, sum);
    return tmp[0] / (float)n;
}

void sub_broadcast(float* row, int n, float mu, float* dest) {
    __m256 m = _mm256_set1_ps(mu);
    int elems = 0;
    while (elems < n) {
        int left = n - elems;
        __m256 v = ld_simd_reg(row + elems, left);
        store_simd_reg(dest + elems, _mm256_sub_ps(v, m), left);
        elems += 8;
    }
}

void square(float* in, int n, float* dest) {
    int elems = 0;
    while (elems < n) {
        int left = n - elems;
        __m256 v = ld_simd_reg(in + elems, left);
        store_simd_reg(dest + elems, _mm256_mul_ps(v, v), left);
        elems += 8;
    }
}

float inv_sqrt(float x) {
    __m256 v = _mm256_set1_ps(x);
    __m256 y = _mm256_rsqrt_ps(v);
    //one newton step: y * (1.5 - 0.5 * x * y * y)
    __m256 halfx = _mm256_mul_ps(v, _mm256_set1_ps(0.5f));
    __m256 threehalf = _mm256_set1_ps(1.5f);
    y = _mm256_mul_ps(y, _mm256_fnmadd_ps(halfx, _mm256_mul_ps(y, y), threehalf));
    float tmp[8];
    _mm256_storeu_ps(tmp, y);
    return tmp[0];
}

void scale_mul_add(float* centered, int n, float inv_std, float* gamma, float* beta, float* dest) {
    __m256 s = _mm256_set1_ps(inv_std);
    int elems = 0;
    while (elems < n) {
        int left = n - elems;
        __m256 c = ld_simd_reg(centered + elems, left);
        __m256 g = ld_simd_reg(gamma + elems, left);
        __m256 b = ld_simd_reg(beta + elems, left);
        __m256 y = _mm256_mul_ps(c, s);
        y = _mm256_fmadd_ps(y, g, b);
        store_simd_reg(dest + elems, y, left);
        elems += 8;
    }
}

void relu(float* x, int n) {
    __m256 z = _mm256_setzero_ps();
    int elems = 0;
    while (elems < n) {
        int left = n - elems;
        __m256 v = ld_simd_reg(x + elems, left);
        store_simd_reg(x + elems, _mm256_max_ps(v, z), left);
        elems += 8;
    }
}

void add_bias(float* mat, float* bias, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        add_2_arrays(mat + r * cols, bias, cols, cols, mat + r * cols);
    }
}

void softmax(float* in, int rows, int cols, float* dest) {
    for (int r = 0; r < rows; r++) {
        float* row = in + r * cols;
        float* out = dest + r * cols;
        //row max so exp doesnt blow up
        float mx = row[0];
        int i = 1;
        while (i < cols) {
            if (row[i] > mx) mx = row[i];
            i++;
        }
        sub_broadcast(row, cols, mx, out);
        exp_vec(out, cols);
    }
    mean(dest, dest, rows, cols);
}

#endif

// Compare 2 Vectors (Bitwise Grfeater Than)
