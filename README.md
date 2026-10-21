# AVX-Former

encoder/decoder transformer in C, with the math done in AVX2. i started with the simd helpers in helpers.c and then just kept wiring layers until a forward pass actually ran.

this is windows / msvc. i compile it with:

    cl /O2 /arch:AVX2 transformer.c

transformer.c includes the other .c files so you only need that one command. then run transformer.exe.

files (messy but whatever):
- helpers.c — avx2 kernels (matmul, softmax, etc)
- pretransformer.c — token lookup + positional add
- multiHeadAttention.c
- FeedForwardNetwork.c
- PostSubLayer.c — residual + layernorm
- postTransformer.c — last linear + softmax over vocab
- transformer.c — encoder, decoder, dummy main

weights are dummy. just checking the forward pass works.

Nehal70
