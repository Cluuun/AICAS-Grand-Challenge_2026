#include <hls_math.h>

extern "C" {
void rmsnorm_accel(float *in_data, float *weights, float *out_data, int size, float epsilon) {

    #pragma HLS INTERFACE m_axi port=in_data  bundle=gmem0 offset=slave depth=960
    #pragma HLS INTERFACE m_axi port=weights  bundle=gmem1 offset=slave depth=960
    #pragma HLS INTERFACE m_axi port=out_data bundle=gmem2 offset=slave depth=960

    #pragma HLS INTERFACE s_axilite port=in_data  bundle=control
    #pragma HLS INTERFACE s_axilite port=weights  bundle=control
    #pragma HLS INTERFACE s_axilite port=out_data bundle=control
    #pragma HLS INTERFACE s_axilite port=size     bundle=control
    #pragma HLS INTERFACE s_axilite port=epsilon  bundle=control
    #pragma HLS INTERFACE s_axilite port=return   bundle=control

    const int MAX_SIZE = 1024;
    float local_in[MAX_SIZE];
    float local_weights[MAX_SIZE];

    // --- Step 1: DDR read → BRAM, compute sum of squares ---
    float partial_sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    #pragma HLS ARRAY_PARTITION variable=partial_sums complete

    for (int i = 0; i < size; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=960 max=960
        float val = in_data[i];
        local_in[i] = val;
        local_weights[i] = weights[i];
        partial_sums[i & 3] += val * val;
    }
    float sum_squares = partial_sums[0] + partial_sums[1] + partial_sums[2] + partial_sums[3];

    // --- Step 2: compute 1 / rms ---
    float inv_rms = hls::rsqrtf(sum_squares / (float)size + epsilon);

    // --- Step 3: BRAM → DDR ---
    for (int i = 0; i < size; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=960 max=960
        out_data[i] = local_in[i] * inv_rms * local_weights[i];
    }
}
}
