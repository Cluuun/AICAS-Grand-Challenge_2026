// Attention-only cos-sim: k_attention (P0 FP32) vs k_attention_fused (A1-P1).

#include "../../host/attn_tb_util.h"
#include "../kernels/attention.hpp"
#include "../kernels/attention_fused.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

static std::vector<float> read_bin(const std::string & path, size_t n) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        fprintf(stderr, "open %s failed\n", path.c_str());
        std::exit(1);
    }
    const size_t bytes = (size_t) f.tellg();
    if (bytes != n * sizeof(float)) {
        fprintf(stderr, "%s size mismatch\n", path.c_str());
        std::exit(2);
    }
    std::vector<float> v(n);
    f.seekg(0);
    f.read((char *) v.data(), bytes);
    return v;
}

static float cos_sim(const float * a, const float * b, size_t n) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n; ++i) {
        dot += (double) a[i] * b[i];
        na  += (double) a[i] * a[i];
        nb  += (double) b[i] * b[i];
    }
    const double d = std::sqrt(na) * std::sqrt(nb);
    return (d > 0) ? (float) (dot / d) : 0.0f;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <mmproj.gguf> [input_f32.bin]\n", argv[0]);
        return 1;
    }
    const char * gguf = argv[1];
    const std::string img_path =
        (argc >= 3) ? argv[2]
                    : "/home/user/workspace/KV260/RTL/vision_head/build/a1_p0_golden/input_f32.bin";

    auto * ctx = vh::vh_load(gguf);
    if (!ctx) return 2;

    constexpr int M = vh::N_POS;
    constexpr int C = vh::C_EMB;
    const size_t  n = (size_t) M * C;

    auto img = read_bin(img_path, (size_t) 3 * 512 * 512);
    std::vector<float> Q(n), K(n), V(n), O_ref(n), O_fused(n);

    vh_attn_tb_qkv_layer0(ctx, img.data(), Q.data(), K.data(), V.data());
    k_attention(Q.data(), K.data(), V.data(), O_ref.data());
    k_attention_fused(Q.data(), K.data(), V.data(), O_fused.data());

    const float cs = cos_sim(O_ref.data(), O_fused.data(), n);
    float max_abs = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        const float d = std::fabs(O_ref[i] - O_fused[i]);
        if (d > max_abs) max_abs = d;
    }

    const bool pass = cs > 0.999f;
    printf("attn fused vs FP32 ref: cos=%.6f max_abs=%.6f PASS=%s\n",
           cs, max_abs, pass ? "YES" : "NO");
    vh::vh_release(ctx);
    return pass ? 0 : 3;
}
