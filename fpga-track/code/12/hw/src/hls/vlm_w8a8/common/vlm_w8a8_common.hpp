#ifndef VLM_W8A8_COMMON_HPP
#define VLM_W8A8_COMMON_HPP

#include "vlm_w8a8_abi_v10.h"

#include <ap_int.h>
#include <hls_half.h>
#include <hls_math.h>
#include <stdint.h>
#include <string.h>

typedef ap_uint<128> vlm_w8a8_axi_t;

struct vlm_w8a8_q8_block_raw {
    uint16_t d_bits;
    int8_t qs[VLM_W8A8_QK];
};

static inline uint8_t vlm_w8a8_load_u8(const vlm_w8a8_axi_t * arena, uint64_t byte_offset) {
    #pragma HLS INLINE
    const uint64_t word_idx = byte_offset / 16u;
    const int byte_lane = byte_offset % 16u;
    const vlm_w8a8_axi_t word = arena[word_idx];
    switch (byte_lane) {
        case 0:  return static_cast<uint8_t>(word.range(7, 0));
        case 1:  return static_cast<uint8_t>(word.range(15, 8));
        case 2:  return static_cast<uint8_t>(word.range(23, 16));
        case 3:  return static_cast<uint8_t>(word.range(31, 24));
        case 4:  return static_cast<uint8_t>(word.range(39, 32));
        case 5:  return static_cast<uint8_t>(word.range(47, 40));
        case 6:  return static_cast<uint8_t>(word.range(55, 48));
        case 7:  return static_cast<uint8_t>(word.range(63, 56));
        case 8:  return static_cast<uint8_t>(word.range(71, 64));
        case 9:  return static_cast<uint8_t>(word.range(79, 72));
        case 10: return static_cast<uint8_t>(word.range(87, 80));
        case 11: return static_cast<uint8_t>(word.range(95, 88));
        case 12: return static_cast<uint8_t>(word.range(103, 96));
        case 13: return static_cast<uint8_t>(word.range(111, 104));
        case 14: return static_cast<uint8_t>(word.range(119, 112));
        default: return static_cast<uint8_t>(word.range(127, 120));
    }
}

static inline void vlm_w8a8_store_u8(vlm_w8a8_axi_t * arena, uint64_t byte_offset, uint8_t value) {
    #pragma HLS INLINE
    const uint64_t word_idx = byte_offset / 16u;
    const int byte_lane = byte_offset % 16u;
    vlm_w8a8_axi_t word = arena[word_idx];
    switch (byte_lane) {
        case 0:  word.range(7, 0) = value; break;
        case 1:  word.range(15, 8) = value; break;
        case 2:  word.range(23, 16) = value; break;
        case 3:  word.range(31, 24) = value; break;
        case 4:  word.range(39, 32) = value; break;
        case 5:  word.range(47, 40) = value; break;
        case 6:  word.range(55, 48) = value; break;
        case 7:  word.range(63, 56) = value; break;
        case 8:  word.range(71, 64) = value; break;
        case 9:  word.range(79, 72) = value; break;
        case 10: word.range(87, 80) = value; break;
        case 11: word.range(95, 88) = value; break;
        case 12: word.range(103, 96) = value; break;
        case 13: word.range(111, 104) = value; break;
        case 14: word.range(119, 112) = value; break;
        default: word.range(127, 120) = value; break;
    }
    arena[word_idx] = word;
}

template<typename T>
static inline void vlm_w8a8_load_obj(const vlm_w8a8_axi_t * arena, uint64_t byte_offset, T * out) {
    #pragma HLS INLINE
    uint8_t * dst = reinterpret_cast<uint8_t *>(out);
    for (size_t i = 0; i < sizeof(T); ++i) {
        #pragma HLS PIPELINE II=1
        dst[i] = vlm_w8a8_load_u8(arena, byte_offset + i);
    }
}

template<typename T>
static inline void vlm_w8a8_store_obj(vlm_w8a8_axi_t * arena, uint64_t byte_offset, const T & value) {
    #pragma HLS INLINE
    const uint8_t * src = reinterpret_cast<const uint8_t *>(&value);
    for (size_t i = 0; i < sizeof(T); ++i) {
        #pragma HLS PIPELINE II=1
        vlm_w8a8_store_u8(arena, byte_offset + i, src[i]);
    }
}

static inline float vlm_w8a8_fp16_bits_to_f32(uint16_t bits) {
    #pragma HLS INLINE
    half h;
    memcpy(&h, &bits, sizeof(bits));
    return static_cast<float>(h);
}

static inline uint16_t vlm_w8a8_f32_to_fp16_bits(float value) {
    #pragma HLS INLINE
    const half h = static_cast<half>(value);
    uint16_t bits = 0;
    memcpy(&bits, &h, sizeof(bits));
    return bits;
}

static inline uint32_t vlm_w8a8_f32_bits(float value) {
    #pragma HLS INLINE
    union {
        float f;
        uint32_t u;
    } cvt;
    cvt.f = value;
    return cvt.u;
}

static inline float vlm_w8a8_bits_to_f32(uint32_t bits) {
    #pragma HLS INLINE
    union {
        float f;
        uint32_t u;
    } cvt;
    cvt.u = bits;
    return cvt.f;
}

static const float vlm_w8a8_silu_lut[513] = {
    -0.00268280f, -0.00275712f, -0.00283345f, -0.00291186f, -0.00299238f, -0.00307508f, -0.00316001f, -0.00324724f,
    -0.00333682f, -0.00342881f, -0.00352328f, -0.00362029f, -0.00371991f, -0.00382220f, -0.00392724f, -0.00403510f,
    -0.00414584f, -0.00425955f, -0.00437629f, -0.00449615f, -0.00461921f, -0.00474556f, -0.00487526f, -0.00500842f,
    -0.00514511f, -0.00528543f, -0.00542948f, -0.00557734f, -0.00572911f, -0.00588490f, -0.00604480f, -0.00620892f,
    -0.00637736f, -0.00655023f, -0.00672765f, -0.00690973f, -0.00709659f, -0.00728834f, -0.00748511f, -0.00768701f,
    -0.00789419f, -0.00810677f, -0.00832489f, -0.00854868f, -0.00877827f, -0.00901383f, -0.00925548f, -0.00950338f,
    -0.00975768f, -0.01001855f, -0.01028613f, -0.01056059f, -0.01084211f, -0.01113084f, -0.01142696f, -0.01173065f,
    -0.01204209f, -0.01236147f, -0.01268898f, -0.01302480f, -0.01336914f, -0.01372219f, -0.01408417f, -0.01445528f,
    -0.01483574f, -0.01522576f, -0.01562556f, -0.01603538f, -0.01645545f, -0.01688599f, -0.01732725f, -0.01777948f,
    -0.01824293f, -0.01871784f, -0.01920448f, -0.01970312f, -0.02021401f, -0.02073745f, -0.02127369f, -0.02182303f,
    -0.02238576f, -0.02296216f, -0.02355254f, -0.02415720f, -0.02477645f, -0.02541060f, -0.02605997f, -0.02672488f,
    -0.02740566f, -0.02810264f, -0.02881617f, -0.02954658f, -0.03029423f, -0.03105946f, -0.03184263f, -0.03264411f,
    -0.03346425f, -0.03430345f, -0.03516205f, -0.03604046f, -0.03693905f, -0.03785821f, -0.03879833f, -0.03975982f,
    -0.04074306f, -0.04174846f, -0.04277643f, -0.04382737f, -0.04490170f, -0.04599984f, -0.04712220f, -0.04826919f,
    -0.04944124f, -0.05063877f, -0.05186221f, -0.05311197f, -0.05438847f, -0.05569215f, -0.05702342f, -0.05838270f,
    -0.05977041f, -0.06118698f, -0.06263280f, -0.06410829f, -0.06561387f, -0.06714991f, -0.06871684f, -0.07031502f,
    -0.07194484f, -0.07360668f, -0.07530089f, -0.07702784f, -0.07878787f, -0.08058130f, -0.08240846f, -0.08426964f,
    -0.08616514f, -0.08809522f, -0.09006014f, -0.09206014f, -0.09409542f, -0.09616617f, -0.09827257f, -0.10041474f,
    -0.10259281f, -0.10480685f, -0.10705692f, -0.10934303f, -0.11166518f, -0.11402330f, -0.11641730f, -0.11884706f,
    -0.12131238f, -0.12381306f, -0.12634880f, -0.12891929f, -0.13152415f, -0.13416294f, -0.13683517f, -0.13954027f,
    -0.14227762f, -0.14504653f, -0.14784623f, -0.15067589f, -0.15353457f, -0.15642129f, -0.15933495f, -0.16227437f,
    -0.16523829f, -0.16822534f, -0.17123405f, -0.17426285f, -0.17731006f, -0.18037390f, -0.18345245f, -0.18654368f,
    -0.18964545f, -0.19275547f, -0.19587133f, -0.19899048f, -0.20211023f, -0.20522775f, -0.20834005f, -0.21144400f,
    -0.21453630f, -0.21761350f, -0.22067199f, -0.22370798f, -0.22671751f, -0.22969646f, -0.23264053f, -0.23554521f,
    -0.23840584f, -0.24121757f, -0.24397536f, -0.24667396f, -0.24930795f, -0.25187172f, -0.25435945f, -0.25676514f,
    -0.25908260f, -0.26130542f, -0.26342701f, -0.26544062f, -0.26733925f, -0.26911577f, -0.27076282f, -0.27227289f,
    -0.27363829f, -0.27485113f, -0.27590338f, -0.27678685f, -0.27749318f, -0.27801388f, -0.27834031f, -0.27846370f,
    -0.27837517f, -0.27806574f, -0.27752630f, -0.27674768f, -0.27572064f, -0.27443586f, -0.27288399f, -0.27105564f,
    -0.26894142f, -0.26653192f, -0.26381776f, -0.26078958f, -0.25743810f, -0.25375408f, -0.24972839f, -0.24535199f,
    -0.24061598f, -0.23551159f, -0.23003024f, -0.22416352f, -0.21790321f, -0.21124134f, -0.20417018f, -0.19668224f,
    -0.18877033f, -0.18042757f, -0.17164736f, -0.16242348f, -0.15275003f, -0.14262148f, -0.13203270f, -0.12097894f,
    -0.10945587f, -0.09745959f, -0.08498660f, -0.07203387f, -0.05859883f, -0.04467934f, -0.03027376f, -0.01538088f,
    0.00000000f, 0.01586912f, 0.03222624f, 0.04907066f, 0.06640117f, 0.08421613f, 0.10251340f, 0.12129041f,
    0.14054413f, 0.16027106f, 0.18046730f, 0.20112852f, 0.22224997f, 0.24382652f, 0.26585264f, 0.28832243f,
    0.31122967f, 0.33456776f, 0.35832982f, 0.38250866f, 0.40709679f, 0.43208648f, 0.45746976f, 0.48323841f,
    0.50938402f, 0.53589801f, 0.56277161f, 0.58999592f, 0.61756190f, 0.64546042f, 0.67368224f, 0.70221808f,
    0.73105858f, 0.76019436f, 0.78961601f, 0.81931414f, 0.84927936f, 0.87950232f, 0.90997370f, 0.94068426f,
    0.97162483f, 1.00278630f, 1.03415969f, 1.06573612f, 1.09750682f, 1.12946315f, 1.16159662f, 1.19389887f,
    1.22636171f, 1.25897711f, 1.29173718f, 1.32463423f, 1.35766075f, 1.39080938f, 1.42407299f, 1.45744458f,
    1.49091740f, 1.52448486f, 1.55814055f, 1.59187828f, 1.62569205f, 1.65957604f, 1.69352464f, 1.72753243f,
    1.76159416f, 1.79570479f, 1.82985947f, 1.86405354f, 1.89828249f, 1.93254202f, 1.96682801f, 2.00113650f,
    2.03546370f, 2.06980600f, 2.10415995f, 2.13852225f, 2.17288977f, 2.20725952f, 2.24162867f, 2.27599453f,
    2.31035455f, 2.34470632f, 2.37904755f, 2.41337610f, 2.44768994f, 2.48198715f, 2.51626595f, 2.55052466f,
    2.58476171f, 2.61897563f, 2.65316505f, 2.68732871f, 2.72146543f, 2.75557411f, 2.78965377f, 2.82370347f,
    2.85772238f, 2.89170973f, 2.92566483f, 2.95958706f, 2.99347585f, 3.02733071f, 3.06115120f, 3.09493694f,
    3.12868762f, 3.16240294f, 3.19608270f, 3.22972670f, 3.26333482f, 3.29690697f, 3.33044308f, 3.36394315f,
    3.39740719f, 3.43083526f, 3.46422743f, 3.49758383f, 3.53090458f, 3.56418986f, 3.59743986f, 3.63065478f,
    3.66383486f, 3.69698036f, 3.73009154f, 3.76316870f, 3.79621213f, 3.82922216f, 3.86219911f, 3.89514332f,
    3.92805516f, 3.96093498f, 3.99378316f, 4.02660009f, 4.05938613f, 4.09214171f, 4.12486720f, 4.15756302f,
    4.19022959f, 4.22286730f, 4.25547658f, 4.28805785f, 4.32061153f, 4.35313803f, 4.38563779f, 4.41811123f,
    4.45055876f, 4.48298081f, 4.51537780f, 4.54775016f, 4.58009830f, 4.61242263f, 4.64472357f, 4.67700154f,
    4.70925694f, 4.74149018f, 4.77370167f, 4.80589179f, 4.83806095f, 4.87020954f, 4.90233795f, 4.93444655f,
    4.96653575f, 4.99860589f, 5.03065737f, 5.06269054f, 5.09470577f, 5.12670342f, 5.15868383f, 5.19064736f,
    5.22259434f, 5.25452512f, 5.28644003f, 5.31833940f, 5.35022355f, 5.38209280f, 5.41394746f, 5.44578784f,
    5.47761424f, 5.50942697f, 5.54122631f, 5.57301255f, 5.60478599f, 5.63654688f, 5.66829552f, 5.70003216f,
    5.73175707f, 5.76347052f, 5.79517275f, 5.82686401f, 5.85854455f, 5.89021462f, 5.92187444f, 5.95352424f,
    5.98516426f, 6.01679472f, 6.04841583f, 6.08002781f, 6.11163086f, 6.14322520f, 6.17481102f, 6.20638853f,
    6.23795791f, 6.26951935f, 6.30107304f, 6.33261916f, 6.36415789f, 6.39568941f, 6.42721387f, 6.45873145f,
    6.49024232f, 6.52174662f, 6.55324452f, 6.58473617f, 6.61622173f, 6.64770132f, 6.67917511f, 6.71064323f,
    6.74210581f, 6.77356299f, 6.80501489f, 6.83646166f, 6.86790341f, 6.89934027f, 6.93077235f, 6.96219977f,
    6.99362264f, 7.02504108f, 7.05645520f, 7.08786510f, 7.11927089f, 7.15067266f, 7.18207052f, 7.21346457f,
    7.24485489f, 7.27624158f, 7.30762474f, 7.33900444f, 7.37038079f, 7.40175385f, 7.43312371f, 7.46449045f,
    7.49585416f, 7.52721490f, 7.55857276f, 7.58992780f, 7.62128009f, 7.65262971f, 7.68397672f, 7.71532119f,
    7.74666318f, 7.77800276f, 7.80933999f, 7.84067492f, 7.87200762f, 7.90333814f, 7.93466655f, 7.96599288f,
    7.99731720f,
};

static inline float vlm_w8a8_silu(float x) {
    #pragma HLS INLINE
    if (x <= -8.0f) {
        return vlm_w8a8_silu_lut[0];
    }
    if (x >= 8.0f) {
        return vlm_w8a8_silu_lut[512];
    }
    const float idx_f = (x + 8.0f) * 32.0f;
    const uint32_t idx = static_cast<uint32_t>(idx_f);
    const float frac = idx_f - static_cast<float>(idx);
    const float y0 = vlm_w8a8_silu_lut[idx];
    const float y1 = vlm_w8a8_silu_lut[idx + 1u];
    return y0 + frac * (y1 - y0);
}

static inline float vlm_w8a8_load_a8(const vlm_w8a8_axi_t * arena, uint64_t q_base, uint64_t scale_base, int row, int col, int row_stride) {
    #pragma HLS INLINE
    const int8_t q = static_cast<int8_t>(vlm_w8a8_load_u8(arena, q_base + static_cast<uint64_t>(row * row_stride + col)));
    uint16_t scale_bits = 0;
    vlm_w8a8_load_obj(arena, scale_base + static_cast<uint64_t>(row * sizeof(uint16_t)), &scale_bits);
    return vlm_w8a8_fp16_bits_to_f32(scale_bits) * static_cast<float>(q);
}

static inline vlm_w8a8_q8_block_raw vlm_w8a8_load_q8_block(const vlm_w8a8_axi_t * arena, uint64_t byte_offset) {
    #pragma HLS INLINE
    vlm_w8a8_q8_block_raw block;
    vlm_w8a8_load_obj(arena, byte_offset, &block);
    return block;
}


static inline void vlm_w8a8_dsp_simd_mul_2x_i8(
        int8_t a0,
        int8_t b0,
        int8_t a1,
        int8_t b1,
        int32_t * prod0,
        int32_t * prod1) {
    #pragma HLS INLINE
    int16_t mul0 = static_cast<int16_t>(a0) * static_cast<int16_t>(b0);
    int16_t mul1 = static_cast<int16_t>(a1) * static_cast<int16_t>(b1);
    #pragma HLS BIND_OP variable=mul0 op=mul impl=dsp
    #pragma HLS BIND_OP variable=mul1 op=mul impl=dsp
    *prod0 = static_cast<int32_t>(mul0);
    *prod1 = static_cast<int32_t>(mul1);
}

static inline ap_uint<8> vlm_w8a8_abs_i8_u8(int8_t value) {
    #pragma HLS INLINE
    const ap_int<9> widened = static_cast<ap_int<9> >(value);
    ap_int<9> mag = widened;
    if (widened < 0) {
        mag = -widened;
    }
    return static_cast<ap_uint<8> >(mag);
}

static inline void vlm_w8a8_dsp_mul_2row_i8(
        int8_t a_row0,
        int8_t a_row1,
        int8_t b,
        ap_int<16> * prod_row0,
        ap_int<16> * prod_row1) {
    #pragma HLS INLINE
    const ap_int<9> a0_s = static_cast<ap_int<9> >(a_row0);
    const ap_int<9> a1_s = static_cast<ap_int<9> >(a_row1);
    const ap_int<9> b_s = static_cast<ap_int<9> >(b);
    const ap_uint<8> a0_u = static_cast<ap_uint<8> >(a0_s + 128);
    const ap_uint<8> a1_u = static_cast<ap_uint<8> >(a1_s + 128);
    const ap_uint<8> b_u = static_cast<ap_uint<8> >(b_s + 128);

    ap_uint<26> packed_a = 0;
    packed_a.range(7, 0) = a0_u;
    packed_a.range(25, 18) = a1_u;

    ap_uint<34> packed_prod = packed_a * b_u;
    #pragma HLS BIND_OP variable=packed_prod op=mul impl=dsp

    const ap_int<18> raw0 = static_cast<ap_int<18> >(packed_prod.range(15, 0));
    const ap_int<18> raw1 = static_cast<ap_int<18> >(packed_prod.range(33, 18));
    const ap_int<18> correction_b = static_cast<ap_int<18> >(b_s) << 7;
    const ap_int<18> signed0 = raw0 - (static_cast<ap_int<18> >(a0_s) << 7) - correction_b - 16384;
    const ap_int<18> signed1 = raw1 - (static_cast<ap_int<18> >(a1_s) << 7) - correction_b - 16384;
    *prod_row0 = static_cast<ap_int<16> >(signed0);
    *prod_row1 = static_cast<ap_int<16> >(signed1);
}

static inline void vlm_i8x2_dot4_dsp48e2(
        const int8_t a_row0[4],
        const int8_t a_row1[4],
        const int8_t b[4],
        ap_int<20> * partial_row0,
        ap_int<20> * partial_row1) {
    #pragma HLS INLINE

    ap_uint<18> raw_sum0 = 0;
    ap_uint<18> raw_sum1 = 0;
    ap_int<11> sum_a0 = 0;
    ap_int<11> sum_a1 = 0;
    ap_int<11> sum_b = 0;

    for (uint32_t ko = 0; ko < 4u; ++ko) {
        #pragma HLS UNROLL
        const ap_int<9> a0_s = static_cast<ap_int<9> >(a_row0[ko]);
        const ap_int<9> a1_s = static_cast<ap_int<9> >(a_row1[ko]);
        const ap_int<9> b_s = static_cast<ap_int<9> >(b[ko]);
        const ap_uint<8> a0_u = static_cast<ap_uint<8> >(a0_s + 128);
        const ap_uint<8> a1_u = static_cast<ap_uint<8> >(a1_s + 128);
        const ap_uint<8> b_u = static_cast<ap_uint<8> >(b_s + 128);

        ap_uint<26> packed_a = 0;
        packed_a.range(7, 0) = a0_u;
        packed_a.range(25, 18) = a1_u;

        ap_uint<34> packed_prod = packed_a * b_u;
        #pragma HLS BIND_OP variable=packed_prod op=mul impl=dsp

        raw_sum0 += static_cast<ap_uint<16> >(packed_prod.range(15, 0));
        raw_sum1 += static_cast<ap_uint<16> >(packed_prod.range(33, 18));
        sum_a0 += a0_s;
        sum_a1 += a1_s;
        sum_b += b_s;
    }

    const ap_int<20> correction_b = static_cast<ap_int<20> >(sum_b) << 7;
    *partial_row0 = static_cast<ap_int<20> >(raw_sum0) -
            (static_cast<ap_int<20> >(sum_a0) << 7) - correction_b - 65536;
    *partial_row1 = static_cast<ap_int<20> >(raw_sum1) -
            (static_cast<ap_int<20> >(sum_a1) << 7) - correction_b - 65536;
}

static inline void vlm_i8x2_dot4_u8packed_raw(
        const int8_t a_row0[4],
        const int8_t a_row1[4],
        const int8_t b[4],
        ap_uint<18> * raw_row0,
        ap_uint<18> * raw_row1) {
    #pragma HLS INLINE

    ap_uint<18> raw_sum0 = 0;
    ap_uint<18> raw_sum1 = 0;

    for (uint32_t ko = 0; ko < 4u; ++ko) {
        #pragma HLS UNROLL
        const ap_int<9> a0_s = static_cast<ap_int<9> >(a_row0[ko]);
        const ap_int<9> a1_s = static_cast<ap_int<9> >(a_row1[ko]);
        const ap_int<9> b_s = static_cast<ap_int<9> >(b[ko]);
        const ap_uint<8> a0_u = static_cast<ap_uint<8> >(a0_s + 128);
        const ap_uint<8> a1_u = static_cast<ap_uint<8> >(a1_s + 128);
        const ap_uint<8> b_u = static_cast<ap_uint<8> >(b_s + 128);

        ap_uint<26> packed_a = 0;
        packed_a.range(7, 0) = a0_u;
        packed_a.range(25, 18) = a1_u;

        ap_uint<34> packed_prod = packed_a * b_u;
        #pragma HLS BIND_OP variable=packed_prod op=mul impl=dsp

        raw_sum0 += static_cast<ap_uint<16> >(packed_prod.range(15, 0));
        raw_sum1 += static_cast<ap_uint<16> >(packed_prod.range(33, 18));
    }

    *raw_row0 = raw_sum0;
    *raw_row1 = raw_sum1;
}

static inline ap_int<9> vlm_i8_from_pack(ap_uint<32> pack, uint32_t lane) {
    #pragma HLS INLINE
    const ap_uint<8> bits = pack.range(static_cast<int>((lane + 1u) * 8u - 1u), static_cast<int>(lane * 8u));
    return static_cast<ap_int<8> >(bits);
}

static inline ap_int<11> vlm_i8x4_sum_pack(ap_uint<32> pack) {
    #pragma HLS INLINE
    ap_int<11> sum = 0;
    for (uint32_t lane = 0; lane < 4u; ++lane) {
        #pragma HLS UNROLL
        sum += vlm_i8_from_pack(pack, lane);
    }
    return sum;
}

static inline ap_uint<36> vlm_i8x2_dot4_u8packed_raw_pack(
        ap_uint<32> a_row0_pack,
        ap_uint<32> a_row1_pack,
        ap_uint<32> b_pack) {
    #pragma HLS INLINE

    ap_uint<18> raw_sum0 = 0;
    ap_uint<18> raw_sum1 = 0;

    for (uint32_t ko = 0; ko < 4u; ++ko) {
        #pragma HLS UNROLL
        const ap_int<9> a0_s = vlm_i8_from_pack(a_row0_pack, ko);
        const ap_int<9> a1_s = vlm_i8_from_pack(a_row1_pack, ko);
        const ap_int<9> b_s = vlm_i8_from_pack(b_pack, ko);
        const ap_uint<8> a0_u = static_cast<ap_uint<8> >(a0_s + 128);
        const ap_uint<8> a1_u = static_cast<ap_uint<8> >(a1_s + 128);
        const ap_uint<8> b_u = static_cast<ap_uint<8> >(b_s + 128);

        ap_uint<26> packed_a = 0;
        packed_a.range(7, 0) = a0_u;
        packed_a.range(25, 18) = a1_u;

        ap_uint<34> packed_prod = packed_a * b_u;
        #pragma HLS BIND_OP variable=packed_prod op=mul impl=dsp

        raw_sum0 += static_cast<ap_uint<16> >(packed_prod.range(15, 0));
        raw_sum1 += static_cast<ap_uint<16> >(packed_prod.range(33, 18));
    }

    ap_uint<36> packed_raw = 0;
    packed_raw.range(17, 0) = raw_sum0;
    packed_raw.range(35, 18) = raw_sum1;
    return packed_raw;
}

#endif
