#include "../src/reuse_common.h"
#include "../src/utils.h"

// 设计的consideration
// 1. 在同一个模块中，完成X的读取，多层W的读取，Y的写回
// 2. 将W的读取进行打包，解决WS并非byte-aligned的问题，每一块权重的读取是G*G个WQ，G个WS

// model hyperparameters
constexpr int G                 = REUSE_CP;
constexpr int T                 = REUSE_LLM_TILE_T;
constexpr int S                 = LLAMA_S;  // LLM KV cache 最大上下文长度
constexpr int C                 = LLAMA_C;
constexpr int H                 = LLAMA_KVH;
constexpr int QH                = LLAMA_H;
constexpr int HC                = LLAMA_HC;
// design hyperparameters
constexpr int DW_KV_CACHE_PACK  = 128;  // 128 bit to pack 8 values and 1 scale (8*8 + 4 = 68 < 128)
constexpr int TP = 1;
constexpr int SP = G; 
constexpr int CP = G;
// derived hyperparameters
constexpr int HCT               = HC / CP;
constexpr int ST                = S  / SP;
constexpr int TST               = T  / SP;
constexpr int K_CACHE_PACKS     = LLAMA_L * H * S * HCT;
constexpr int V_CACHE_PACKS     = LLAMA_L * H * HC * ST;
constexpr int KV_CACHE_PACKS    = K_CACHE_PACKS + V_CACHE_PACKS;
constexpr int K_CACHE_BYTES     = K_CACHE_PACKS * DW_KV_CACHE_PACK / 8;
constexpr int V_CACHE_BYTES     = V_CACHE_PACKS * DW_KV_CACHE_PACK / 8;
constexpr int KV_CACHE_BYTES    = K_CACHE_BYTES + V_CACHE_BYTES;
// K cache and V cache are adjacent in memory, use a unified memory interface.
// Native GQA stores only LLAMA_KVH KV heads; QK/RV reuse each KV head for three Q heads.
constexpr int V_CACHE_OFFSET    = K_CACHE_PACKS;

static_assert(QH % H == 0, "KV_CACHE native GQA expects query heads to be an integer multiple of KV heads");
static_assert(HC == 64, "KV_CACHE expects 64-dim heads aligned with QK/RV GEMM");
static_assert(T == SP, "KV_CACHE writeback assumes one 8-token tile per decode chunk");

// define data types
typedef ap_int <DW_KV_CACHE_PACK> cache_pack_t;
typedef ap_int <DW_AQ           > aq_t;
typedef ap_int <DW_AS           > as_t;
constexpr int KV_SCALE_MASK = (1 << DW_AS) - 1;

static_assert(DW_AQ*CP+DW_AS <= cache_pack_t::width, "cache_pack_t is not wide enough");
static_assert(DW_AQ*SP+DW_AS <= cache_pack_t::width, "cache_pack_t is not wide enough");



void rd_kv_cache(
    int l_begin,
    int l_close,
    int valid_s,
    int valid_st,

    cache_pack_t* memory_k_cache,

    hls::stream<hls::vector<aq_t, CP> >& kq_cache_i_stream,
    hls::stream<hls::vector<as_t,  1> >& ks_cache_i_stream,
    hls::stream<hls::vector<aq_t, SP> >& vq_cache_i_stream,
    hls::stream<hls::vector<as_t,  1> >& vs_cache_i_stream
){
    for(int l=l_begin; l<l_close && l<LLAMA_L; ++l){
        for(int h=0; h<H; ++h){
            // read K cache：LLM causal windowing 只 replay 当前可见 token 前缀。
            for(int s=0; s<valid_s; ++s){
                #pragma HLS loop_tripcount min=T max=S
                for(int hct=0; hct<HCT; ++hct){
                    #pragma HLS pipeline II=1
                    hls::vector<aq_t, CP> kq_vec;
                    hls::vector<as_t, 1 > ks_vec;
                    cache_pack_t pack = memory_k_cache[l*H*S*HCT + h*S*HCT + s*HCT + hct];
                    // unpack
                    for(int cp=0; cp<CP; ++cp){
                        #pragma HLS unroll
                        kq_vec[cp] = pack.range((cp+1)*DW_AQ-1, cp*DW_AQ);
                    }
                    ks_vec[0] = pack.range(CP*DW_AQ+DW_AS-1, CP*DW_AQ);
                    kq_cache_i_stream.write(kq_vec);
                    ks_cache_i_stream.write(ks_vec);
                }
            }
            // read V cache
            for(int hc=0; hc<HC; ++hc){
                for(int st=0; st<valid_st; ++st){
                    #pragma HLS loop_tripcount min=TST max=ST
                    #pragma HLS pipeline II=1
                    hls::vector<aq_t, SP> vq_vec;
                    hls::vector<as_t, 1 > vs_vec;
                    cache_pack_t pack = memory_k_cache[V_CACHE_OFFSET + l*H*HC*ST + h*HC*ST + hc*ST + st];
                    // unpack
                    for(int sp=0; sp<SP; ++sp){
                        #pragma HLS unroll
                        vq_vec[sp] = pack.range((sp+1)*DW_AQ-1, sp*DW_AQ);
                    }
                    vs_vec[0] = pack.range(SP*DW_AQ+DW_AS-1, SP*DW_AQ);
                    vq_cache_i_stream.write(vq_vec);
                    vs_cache_i_stream.write(vs_vec);
                }
            }
        }
    }
}



void wr_kv_cache(
    int l_begin,
    int l_close,
    int chunk,
    int valid_s,
    int valid_st,

    cache_pack_t* memory_k_cache,

    hls::stream<hls::vector<aq_t, CP> >& kq_cache_o_stream,
    hls::stream<hls::vector<as_t,  1> >& ks_cache_o_stream,

    hls::stream<hls::vector<aq_t, SP> >& vq_cache_o_stream,
    hls::stream<hls::vector<as_t,  1> >& vs_cache_o_stream
){
    // write only a block of cache: pos_id~pos_id+T
    for(int l=l_begin; l<l_close && l<LLAMA_L; ++l){
        for(int h=0; h<H; ++h){
            // directly write to memory, no buffer
            for(int t=0; t<T; ++t){
                for(int hct=0; hct<HCT; ++hct){
                    #pragma HLS pipeline II=1
                    hls::vector<aq_t, CP> kq_vec = kq_cache_o_stream.read();
                    hls::vector<as_t, 1 > ks_vec = ks_cache_o_stream.read();
                    cache_pack_t pack = 0;
                    // pack
                    for(int cp=0; cp<CP; ++cp){
                        #pragma HLS unroll
                        pack.range((cp+1)*DW_AQ-1, cp*DW_AQ) = kq_vec[cp];
                    }
                    pack.range(CP*DW_AQ+DW_AS-1, CP*DW_AQ) = ks_vec[0];
                    int s_idx = chunk * T + t;
                    if (s_idx < valid_s && s_idx < S) {
                        memory_k_cache[l*H*S*HCT + h*S*HCT + s_idx*HCT + hct] = pack;
                    }
                }
            }
            for(int hc=0; hc<HC; ++hc){
                for(int st=chunk; st<chunk+TST; ++st){
                    #pragma HLS pipeline II=1
                    hls::vector<aq_t, SP> vq_vec = vq_cache_o_stream.read();
                    hls::vector<as_t, 1 > vs_vec = vs_cache_o_stream.read();
                    cache_pack_t pack = 0;
                    // pack
                    for(int sp=0; sp<SP; ++sp){
                        #pragma HLS unroll
                        pack.range((sp+1)*DW_AQ-1, sp*DW_AQ) = vq_vec[sp];
                    }
                    pack.range(SP*DW_AQ+DW_AS-1, SP*DW_AQ) = vs_vec[0];
                    if (st < valid_st && st < ST) {
                        memory_k_cache[V_CACHE_OFFSET + l*H*HC*ST + h*HC*ST + hc*ST + st] = pack;
                    }
                }
            }
        }
    }
}


void top(
    REUSE_MODE_T mode,
    int l_begin,
    int l_close,
    int pos,

    cache_pack_t* memory_k_cache,

    // cache streams
    hls::stream<hls::vector<aq_t, CP> >& kq_cache_i_stream,
    hls::stream<hls::vector<aq_t, CP> >& kq_cache_o_stream,
    hls::stream<hls::vector<as_t,  1> >& ks_cache_i_stream,
    hls::stream<hls::vector<as_t,  1> >& ks_cache_o_stream,

    hls::stream<hls::vector<aq_t, SP> >& vq_cache_i_stream,
    hls::stream<hls::vector<aq_t, SP> >& vq_cache_o_stream,
    hls::stream<hls::vector<as_t,  1> >& vs_cache_i_stream,
    hls::stream<hls::vector<as_t,  1> >& vs_cache_o_stream
){
    #pragma HLS interface ap_ctrl_chain port=return
    #pragma HLS interface mode=m_axi bundle=gmem1 port=memory_k_cache     depth=KV_CACHE_PACKS    offset=direct

    #pragma HLS interface axis port=kq_cache_i_stream
    #pragma HLS interface axis port=kq_cache_o_stream
    #pragma HLS interface axis port=ks_cache_i_stream
    #pragma HLS interface axis port=ks_cache_o_stream
    #pragma HLS interface axis port=vq_cache_i_stream
    #pragma HLS interface axis port=vq_cache_o_stream
    #pragma HLS interface axis port=vs_cache_i_stream
    #pragma HLS interface axis port=vs_cache_o_stream

    #pragma HLS aggregate variable=kq_cache_i_stream compact=bit
    #pragma HLS aggregate variable=kq_cache_o_stream compact=bit
    #pragma HLS aggregate variable=ks_cache_i_stream compact=bit
    #pragma HLS aggregate variable=ks_cache_o_stream compact=bit
    #pragma HLS aggregate variable=vq_cache_i_stream compact=bit
    #pragma HLS aggregate variable=vq_cache_o_stream compact=bit
    #pragma HLS aggregate variable=vs_cache_i_stream compact=bit
    #pragma HLS aggregate variable=vs_cache_o_stream compact=bit

    // ViT attention 不使用 LLM KV cache。mode=ViT 时保持 memory 和所有 cache stream no-op，
    // 避免 Connector/ViT 调度误触发 cache 端口导致资源或死锁问题。
    if (is_vit_mode(mode)) {
        return;
    }

    // LLM 调度只传绝对 pos，模块内部派生当前 8-token chunk，和其他复用 IP 的接口一致。
    reuse_pos_t decoded = reuse_decode_pos(mode, pos);
    int chunk = decoded.chunk;
    int valid_s = reuse_attention_valid_s(mode, pos);
    int valid_st = reuse_attention_valid_st(mode, pos);

    rd_kv_cache(l_begin, l_close, valid_s, valid_st, memory_k_cache, kq_cache_i_stream, ks_cache_i_stream, vq_cache_i_stream, vs_cache_i_stream);
    wr_kv_cache(l_begin, l_close, chunk, valid_s, valid_st, memory_k_cache, kq_cache_o_stream, ks_cache_o_stream, vq_cache_o_stream, vs_cache_o_stream);
}


cache_pack_t make_k_pack(int l, int h, int s, int hct) {
    // Testbench 使用和 memory layout 一致的 K pack，低位为 8 个 q lane，随后一个 scale。
    cache_pack_t pack = 0;
    for (int cp = 0; cp < CP; ++cp) {
        ap_int<DW_AQ> q = (ap_int<DW_AQ>)((l * 17 + h * 7 + s * 3 + hct + cp) & 0x7f);
        pack.range((cp + 1) * DW_AQ - 1, cp * DW_AQ) = q;
    }
    ap_uint<DW_AS> scale = (ap_uint<DW_AS>)((l + h + s + hct) & KV_SCALE_MASK);
    pack.range(CP * DW_AQ + DW_AS - 1, CP * DW_AQ) = scale;
    return pack;
}

cache_pack_t make_v_pack(int l, int h, int hc, int st) {
    // Testbench 使用和 memory layout 一致的 V pack，V cache 按 [H][HC][S/SP] x SP 保存。
    cache_pack_t pack = 0;
    for (int sp = 0; sp < SP; ++sp) {
        ap_int<DW_AQ> q = (ap_int<DW_AQ>)((l * 19 + h * 5 + hc * 3 + st + sp) & 0x7f);
        pack.range((sp + 1) * DW_AQ - 1, sp * DW_AQ) = q;
    }
    ap_uint<DW_AS> scale = (ap_uint<DW_AS>)((l + h + hc + st) & KV_SCALE_MASK);
    pack.range(SP * DW_AQ + DW_AS - 1, SP * DW_AQ) = scale;
    return pack;
}

void fill_memory_layer(int l, vector<cache_pack_t> &memory) {
    // 模拟 PS/M_AXI 外部 memory 中的 LLM K/V cache 布局，只填当前测试层即可。
    for (int h = 0; h < H; ++h) {
        for (int s = 0; s < S; ++s) {
            for (int hct = 0; hct < HCT; ++hct) {
                int idx = l * H * S * HCT + h * S * HCT + s * HCT + hct;
                memory[idx] = make_k_pack(l, h, s, hct);
            }
        }
        for (int hc = 0; hc < HC; ++hc) {
            for (int st = 0; st < ST; ++st) {
                int idx = V_CACHE_OFFSET + l * H * HC * ST + h * HC * ST + hc * ST + st;
                memory[idx] = make_v_pack(l, h, hc, st);
            }
        }
    }
}

void feed_llm_current_tile_outputs(
    int l,
    int chunk,
    hls::stream<hls::vector<aq_t, CP> > &kq_cache_o_stream,
    hls::stream<hls::vector<as_t,  1> > &ks_cache_o_stream,
    hls::stream<hls::vector<aq_t, SP> > &vq_cache_o_stream,
    hls::stream<hls::vector<as_t,  1> > &vs_cache_o_stream
) {
    // 模拟 QK_GEMM/RV_GEMM 写回当前 8-token tile 的 K/V cache，顺序匹配 wr_kv_cache。
    for (int h = 0; h < H; ++h) {
        for (int t = 0; t < T; ++t) {
            for (int hct = 0; hct < HCT; ++hct) {
                hls::vector<aq_t, CP> q_vec;
                hls::vector<as_t,  1> s_vec;
                for (int cp = 0; cp < CP; ++cp) {
                    q_vec[cp] = (aq_t)((l + h + chunk + t + hct + cp + 37) & 0x7f);
                }
                s_vec[0] = (as_t)((l + h + chunk + t + hct + 7) & KV_SCALE_MASK);
                kq_cache_o_stream.write(q_vec);
                ks_cache_o_stream.write(s_vec);
            }
        }
        for (int hc = 0; hc < HC; ++hc) {
            hls::vector<aq_t, SP> q_vec;
            hls::vector<as_t,  1> s_vec;
            for (int sp = 0; sp < SP; ++sp) {
                q_vec[sp] = (aq_t)((l + h + hc + chunk + sp + 41) & 0x7f);
            }
            s_vec[0] = (as_t)((l + h + hc + chunk + 9) & KV_SCALE_MASK);
            vq_cache_o_stream.write(q_vec);
            vs_cache_o_stream.write(s_vec);
        }
    }
}

void test_llm_mode() {
    // LLM mode 覆盖：从 memory 只 replay causal 可见前缀，并只把当前 partial tile 的真实 lane 写回 memory。
    constexpr int TEST_L = 0;
    constexpr int TEST_POS = 100;
    constexpr int TEST_CHUNK = TEST_POS / T;
    constexpr int TEST_VALID_S = TEST_POS + 1;
    constexpr int TEST_VALID_ST = (TEST_VALID_S + SP - 1) / SP;

    vector<cache_pack_t> memory(KV_CACHE_PACKS);
    fill_memory_layer(TEST_L, memory);

    hls::stream<hls::vector<aq_t, CP> > kq_cache_i_stream("llm_kq_i");
    hls::stream<hls::vector<as_t,  1> > ks_cache_i_stream("llm_ks_i");
    hls::stream<hls::vector<aq_t, CP> > kq_cache_o_stream("llm_kq_o");
    hls::stream<hls::vector<as_t,  1> > ks_cache_o_stream("llm_ks_o");
    hls::stream<hls::vector<aq_t, SP> > vq_cache_i_stream("llm_vq_i");
    hls::stream<hls::vector<as_t,  1> > vs_cache_i_stream("llm_vs_i");
    hls::stream<hls::vector<aq_t, SP> > vq_cache_o_stream("llm_vq_o");
    hls::stream<hls::vector<as_t,  1> > vs_cache_o_stream("llm_vs_o");

    feed_llm_current_tile_outputs(TEST_L, TEST_CHUNK, kq_cache_o_stream, ks_cache_o_stream,
                                  vq_cache_o_stream, vs_cache_o_stream);

    top(MODE_LLM, TEST_L, TEST_L + 1, TEST_POS, memory.data(),
        kq_cache_i_stream, kq_cache_o_stream, ks_cache_i_stream, ks_cache_o_stream,
        vq_cache_i_stream, vq_cache_o_stream, vs_cache_i_stream, vs_cache_o_stream);

    assert(kq_cache_i_stream.size() == H * TEST_VALID_S * HCT);
    assert(ks_cache_i_stream.size() == H * TEST_VALID_S * HCT);
    assert(vq_cache_i_stream.size() == H * HC * TEST_VALID_ST);
    assert(vs_cache_i_stream.size() == H * HC * TEST_VALID_ST);
    assert(kq_cache_o_stream.empty());
    assert(ks_cache_o_stream.empty());
    assert(vq_cache_o_stream.empty());
    assert(vs_cache_o_stream.empty());
    while (!kq_cache_i_stream.empty()) kq_cache_i_stream.read();
    while (!ks_cache_i_stream.empty()) ks_cache_i_stream.read();
    while (!vq_cache_i_stream.empty()) vq_cache_i_stream.read();
    while (!vs_cache_i_stream.empty()) vs_cache_i_stream.read();

    // 抽样检查当前 chunk 的 K/V 目标地址确实被写回。
    int k_idx = TEST_L * H * S * HCT + 0 * S * HCT + (TEST_CHUNK * T) * HCT + 0;
    int k_future_idx = TEST_L * H * S * HCT + 0 * S * HCT + (TEST_POS + 1) * HCT + 0;
    int v_idx = V_CACHE_OFFSET + TEST_L * H * HC * ST + 0 * HC * ST + 0 * ST + TEST_CHUNK;
    assert(memory[k_idx] != make_k_pack(TEST_L, 0, TEST_CHUNK * T, 0));
    assert(memory[k_future_idx] == make_k_pack(TEST_L, 0, TEST_POS + 1, 0));
    assert(memory[v_idx] != make_v_pack(TEST_L, 0, 0, TEST_CHUNK));

    cout << "KV_CACHE LLM mode passed" << endl;
}

void test_vit_noop() {
    // ViT mode 必须不读写 memory 和 cache stream；预置数据应原样保留。
    vector<cache_pack_t> memory(1);
    memory[0] = 12345;

    hls::stream<hls::vector<aq_t, CP> > kq_cache_i_stream("vit_kq_i");
    hls::stream<hls::vector<as_t,  1> > ks_cache_i_stream("vit_ks_i");
    hls::stream<hls::vector<aq_t, CP> > kq_cache_o_stream("vit_kq_o");
    hls::stream<hls::vector<as_t,  1> > ks_cache_o_stream("vit_ks_o");
    hls::stream<hls::vector<aq_t, SP> > vq_cache_i_stream("vit_vq_i");
    hls::stream<hls::vector<as_t,  1> > vs_cache_i_stream("vit_vs_i");
    hls::stream<hls::vector<aq_t, SP> > vq_cache_o_stream("vit_vq_o");
    hls::stream<hls::vector<as_t,  1> > vs_cache_o_stream("vit_vs_o");

    hls::vector<aq_t, CP> aq_vec;
    hls::vector<as_t,  1> as_vec;
    for (int i = 0; i < CP; ++i) aq_vec[i] = i;
    as_vec[0] = 3;
    kq_cache_i_stream.write(aq_vec);
    ks_cache_i_stream.write(as_vec);
    kq_cache_o_stream.write(aq_vec);
    ks_cache_o_stream.write(as_vec);
    vq_cache_i_stream.write(aq_vec);
    vs_cache_i_stream.write(as_vec);
    vq_cache_o_stream.write(aq_vec);
    vs_cache_o_stream.write(as_vec);

    top(MODE_VIT, 0, 1, 0, memory.data(),
        kq_cache_i_stream, kq_cache_o_stream, ks_cache_i_stream, ks_cache_o_stream,
        vq_cache_i_stream, vq_cache_o_stream, vs_cache_i_stream, vs_cache_o_stream);

    assert(memory[0] == 12345);
    assert(kq_cache_i_stream.size() == 1);
    assert(ks_cache_i_stream.size() == 1);
    assert(kq_cache_o_stream.size() == 1);
    assert(ks_cache_o_stream.size() == 1);
    assert(vq_cache_i_stream.size() == 1);
    assert(vs_cache_i_stream.size() == 1);
    assert(vq_cache_o_stream.size() == 1);
    assert(vs_cache_o_stream.size() == 1);
    kq_cache_i_stream.read();
    ks_cache_i_stream.read();
    kq_cache_o_stream.read();
    ks_cache_o_stream.read();
    vq_cache_i_stream.read();
    vs_cache_i_stream.read();
    vq_cache_o_stream.read();
    vs_cache_o_stream.read();

    cout << "KV_CACHE ViT no-op passed" << endl;
}

int main() {
    test_llm_mode();
    test_vit_noop();
    cout << "KV_CACHE CSim done with 0 errors" << endl;
    return 0;
}
