#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <tuple>
#include <string>
#include "sm_profiler.h"

// sm-profiler event IDs (must match decode_ldg.cu)
#define SM_PROF_EMBEDDING        0
#define SM_PROF_QKV_PROJ         1
#define SM_PROF_QK_NORM_ROPE     2
#define SM_PROF_ATTN_COMPUTE     3
#define SM_PROF_ATTN_PREFETCH    4
#define SM_PROF_O_PROJ_MLP       5
#define SM_PROF_FINAL_NORM       6
#define SM_PROF_GRID_SYNC        7
#define SM_PROF_NUM_EVENTS       8

// Number of warps per block (must match LDG_BLOCK_SIZE / 32 in decode_ldg.cu)
#define LDG_NUM_WARPS 16

// Forward declarations — defined in decode_ldg.cu
extern "C" void launch_ldg_decode(
    int input_token_id,
    int* output_token_id,
    const void* embed_weight,
    const void* layer_weights,
    const void* final_norm_weight,
    const void* lm_head_weight,
    const void* cos_table,
    const void* sin_table,
    void* k_cache,
    void* v_cache,
    void* hidden_buffer,
    void* g_activations,
    void* g_residual,
    void* g_q,
    void* g_k,
    void* g_v,
    void* g_attn_out,
    void* attn_partial_max,
    void* attn_partial_sum,
    void* attn_partial_out,
    void* g_mlp_intermediate,
    void* g_normalized,
    void* block_max_vals,
    void* block_max_idxs,
    int num_blocks,
    int num_layers,
    int position,
    int cache_len,
    int max_seq_len,
    float attn_scale,
    uint64_t* profiler_buffer,
    cudaStream_t stream
);

extern "C" void launch_ldg_decode_with_logits(
    int input_token_id,
    int* output_token_id,
    float* logits_output,
    const void* embed_weight,
    const void* layer_weights,
    const void* final_norm_weight,
    const void* lm_head_weight,
    const void* cos_table,
    const void* sin_table,
    void* k_cache,
    void* v_cache,
    void* hidden_buffer,
    void* g_activations,
    void* g_residual,
    void* g_q,
    void* g_k,
    void* g_v,
    void* g_attn_out,
    void* attn_partial_max,
    void* attn_partial_sum,
    void* attn_partial_out,
    void* g_mlp_intermediate,
    void* g_normalized,
    void* block_max_vals,
    void* block_max_idxs,
    int num_blocks,
    int num_layers,
    int position,
    int cache_len,
    int max_seq_len,
    float attn_scale,
    uint64_t* profiler_buffer,
    cudaStream_t stream
);

extern "C" void launch_ldg_generate_nosync(
    int first_token_id,
    int num_steps,
    const void* embed_weight,
    const void* layer_weights,
    const void* final_norm_weight,
    const void* lm_head_weight,
    const void* cos_table,
    const void* sin_table,
    void* k_cache,
    void* v_cache,
    void* hidden_buffer,
    void* g_activations,
    void* g_residual,
    void* g_q,
    void* g_k,
    void* g_v,
    void* g_attn_out,
    void* attn_partial_max,
    void* attn_partial_sum,
    void* attn_partial_out,
    void* g_mlp_intermediate,
    void* g_normalized,
    void* block_max_vals,
    void* block_max_idxs,
    int* output_log,
    int* valid_steps,
    int eos_token_id,
    int num_blocks,
    int num_layers,
    int start_position,
    int max_seq_len,
    float attn_scale,
    uint64_t* profiler_buffer,
    cudaStream_t stream
);

extern "C" void launch_ldg_generate_from_device_nosync(
    const int* first_token_device,
    int num_steps,
    const void* embed_weight,
    const void* layer_weights,
    const void* final_norm_weight,
    const void* lm_head_weight,
    const void* cos_table,
    const void* sin_table,
    void* k_cache,
    void* v_cache,
    void* hidden_buffer,
    void* g_activations,
    void* g_residual,
    void* g_q,
    void* g_k,
    void* g_v,
    void* g_attn_out,
    void* attn_partial_max,
    void* attn_partial_sum,
    void* attn_partial_out,
    void* g_mlp_intermediate,
    void* g_normalized,
    void* block_max_vals,
    void* block_max_idxs,
    int* output_log,
    int* valid_steps,
    int eos_token_id,
    int num_blocks,
    int num_layers,
    int start_position,
    int max_seq_len,
    float attn_scale,
    uint64_t* profiler_buffer,
    cudaStream_t stream
);

// ---- sm-profiler state ----
static sm_profiler_buffer_t g_profiler = nullptr;
static bool g_profiler_enabled = false;

void init_profiler(int64_t num_blocks, int64_t max_events_per_group) {
    if (g_profiler) {
        sm_profiler_destroy_buffer(g_profiler);
    }
    g_profiler = sm_profiler_create_buffer(
        static_cast<uint32_t>(num_blocks),
        LDG_NUM_WARPS,
        static_cast<uint32_t>(max_events_per_group),
        1  // enabled
    );
    sm_profiler_register_event(g_profiler, SM_PROF_EMBEDDING, "embedding");
    sm_profiler_register_event(g_profiler, SM_PROF_QKV_PROJ, "qkv_projection");
    sm_profiler_register_event(g_profiler, SM_PROF_QK_NORM_ROPE, "qk_norm_rope_cache");
    sm_profiler_register_event(g_profiler, SM_PROF_ATTN_COMPUTE, "attn_compute");
    sm_profiler_register_event(g_profiler, SM_PROF_ATTN_PREFETCH, "attn_prefetch");
    sm_profiler_register_event(g_profiler, SM_PROF_O_PROJ_MLP, "o_proj_postnorm_mlp");
    sm_profiler_register_event(g_profiler, SM_PROF_FINAL_NORM, "final_norm");
    sm_profiler_register_event(g_profiler, SM_PROF_GRID_SYNC, "grid_sync");
    g_profiler_enabled = true;
}

void reset_profiler() {
    if (g_profiler) {
        sm_profiler_init_buffer(g_profiler);
    }
}

void export_profiler(const std::string& filename) {
    if (g_profiler) {
        sm_profiler_export_to_file(g_profiler, filename.c_str());
    }
}

void destroy_profiler() {
    if (g_profiler) {
        sm_profiler_destroy_buffer(g_profiler);
        g_profiler = nullptr;
        g_profiler_enabled = false;
    }
}

int decode(
    int input_token_id,
    int position,
    int cache_len,
    at::Tensor layer_weights_bytes,
    at::Tensor embed_weight,
    at::Tensor final_norm_weight,
    at::Tensor lm_head_weight,
    at::Tensor cos_table,
    at::Tensor sin_table,
    at::Tensor k_cache,
    at::Tensor v_cache,
    at::Tensor hidden_buffer,
    at::Tensor g_activations,
    at::Tensor g_residual,
    at::Tensor g_q,
    at::Tensor g_k,
    at::Tensor g_v,
    at::Tensor g_attn_out,
    at::Tensor attn_partial_max,
    at::Tensor attn_partial_sum,
    at::Tensor attn_partial_out,
    at::Tensor g_mlp_intermediate,
    at::Tensor g_normalized,
    at::Tensor block_max_vals,
    at::Tensor block_max_idxs,
    int64_t num_blocks,
    int64_t num_layers,
    int64_t max_seq_len,
    double attn_scale
) {
    auto output_token = at::empty({1},
        at::TensorOptions().dtype(at::kInt).device(at::kCUDA));

    auto stream = at::cuda::getCurrentCUDAStream();

    uint64_t* prof_ptr = g_profiler_enabled ?
        sm_profiler_get_device_ptr(g_profiler) : nullptr;

    launch_ldg_decode(
        input_token_id,
        output_token.data_ptr<int>(),
        embed_weight.data_ptr<at::Half>(),
        reinterpret_cast<const void*>(layer_weights_bytes.data_ptr<uint8_t>()),
        final_norm_weight.data_ptr<at::Half>(),
        lm_head_weight.data_ptr<at::Half>(),
        cos_table.data_ptr<at::Half>(),
        sin_table.data_ptr<at::Half>(),
        k_cache.data_ptr<at::Half>(),
        v_cache.data_ptr<at::Half>(),
        hidden_buffer.data_ptr<at::Half>(),
        g_activations.data_ptr<float>(),
        g_residual.data_ptr<float>(),
        g_q.data_ptr<float>(),
        g_k.data_ptr<float>(),
        g_v.data_ptr<float>(),
        g_attn_out.data_ptr<float>(),
        attn_partial_max.data_ptr<float>(),
        attn_partial_sum.data_ptr<float>(),
        attn_partial_out.data_ptr<float>(),
        g_mlp_intermediate.data_ptr<float>(),
        g_normalized.data_ptr<float>(),
        block_max_vals.data_ptr<float>(),
        block_max_idxs.data_ptr<int>(),
        static_cast<int>(num_blocks),
        static_cast<int>(num_layers),
        position,
        cache_len,
        static_cast<int>(max_seq_len),
        static_cast<float>(attn_scale),
        prof_ptr,
        stream
    );

    cudaStreamSynchronize(stream);

    int token_id = output_token.cpu().item<int>();
    return token_id;
}

std::tuple<int, at::Tensor> decode_with_logits(
    int input_token_id,
    int position,
    int cache_len,
    at::Tensor layer_weights_bytes,
    at::Tensor embed_weight,
    at::Tensor final_norm_weight,
    at::Tensor lm_head_weight,
    at::Tensor cos_table,
    at::Tensor sin_table,
    at::Tensor k_cache,
    at::Tensor v_cache,
    at::Tensor hidden_buffer,
    at::Tensor g_activations,
    at::Tensor g_residual,
    at::Tensor g_q,
    at::Tensor g_k,
    at::Tensor g_v,
    at::Tensor g_attn_out,
    at::Tensor attn_partial_max,
    at::Tensor attn_partial_sum,
    at::Tensor attn_partial_out,
    at::Tensor g_mlp_intermediate,
    at::Tensor g_normalized,
    at::Tensor block_max_vals,
    at::Tensor block_max_idxs,
    int64_t num_blocks,
    int64_t num_layers,
    int64_t max_seq_len,
    double attn_scale
) {
    constexpr int VOCAB_SIZE = 151936;

    auto logits = at::empty({VOCAB_SIZE},
        at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
    auto output_token = at::empty({1},
        at::TensorOptions().dtype(at::kInt).device(at::kCUDA));

    auto stream = at::cuda::getCurrentCUDAStream();

    uint64_t* prof_ptr = g_profiler_enabled ?
        sm_profiler_get_device_ptr(g_profiler) : nullptr;

    launch_ldg_decode_with_logits(
        input_token_id,
        output_token.data_ptr<int>(),
        logits.data_ptr<float>(),
        embed_weight.data_ptr<at::Half>(),
        reinterpret_cast<const void*>(layer_weights_bytes.data_ptr<uint8_t>()),
        final_norm_weight.data_ptr<at::Half>(),
        lm_head_weight.data_ptr<at::Half>(),
        cos_table.data_ptr<at::Half>(),
        sin_table.data_ptr<at::Half>(),
        k_cache.data_ptr<at::Half>(),
        v_cache.data_ptr<at::Half>(),
        hidden_buffer.data_ptr<at::Half>(),
        g_activations.data_ptr<float>(),
        g_residual.data_ptr<float>(),
        g_q.data_ptr<float>(),
        g_k.data_ptr<float>(),
        g_v.data_ptr<float>(),
        g_attn_out.data_ptr<float>(),
        attn_partial_max.data_ptr<float>(),
        attn_partial_sum.data_ptr<float>(),
        attn_partial_out.data_ptr<float>(),
        g_mlp_intermediate.data_ptr<float>(),
        g_normalized.data_ptr<float>(),
        block_max_vals.data_ptr<float>(),
        block_max_idxs.data_ptr<int>(),
        static_cast<int>(num_blocks),
        static_cast<int>(num_layers),
        position,
        cache_len,
        static_cast<int>(max_seq_len),
        static_cast<float>(attn_scale),
        prof_ptr,
        stream
    );

    // Synchronise so the host can read the token id
    cudaStreamSynchronize(stream);

    int token_id = output_token.cpu().item<int>();
    return std::make_tuple(token_id, logits);
}

std::tuple<at::Tensor, int64_t> generate_nosync(
    int first_token_id,
    int num_steps,
    at::Tensor layer_weights_bytes,
    at::Tensor embed_weight,
    at::Tensor final_norm_weight,
    at::Tensor lm_head_weight,
    at::Tensor cos_table,
    at::Tensor sin_table,
    at::Tensor k_cache,
    at::Tensor v_cache,
    at::Tensor hidden_buffer,
    at::Tensor g_activations,
    at::Tensor g_residual,
    at::Tensor g_q,
    at::Tensor g_k,
    at::Tensor g_v,
    at::Tensor g_attn_out,
    at::Tensor attn_partial_max,
    at::Tensor attn_partial_sum,
    at::Tensor attn_partial_out,
    at::Tensor g_mlp_intermediate,
    at::Tensor g_normalized,
    at::Tensor block_max_vals,
    at::Tensor block_max_idxs,
    int64_t eos_token_id,
    int64_t num_blocks,
    int64_t num_layers,
    int64_t start_position,
    int64_t max_seq_len,
    double attn_scale
) {
    auto output_log = at::empty(
        {num_steps},
        at::TensorOptions().dtype(at::kInt).device(at::kCUDA));
    auto stream = at::cuda::getCurrentCUDAStream();
    uint64_t* prof_ptr = g_profiler_enabled ?
        sm_profiler_get_device_ptr(g_profiler) : nullptr;
    int valid_steps = 0;

    launch_ldg_generate_nosync(
        first_token_id,
        num_steps,
        embed_weight.data_ptr<at::Half>(),
        reinterpret_cast<const void*>(layer_weights_bytes.data_ptr<uint8_t>()),
        final_norm_weight.data_ptr<at::Half>(),
        lm_head_weight.data_ptr<at::Half>(),
        cos_table.data_ptr<at::Half>(),
        sin_table.data_ptr<at::Half>(),
        k_cache.data_ptr<at::Half>(),
        v_cache.data_ptr<at::Half>(),
        hidden_buffer.data_ptr<at::Half>(),
        g_activations.data_ptr<float>(),
        g_residual.data_ptr<float>(),
        g_q.data_ptr<float>(),
        g_k.data_ptr<float>(),
        g_v.data_ptr<float>(),
        g_attn_out.data_ptr<float>(),
        attn_partial_max.data_ptr<float>(),
        attn_partial_sum.data_ptr<float>(),
        attn_partial_out.data_ptr<float>(),
        g_mlp_intermediate.data_ptr<float>(),
        g_normalized.data_ptr<float>(),
        block_max_vals.data_ptr<float>(),
        block_max_idxs.data_ptr<int>(),
        output_log.data_ptr<int>(),
        &valid_steps,
        static_cast<int>(eos_token_id),
        static_cast<int>(num_blocks),
        static_cast<int>(num_layers),
        static_cast<int>(start_position),
        static_cast<int>(max_seq_len),
        static_cast<float>(attn_scale),
        prof_ptr,
        stream
    );
    cudaStreamSynchronize(stream);
    return std::make_tuple(output_log, static_cast<int64_t>(valid_steps));
}

std::tuple<at::Tensor, int64_t> generate_from_device_nosync(
    at::Tensor first_token_tensor,
    int num_steps,
    at::Tensor layer_weights_bytes,
    at::Tensor embed_weight,
    at::Tensor final_norm_weight,
    at::Tensor lm_head_weight,
    at::Tensor cos_table,
    at::Tensor sin_table,
    at::Tensor k_cache,
    at::Tensor v_cache,
    at::Tensor hidden_buffer,
    at::Tensor g_activations,
    at::Tensor g_residual,
    at::Tensor g_q,
    at::Tensor g_k,
    at::Tensor g_v,
    at::Tensor g_attn_out,
    at::Tensor attn_partial_max,
    at::Tensor attn_partial_sum,
    at::Tensor attn_partial_out,
    at::Tensor g_mlp_intermediate,
    at::Tensor g_normalized,
    at::Tensor block_max_vals,
    at::Tensor block_max_idxs,
    int64_t eos_token_id,
    int64_t num_blocks,
    int64_t num_layers,
    int64_t start_position,
    int64_t max_seq_len,
    double attn_scale
) {
    auto output_log = at::empty(
        {num_steps},
        at::TensorOptions().dtype(at::kInt).device(at::kCUDA));
    auto stream = at::cuda::getCurrentCUDAStream();
    uint64_t* prof_ptr = g_profiler_enabled ?
        sm_profiler_get_device_ptr(g_profiler) : nullptr;
    int valid_steps = 0;

    launch_ldg_generate_from_device_nosync(
        first_token_tensor.data_ptr<int>(),
        num_steps,
        embed_weight.data_ptr<at::Half>(),
        reinterpret_cast<const void*>(layer_weights_bytes.data_ptr<uint8_t>()),
        final_norm_weight.data_ptr<at::Half>(),
        lm_head_weight.data_ptr<at::Half>(),
        cos_table.data_ptr<at::Half>(),
        sin_table.data_ptr<at::Half>(),
        k_cache.data_ptr<at::Half>(),
        v_cache.data_ptr<at::Half>(),
        hidden_buffer.data_ptr<at::Half>(),
        g_activations.data_ptr<float>(),
        g_residual.data_ptr<float>(),
        g_q.data_ptr<float>(),
        g_k.data_ptr<float>(),
        g_v.data_ptr<float>(),
        g_attn_out.data_ptr<float>(),
        attn_partial_max.data_ptr<float>(),
        attn_partial_sum.data_ptr<float>(),
        attn_partial_out.data_ptr<float>(),
        g_mlp_intermediate.data_ptr<float>(),
        g_normalized.data_ptr<float>(),
        block_max_vals.data_ptr<float>(),
        block_max_idxs.data_ptr<int>(),
        output_log.data_ptr<int>(),
        &valid_steps,
        static_cast<int>(eos_token_id),
        static_cast<int>(num_blocks),
        static_cast<int>(num_layers),
        static_cast<int>(start_position),
        static_cast<int>(max_seq_len),
        static_cast<float>(attn_scale),
        prof_ptr,
        stream
    );
    cudaStreamSynchronize(stream);
    return std::make_tuple(output_log, static_cast<int64_t>(valid_steps));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode", &decode,
          "Launch fused decode kernel and return token_id (argmax only, no logits)");
    m.def("decode_with_logits", &decode_with_logits,
          "Launch fused decode kernel and return (token_id, logits[vocab_size])");
    m.def("generate_nosync", &generate_nosync,
          "Queue multiple fused decode steps and return generated token ids");
    m.def("generate_from_device_nosync", &generate_from_device_nosync,
          "Queue multiple fused decode steps using a device first token tensor");
    m.def("init_profiler", &init_profiler,
          "Initialize sm-profiler buffer");
    m.def("reset_profiler", &reset_profiler,
          "Reset profiler buffer before kernel launch");
    m.def("export_profiler", &export_profiler,
          "Export profiler results to Chrome Trace JSON");
    m.def("destroy_profiler", &destroy_profiler,
          "Cleanup profiler resources");
}
