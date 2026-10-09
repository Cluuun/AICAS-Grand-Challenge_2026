#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <mutex>

#include "flashdecode_api.h"

#ifndef FLASHDECODE_EXT_ENABLE_CUDA_ERROR_CHECK
#define FLASHDECODE_EXT_ENABLE_CUDA_ERROR_CHECK 0
#endif

namespace {

inline void check_cuda_tensor(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

inline bool is_supported_dtype(const torch::Tensor& t) {
    return t.scalar_type() == torch::kFloat16 || t.scalar_type() == torch::kBFloat16;
}

inline void check_fd_tensor(const torch::Tensor& t, const char* name) {
    check_cuda_tensor(t, name);
    TORCH_CHECK(is_supported_dtype(t), name, " must be float16 or bfloat16");
}

std::once_flag g_init_once;
std::once_flag g_bf16_init_once;

void ensure_flashdecode_init() {
    std::call_once(g_init_once, []() { flashdecode_mma_init(); });
    std::call_once(g_bf16_init_once, []() { flashdecode_mma_init_bf16(); });
}

inline void maybe_check_cuda(const char* context) {
#if FLASHDECODE_EXT_ENABLE_CUDA_ERROR_CHECK
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, context, ": ", cudaGetErrorString(err));
#else
    (void)context;
#endif
}

}  // namespace

extern "C" void flashdecode_mma_run_qstride_contig_noseqlens(
    const __half *q,
    const __half *k_base,
    const __half *v_base,
    __half *out,
    void *temp,
    int B,
    int H_kv,
    int G,
    int Q_stride,
    int H,
    int S,
    cudaStream_t stream);

extern "C" void flashdecode_mma_run_qstride_bf16(
    const __nv_bfloat16 *q,
    const __nv_bfloat16 *const *k_ptrs,
    const __nv_bfloat16 *const *v_ptrs,
    const int *seq_lengths,
    __nv_bfloat16 *out,
    void *temp,
    int B,
    int H_kv,
    int G,
    int Q_stride,
    int H,
    int S,
    cudaStream_t stream);

extern "C" void flashdecode_mma_run_qstride_contig_noseqlens_bf16(
    const __nv_bfloat16 *q,
    const __nv_bfloat16 *k_base,
    const __nv_bfloat16 *v_base,
    __nv_bfloat16 *out,
    void *temp,
    int B,
    int H_kv,
    int G,
    int Q_stride,
    int H,
    int S,
    cudaStream_t stream);

extern "C" void flashdecode_mma_run_qstride_contig_causal(
    const __half *q,
    const __half *k_base,
    const __half *v_base,
    __half *out,
    void *temp,
    int B,
    int H_kv,
    int G,
    int Q_stride,
    int H,
    int S,
    int virtual_q,
    cudaStream_t stream);

extern "C" void flashdecode_mma_run_qstride_contig_causal_bf16(
    const __nv_bfloat16 *q,
    const __nv_bfloat16 *k_base,
    const __nv_bfloat16 *v_base,
    __nv_bfloat16 *out,
    void *temp,
    int B,
    int H_kv,
    int G,
    int Q_stride,
    int H,
    int S,
    int virtual_q,
    cudaStream_t stream);

class FlashDecodeRunner {
public:
    FlashDecodeRunner(int B, int H_kv, int G, int max_S, int device = -1)
        : B_(B), H_kv_(H_kv), G_(G), max_S_(max_S) {
        TORCH_CHECK(B_ > 0 && H_kv_ > 0 && G_ > 0 && max_S_ > 0, "Invalid runner args");
        H_ = H_kv_ * G_;

        if (device >= 0) {
            device_index_ = device;
        } else {
            device_index_ = c10::cuda::current_device();
        }

        c10::cuda::CUDAGuard guard(device_index_);
        ensure_flashdecode_init();

        auto opts = torch::TensorOptions().device(torch::kCUDA, device_index_);
        const size_t ws_bytes = flashdecode_mma_workspace_size(B_, max_S_, H_kv_, G_);
        workspace_bytes_ = ws_bytes;

        workspace_ = torch::empty({static_cast<long long>(ws_bytes)}, opts.dtype(torch::kUInt8));
    }

    torch::Tensor run(torch::Tensor q_packed,
                      torch::Tensor k,
                      torch::Tensor v,
                      torch::Tensor seq_lengths) {
        (void)seq_lengths;
        auto out = torch::empty({B_, H_kv_, G_, 128}, q_packed.options());
        run_core(q_packed, k, v, out);
        return out;
    }

    void run_into(torch::Tensor q_packed,
                  torch::Tensor k,
                  torch::Tensor v,
                  torch::Tensor seq_lengths,
                  torch::Tensor out) {
        (void)seq_lengths;
        run_core(q_packed, k, v, out);
    }

    torch::Tensor run_decode(torch::Tensor query, torch::Tensor k, torch::Tensor v) {
        ensure_flashdecode_init();
        auto out = torch::empty({B_, 1, H_, 128}, query.options());
        run_decode_into(query, k, v, out);
        return out;
    }

    torch::Tensor run_decode_multi(torch::Tensor query, torch::Tensor k, torch::Tensor v) {
        ensure_flashdecode_init();
        TORCH_CHECK(query.dim() == 4, "query must be [B, H_q, Q, 128]");
        const int64_t Q = query.size(2);
        auto out = torch::empty({B_, Q, H_, 128}, query.options());
        run_decode_multi_into(query, k, v, out);
        return out;
    }

    void run_decode_into(torch::Tensor query, torch::Tensor k, torch::Tensor v, torch::Tensor out) {
        ensure_flashdecode_init();
        check_fd_tensor(query, "query");
        check_fd_tensor(k, "k");
        check_fd_tensor(v, "v");
        check_fd_tensor(out, "out");

        TORCH_CHECK(query.get_device() == device_index_, "query device mismatch");
        TORCH_CHECK(k.get_device() == device_index_, "k device mismatch");
        TORCH_CHECK(v.get_device() == device_index_, "v device mismatch");
        TORCH_CHECK(out.get_device() == device_index_, "out device mismatch");
        TORCH_CHECK(query.scalar_type() == k.scalar_type() && k.scalar_type() == v.scalar_type(),
                    "query/k/v dtype mismatch");
        TORCH_CHECK(out.scalar_type() == query.scalar_type(), "out dtype mismatch with query");

        TORCH_CHECK(query.dim() == 4, "query must be [B, H_q, 1, 128]");
        TORCH_CHECK(query.size(0) == B_ && query.size(1) == H_ && query.size(2) == 1 && query.size(3) == 128,
                    "query shape mismatch");
        TORCH_CHECK(k.dim() == 4 && v.dim() == 4, "k/v must be [B, H_kv, S, 128]");
        TORCH_CHECK(k.size(0) == B_ && v.size(0) == B_, "k/v B mismatch");
        TORCH_CHECK(k.size(1) == H_kv_ && v.size(1) == H_kv_, "k/v H_kv mismatch");
        TORCH_CHECK(k.size(3) == 128 && v.size(3) == 128, "k/v Kdim mismatch");
        TORCH_CHECK(k.size(2) == v.size(2), "k/v S mismatch");
        TORCH_CHECK(out.dim() == 4 && out.size(0) == B_ && out.size(1) == 1 && out.size(2) == H_ && out.size(3) == 128,
                    "out must be [B, 1, H_q, 128]");

        const int64_t S = k.size(2);
        TORCH_CHECK(S > 0 && S <= max_S_, "S out of runner range");

        auto query_grouped = query.view({B_, H_kv_, G_, 128});
        auto out_grouped = out.view({B_, H_kv_, G_, 128});

        run_core(query_grouped, k, v, out_grouped);
    }

    void run_decode_multi_into(torch::Tensor query, torch::Tensor k, torch::Tensor v, torch::Tensor out) {
        ensure_flashdecode_init();
        check_fd_tensor(query, "query");
        check_fd_tensor(k, "k");
        check_fd_tensor(v, "v");
        check_fd_tensor(out, "out");

        TORCH_CHECK(query.get_device() == device_index_, "query device mismatch");
        TORCH_CHECK(k.get_device() == device_index_, "k device mismatch");
        TORCH_CHECK(v.get_device() == device_index_, "v device mismatch");
        TORCH_CHECK(out.get_device() == device_index_, "out device mismatch");
        TORCH_CHECK(query.scalar_type() == k.scalar_type() && k.scalar_type() == v.scalar_type(),
                    "query/k/v dtype mismatch");
        TORCH_CHECK(out.scalar_type() == query.scalar_type(), "out dtype mismatch with query");

        TORCH_CHECK(query.dim() == 4, "query must be [B, H_q, Q, 128]");
        TORCH_CHECK(query.size(0) == B_ && query.size(1) == H_ && query.size(3) == 128,
                    "query shape mismatch");
        TORCH_CHECK(k.dim() == 4 && v.dim() == 4, "k/v must be [B, H_kv, S, 128]");
        TORCH_CHECK(k.size(0) == B_ && v.size(0) == B_, "k/v B mismatch");
        TORCH_CHECK(k.size(1) == H_kv_ && v.size(1) == H_kv_, "k/v H_kv mismatch");
        TORCH_CHECK(k.size(3) == 128 && v.size(3) == 128, "k/v Kdim mismatch");
        TORCH_CHECK(k.size(2) == v.size(2), "k/v S mismatch");
        const int64_t Q = query.size(2);
        TORCH_CHECK(Q > 0 && Q <= 16, "query Q must be in [1, 16]");
        TORCH_CHECK(out.dim() == 4 && out.size(0) == B_ && out.size(1) == Q && out.size(2) == H_ && out.size(3) == 128,
                    "out must be [B, Q, H_q, 128]");

        const int64_t S = k.size(2);
        TORCH_CHECK(S >= Q && S <= max_S_, "S out of runner range");
        const int virtual_B = B_ * static_cast<int>(Q);
        ensure_workspace_for(virtual_B, static_cast<int>(S));

        auto stream = at::cuda::getCurrentCUDAStream(device_index_);
        auto k_base = k.contiguous();
        auto v_base = v.contiguous();

        auto query_q_major = query.permute({0, 2, 1, 3}).contiguous();
        auto q_grouped = query_q_major.view({B_ * static_cast<int>(Q), H_kv_, G_, 128});
        auto out_grouped = out.view({B_ * static_cast<int>(Q), H_kv_, G_, 128});

        const int query_stride = G_;

        if (query.scalar_type() == torch::kFloat16) {
            flashdecode_mma_run_qstride_contig_causal(
                reinterpret_cast<const __half*>(q_grouped.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(k_base.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(v_base.data_ptr<at::Half>()),
                reinterpret_cast<__half*>(out_grouped.data_ptr<at::Half>()),
                workspace_.data_ptr<uint8_t>(),
                virtual_B,
                H_kv_,
                G_,
                query_stride,
                H_,
                static_cast<int>(S),
                static_cast<int>(Q),
                stream.stream());
        } else {
            flashdecode_mma_run_qstride_contig_causal_bf16(
                reinterpret_cast<const __nv_bfloat16*>(q_grouped.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(k_base.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(v_base.data_ptr<at::BFloat16>()),
                reinterpret_cast<__nv_bfloat16*>(out_grouped.data_ptr<at::BFloat16>()),
                workspace_.data_ptr<uint8_t>(),
                virtual_B,
                H_kv_,
                G_,
                query_stride,
                H_,
                static_cast<int>(S),
                static_cast<int>(Q),
                stream.stream());
        }
        maybe_check_cuda("flashdecode_mma_run_multi failed");
    }

    void run_decode_multi_masked_into(torch::Tensor query,
                                      torch::Tensor k,
                                      torch::Tensor v,
                                      torch::Tensor attention_mask,
                                      torch::Tensor out) {
        ensure_flashdecode_init();
        check_fd_tensor(query, "query");
        check_fd_tensor(k, "k");
        check_fd_tensor(v, "v");
        check_fd_tensor(out, "out");

        TORCH_CHECK(attention_mask.is_cuda(), "attention_mask must be a CUDA tensor");
        TORCH_CHECK(is_supported_dtype(attention_mask), "attention_mask must be float16 or bfloat16");

        TORCH_CHECK(query.get_device() == device_index_, "query device mismatch");
        TORCH_CHECK(k.get_device() == device_index_, "k device mismatch");
        TORCH_CHECK(v.get_device() == device_index_, "v device mismatch");
        TORCH_CHECK(attention_mask.get_device() == device_index_, "attention_mask device mismatch");
        TORCH_CHECK(out.get_device() == device_index_, "out device mismatch");
        TORCH_CHECK(query.scalar_type() == k.scalar_type() && k.scalar_type() == v.scalar_type(),
                    "query/k/v dtype mismatch");
        TORCH_CHECK(attention_mask.scalar_type() == query.scalar_type(),
                    "attention_mask dtype mismatch with query");
        TORCH_CHECK(out.scalar_type() == query.scalar_type(), "out dtype mismatch with query");

        TORCH_CHECK(query.dim() == 4, "query must be [B, H_q, Q, 128]");
        TORCH_CHECK(query.size(0) == B_ && query.size(1) == H_ && query.size(3) == 128,
                    "query shape mismatch");
        TORCH_CHECK(k.dim() == 4 && v.dim() == 4, "k/v must be [B, H_kv, S, 128]");
        TORCH_CHECK(k.size(0) == B_ && v.size(0) == B_, "k/v B mismatch");
        TORCH_CHECK(k.size(1) == H_kv_ && v.size(1) == H_kv_, "k/v H_kv mismatch");
        TORCH_CHECK(k.size(3) == 128 && v.size(3) == 128, "k/v Kdim mismatch");
        TORCH_CHECK(k.size(2) == v.size(2), "k/v S mismatch");
        const int64_t Q = query.size(2);
        TORCH_CHECK(Q > 0 && Q <= 16, "query Q must be in [1, 16]");
        const int64_t S = k.size(2);
        TORCH_CHECK(S > 0 && S <= max_S_, "S out of runner range");
        TORCH_CHECK(attention_mask.dim() == 4, "attention_mask must be [B, 1, Q, S]");
        TORCH_CHECK(attention_mask.size(0) == B_ && attention_mask.size(1) == 1 &&
                    attention_mask.size(2) == Q && attention_mask.size(3) == S,
                    "attention_mask must be [B, 1, Q, S] and match query/k");
        TORCH_CHECK(out.dim() == 4 && out.size(0) == B_ && out.size(1) == Q && out.size(2) == H_ && out.size(3) == 128,
                    "out must be [B, Q, H_q, 128]");

        auto stream = at::cuda::getCurrentCUDAStream(device_index_);
        auto mask_base = attention_mask.contiguous();

        if (query.scalar_type() == torch::kFloat16) {
            flashdecode_run_decode_multi_masked(
                reinterpret_cast<const __half*>(query.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(k.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(v.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(mask_base.data_ptr<at::Half>()),
                reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
                B_,
                H_,
                H_kv_,
                G_,
                static_cast<int>(Q),
                static_cast<int>(S),
                stream.stream());
        } else {
            flashdecode_run_decode_multi_masked_bf16(
                reinterpret_cast<const __nv_bfloat16*>(query.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(mask_base.data_ptr<at::BFloat16>()),
                reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
                B_,
                H_,
                H_kv_,
                G_,
                static_cast<int>(Q),
                static_cast<int>(S),
                stream.stream());
        }
        maybe_check_cuda("flashdecode_run_decode_multi_masked failed");
    }

private:
    void ensure_workspace_for(int B, int S) {
        const size_t needed_ws = flashdecode_mma_workspace_size(B, S, H_kv_, G_);
        if (needed_ws > workspace_bytes_) {
            auto opts = torch::TensorOptions().device(torch::kCUDA, device_index_).dtype(torch::kUInt8);
            workspace_ = torch::empty({static_cast<long long>(needed_ws)}, opts);
            workspace_bytes_ = needed_ws;
        }
    }

    void ensure_workspace(int S) {
        ensure_workspace_for(B_, S);
    }

    void run_core(torch::Tensor q_packed,
                  torch::Tensor k,
                  torch::Tensor v,
                  torch::Tensor out) {
        check_fd_tensor(q_packed, "q_packed");
        check_fd_tensor(k, "k");
        check_fd_tensor(v, "v");
        check_fd_tensor(out, "out");

        TORCH_CHECK(q_packed.get_device() == device_index_, "q_packed device mismatch");
        TORCH_CHECK(k.get_device() == device_index_, "k device mismatch");
        TORCH_CHECK(v.get_device() == device_index_, "v device mismatch");
        TORCH_CHECK(out.get_device() == device_index_, "out device mismatch");
        TORCH_CHECK(q_packed.scalar_type() == k.scalar_type() && k.scalar_type() == v.scalar_type(),
                    "q_packed/k/v dtype mismatch");
        TORCH_CHECK(out.scalar_type() == q_packed.scalar_type(), "out dtype mismatch with q_packed");

        TORCH_CHECK(q_packed.dim() == 4, "q_packed must be [B, H_kv, Q_stride, 128]");
        TORCH_CHECK(k.dim() == 4, "k must be [B, H_kv, S, 128]");
        TORCH_CHECK(v.dim() == 4, "v must be [B, H_kv, S, 128]");
        TORCH_CHECK(out.dim() == 4, "out must be [B, H_kv, G, 128]");

        const int64_t B = q_packed.size(0);
        const int64_t H_kv = q_packed.size(1);
        const int64_t Q_stride = q_packed.size(2);
        const int64_t Kdim = q_packed.size(3);

        TORCH_CHECK(B == B_, "B mismatch");
        TORCH_CHECK(H_kv == H_kv_, "H_kv mismatch");
        TORCH_CHECK(Kdim == 128, "Only Kdim=128 supported");
        TORCH_CHECK(Q_stride >= G_, "q_packed Q_stride must be >= G");

        TORCH_CHECK(k.size(0) == B_ && v.size(0) == B_, "k/v B mismatch");
        TORCH_CHECK(k.size(1) == H_kv_ && v.size(1) == H_kv_, "k/v H_kv mismatch");
        TORCH_CHECK(k.size(3) == 128 && v.size(3) == 128, "k/v Kdim mismatch");
        TORCH_CHECK(k.size(2) == v.size(2), "k/v S mismatch");

        TORCH_CHECK(out.size(0) == B_ && out.size(1) == H_kv_ && out.size(2) == G_ && out.size(3) == 128,
                    "out shape mismatch");

        const int64_t S = k.size(2);
        TORCH_CHECK(S > 0 && S <= max_S_, "S out of runner range");
        ensure_workspace(static_cast<int>(S));

        auto stream = at::cuda::getCurrentCUDAStream(device_index_);
        if (q_packed.scalar_type() == torch::kFloat16) {
            auto* k_base = reinterpret_cast<const __half*>(k.data_ptr<at::Half>());
            auto* v_base = reinterpret_cast<const __half*>(v.data_ptr<at::Half>());
            auto* q_ptr = reinterpret_cast<const __half*>(q_packed.data_ptr<at::Half>());
            auto* out_ptr = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
            flashdecode_mma_run_qstride_contig_noseqlens(
                q_ptr,
                k_base,
                v_base,
                out_ptr,
                workspace_.data_ptr<uint8_t>(),
                B_,
                H_kv_,
                G_,
                static_cast<int>(Q_stride),
                H_,
                static_cast<int>(S),
                stream.stream());
        } else {
            auto* k_base = reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>());
            auto* v_base = reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>());
            auto* q_ptr = reinterpret_cast<const __nv_bfloat16*>(q_packed.data_ptr<at::BFloat16>());
            auto* out_ptr = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
            flashdecode_mma_run_qstride_contig_noseqlens_bf16(
                q_ptr,
                k_base,
                v_base,
                out_ptr,
                workspace_.data_ptr<uint8_t>(),
                B_,
                H_kv_,
                G_,
                static_cast<int>(Q_stride),
                H_,
                static_cast<int>(S),
                stream.stream());
        }
        maybe_check_cuda("flashdecode_mma_run failed");
    }
    int B_{0};
    int H_kv_{0};
    int G_{0};
    int H_{0};
    int max_S_{0};
    int device_index_{0};
    size_t workspace_bytes_{0};

    torch::Tensor workspace_;
};

torch::Tensor flashdecode_run(torch::Tensor q_packed,
                              torch::Tensor k,
                              torch::Tensor v,
                              torch::Tensor seq_lengths) {
    TORCH_CHECK(q_packed.dim() == 4, "q_packed must be [B, H_kv, G+2, 128]");
    const int B = static_cast<int>(q_packed.size(0));
    const int H_kv = static_cast<int>(q_packed.size(1));
    const int G = static_cast<int>(q_packed.size(2) - 2);
    const int S = static_cast<int>(k.size(2));
    const int dev = q_packed.get_device();

    FlashDecodeRunner runner(B, H_kv, G, S, dev);
    return runner.run(q_packed, k, v, seq_lengths);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flashdecode_run",
          &flashdecode_run,
          "FlashDecode MMA run (q_packed, k, v, seq_lengths) -> out_packed",
          pybind11::arg("q_packed"),
          pybind11::arg("k"),
          pybind11::arg("v"),
          pybind11::arg("seq_lengths"));

    pybind11::class_<FlashDecodeRunner>(m, "FlashDecodeRunner")
        .def(pybind11::init<int, int, int, int, int>(),
             pybind11::arg("B"),
             pybind11::arg("H_kv"),
             pybind11::arg("G"),
             pybind11::arg("max_S"),
             pybind11::arg("device") = -1)
        .def("run", &FlashDecodeRunner::run,
             pybind11::arg("q_packed"),
             pybind11::arg("k"),
             pybind11::arg("v"),
             pybind11::arg("seq_lengths"))
        .def("run_into", &FlashDecodeRunner::run_into,
             pybind11::arg("q_packed"),
             pybind11::arg("k"),
             pybind11::arg("v"),
             pybind11::arg("seq_lengths"),
             pybind11::arg("out"))
        .def("run_decode", &FlashDecodeRunner::run_decode,
             pybind11::arg("query"),
             pybind11::arg("k"),
             pybind11::arg("v"))
        .def("run_decode_into", &FlashDecodeRunner::run_decode_into,
             pybind11::arg("query"),
             pybind11::arg("k"),
             pybind11::arg("v"),
             pybind11::arg("out"))
        .def("run_decode_multi", &FlashDecodeRunner::run_decode_multi,
             pybind11::arg("query"),
             pybind11::arg("k"),
             pybind11::arg("v"))
        .def("run_decode_multi_into", &FlashDecodeRunner::run_decode_multi_into,
             pybind11::arg("query"),
             pybind11::arg("k"),
             pybind11::arg("v"),
             pybind11::arg("out"))
        .def("run_decode_multi_masked_into", &FlashDecodeRunner::run_decode_multi_masked_into,
             pybind11::arg("query"),
             pybind11::arg("k"),
             pybind11::arg("v"),
             pybind11::arg("attention_mask"),
             pybind11::arg("out"));
}
