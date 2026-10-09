#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

__global__ void copy_full_prompt_kv_kernel(
    const __nv_bfloat16* __restrict__ key_src,
    const __nv_bfloat16* __restrict__ value_src,
    __nv_bfloat16* __restrict__ key_dst,
    __nv_bfloat16* __restrict__ value_dst,
    int64_t total,
    int B,
    int H,
    int copy_len,
    int src_seq_len,
    int dst_seq_len,
    int D) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

  const int64_t DH = static_cast<int64_t>(D);
  const int64_t CD = static_cast<int64_t>(copy_len) * D;
  const int64_t SD = static_cast<int64_t>(src_seq_len) * D;
  const int64_t MD = static_cast<int64_t>(dst_seq_len) * D;
  const int64_t HSD = static_cast<int64_t>(H) * SD;
  const int64_t HMD = static_cast<int64_t>(H) * MD;
  const int64_t BHSD = static_cast<int64_t>(B) * HSD;
  const int64_t BHMD = static_cast<int64_t>(B) * HMD;

  for (; idx < total; idx += stride) {
    int64_t t = idx;
    const int d = static_cast<int>(t % DH);
    t /= DH;
    const int s = static_cast<int>(t % copy_len);
    t /= copy_len;
    const int h = static_cast<int>(t % H);
    t /= H;
    const int b = static_cast<int>(t % B);
    const int l = static_cast<int>(t / B);

    const int64_t src_off =
        static_cast<int64_t>(l) * BHSD +
        static_cast<int64_t>(b) * HSD +
        static_cast<int64_t>(h) * SD +
        static_cast<int64_t>(s) * D +
        d;
    const int64_t dst_off =
        static_cast<int64_t>(l) * BHMD +
        static_cast<int64_t>(b) * HMD +
        static_cast<int64_t>(h) * MD +
        static_cast<int64_t>(s) * D +
        d;

    key_dst[dst_off] = key_src[src_off];
    value_dst[dst_off] = value_src[src_off];
  }
}

__global__ void copy_full_prompt_kv_vec16_kernel(
    const uint4* __restrict__ key_src,
    const uint4* __restrict__ value_src,
    uint4* __restrict__ key_dst,
    uint4* __restrict__ value_dst,
    int B,
    int H,
    int copy_len,
    int src_seq_len,
    int dst_seq_len,
    int vecs_per_row) {
  int segment = blockIdx.x;
  const int h = segment % H;
  segment /= H;
  const int b = segment % B;
  const int l = segment / B;

  const int64_t src_base =
      ((static_cast<int64_t>(l) * B + b) * H + h) * src_seq_len * vecs_per_row;
  const int64_t dst_base =
      ((static_cast<int64_t>(l) * B + b) * H + h) * dst_seq_len * vecs_per_row;
  const int64_t nvec = static_cast<int64_t>(copy_len) * vecs_per_row;

  for (int64_t v = threadIdx.x; v < nvec; v += blockDim.x) {
    key_dst[dst_base + v] = key_src[src_base + v];
    value_dst[dst_base + v] = value_src[src_base + v];
  }
}

}  // namespace

void copy_full_prompt_kv_cuda(
    torch::Tensor key_src,
    torch::Tensor value_src,
    torch::Tensor key_dst,
    torch::Tensor value_dst,
    int64_t seq_len) {
  if (seq_len == 0) {
    return;
  }

  const int L = static_cast<int>(key_src.size(0));
  const int B = static_cast<int>(key_src.size(1));
  const int H = static_cast<int>(key_src.size(2));
  const int copy_len = static_cast<int>(seq_len);
  const int src_seq_len = static_cast<int>(key_src.size(3));
  const int dst_seq_len = static_cast<int>(key_dst.size(3));
  const int D = static_cast<int>(key_src.size(4));
  const int64_t total =
      static_cast<int64_t>(L) * B * H * copy_len * D;

  constexpr int kThreads = 256;
  auto stream = at::cuda::getCurrentCUDAStream();

  if ((D % 8) == 0) {
    const int vecs_per_row = D / 8;
    const int blocks = L * B * H;
    copy_full_prompt_kv_vec16_kernel<<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<const uint4*>(key_src.data_ptr<at::BFloat16>()),
        reinterpret_cast<const uint4*>(value_src.data_ptr<at::BFloat16>()),
        reinterpret_cast<uint4*>(key_dst.data_ptr<at::BFloat16>()),
        reinterpret_cast<uint4*>(value_dst.data_ptr<at::BFloat16>()),
        B,
        H,
        copy_len,
        src_seq_len,
        dst_seq_len,
        vecs_per_row);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }

  const int blocks = static_cast<int>(std::min<int64_t>((total + kThreads - 1) / kThreads, 65535));
  copy_full_prompt_kv_kernel<<<blocks, kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(key_src.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(value_src.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(key_dst.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(value_dst.data_ptr<at::BFloat16>()),
      total,
      B,
      H,
      copy_len,
      src_seq_len,
      dst_seq_len,
      D);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
