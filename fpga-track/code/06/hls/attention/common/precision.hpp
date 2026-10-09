#pragma once

// Precision policy for the LN / softmax / GELU pipeline (S2).
//
// All kernel DDR-facing interfaces stay FP32. ap_fixed is only used inside
// the compute pipelines. The kernel converts FP32 -> ap_fixed at the
// input edge and back to FP32 at the output edge.
//
// Build switches (g++ or Vitis HLS):
//
//   -DVH_FIXED=0   (default) all internal math is FP32. S1 baseline.
//   -DVH_FIXED=1            LN / softmax / GELU use ap_fixed internally.
//
// ap_fixed is available in g++ host C-sim via the Vitis HLS include
// directory (see hls/Makefile: add -I /tools/Xilinx/<ver>/Vitis/include).
//
// Type budget:
//   act_t : ap_fixed<24, 8, AP_RND, AP_SAT> — 8 integer + 16 fractional bits.
//           Range ±128, resolution 2^-16 = 1.5e-5.
//           Used for LN affine output and GELU output.
//
//           Range justification: from a quick stats sweep on the S0 ref
//           intermediates (10_layer_*_out.bin), most layer outputs stay
//           within ±20; the post_ln output is bounded by the LN weights
//           (max ≈ 19 in this checkpoint). 8 integer bits = ±128 leaves
//           plenty of saturation margin.
//
//           Resolution justification: 1e-3 (ap_fixed<16,6>) caused
//           cos-sim 0.987 after 12-layer compounding. 1.5e-5 brings us
//           back near float (>0.999) and still uses 24-bit multipliers,
//           which fit in 2 DSP58 tiles each on KV260 (ZU5EV has 1200
//           DSPs, comfortably enough for the LN/GELU traffic).
//
//           S3 will switch GEMM activations to INT8 + per-row scale,
//           which is much coarser; the S2 envelope here is set wide so
//           the precision regression is dominated by S3, not by us.
//   acc_t : float — pre-LN residual stream can hit ±560 (verified on
//           layer 11 output); their squared deviations easily exceed any
//           narrow fixed accumulator. HLS turns a single float
//           accumulator into a couple of DSPs and a small chunk of LUTs.
//
// These envelope choices come from a quick statistical sweep over the
// 12-layer encoder intermediates produced by host/vh_compare. If a future
// gguf trips saturation we tighten or widen here, not at the kernel call
// site.

#ifndef VH_FIXED
#  define VH_FIXED 0
#endif

#if VH_FIXED
#  ifndef _GLIBCXX_HAS_GTHREADS
#    define _GLIBCXX_HAS_GTHREADS 0
#  endif
#  include "ap_fixed.h"

namespace vh {
typedef ap_fixed<24, 8, AP_RND, AP_SAT>  act_t;
typedef float                            acc_t;
}  // namespace vh

#  define VH_TO_FIXED(T, x) ((T)(x))
#  define VH_TO_FLOAT(x)    ((float)(x))

#else  // VH_FIXED == 0

namespace vh {
typedef float act_t;
typedef float acc_t;
}  // namespace vh

#  define VH_TO_FIXED(T, x) (x)
#  define VH_TO_FLOAT(x)    (x)

#endif
