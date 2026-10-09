# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/torchinductor_root/az/caz6nvwz6he6xnnbw3fdj6mrplyq3w2cgltf3fddqalqq4zhg6dc.py
# Topologically Sorted Source Nodes: [embedding, convert_element_type_3, pow_1, mean], Original ATen: [aten.embedding, prims.convert_element_type, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   convert_element_type_3 => convert_element_type_3
#   embedding => embedding
#   mean => mean
#   pow_1 => pow_1
# Graph fragment:
#   %embedding : [num_users=2] = call_function[target=torch.ops.aten.embedding.default](args = (%arg282_1, %arg0_1), kwargs = {})
#   %convert_element_type_3 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_3, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [-1], True), kwargs = {})
triton_red_fused_convert_element_type_embedding_mean_pow_0 = async_compile.triton('triton_red_fused_convert_element_type_embedding_mean_pow_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp16', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convert_element_type_embedding_mean_pow_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'AA6F4099AE74BF369D45388847321E195BFBEE5A86FCB9B181180FAA5E25C7E8', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_red_fused_convert_element_type_embedding_mean_pow_0(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    _tmp11 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp2 = tl.full([XBLOCK, R0_BLOCK], 151936, tl.int32)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp1 < 0
        tmp5 = tl.where(tmp4, tmp3, tmp1)
        tl.device_assert((0 <= tmp5) & (tmp5 < 151936), "index out of bounds: 0 <= tmp5 < 151936")
        tmp7 = tl.load(in_ptr1 + (r0_0 + 2048*tmp5), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp8 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(r0_mask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp11, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/dn/cdnhg2imtf3zk42s3rtl2ucc4nkrz7ftwdw74kdhkdbqvqboudod.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => slice_scatter_default_1
# Graph fragment:
#   %slice_scatter_default_1 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%select_int_1, %slice_4, 2, 2, 60, 3), kwargs = {})
triton_poi_fused_1 = async_compile.triton('triton_poi_fused_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*i64', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'AA6F4099AE74BF369D45388847321E195BFBEE5A86FCB9B181180FAA5E25C7E8', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 768}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp69 = tl.load(in_ptr0 + (x0), xmask)
    tmp70 = tl.load(in_ptr1 + (0))
    tmp71 = tl.broadcast_to(tmp70, [XBLOCK])
    tmp72 = tl.load(in_ptr2 + (0))
    tmp73 = tl.broadcast_to(tmp72, [XBLOCK])
    tmp0 = x0
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 60, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (((-2) + x0) % 3)
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp5 == tmp6
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp7
    tmp10 = tl.full([1], 2, tl.int32)
    tmp11 = tl.full([1], 0, tl.int32)
    tmp12 = tmp10 == tmp11
    tmp13 = 2 + 3*(triton_helpers.div_floor_integer((-2) + x0,  3))
    tmp14 = tl.full([1], 1, tl.int64)
    tmp15 = tmp13 >= tmp14
    tmp16 = tl.full([1], 60, tl.int64)
    tmp17 = tmp13 < tmp16
    tmp18 = tl.full([1], 0, tl.int64)
    tmp19 = tmp14 == tmp18
    tmp20 = tmp15 & tmp17
    tmp21 = tmp20 & tmp19
    tmp22 = tmp21 & tmp9
    tmp23 = tl.load(in_ptr0 + (1 + 3*(triton_helpers.div_floor_integer((-2) + x0,  3))), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.load(in_ptr1 + (0))
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK])
    tmp26 = tl.where(tmp22, tmp25, 0)
    tmp27 = tl.load(in_ptr2 + (0))
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK])
    tmp29 = tl.where(tmp22, tmp28, 0)
    tmp30 = tmp26 + tmp29
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp23 * tmp31
    tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
    tmp34 = tl.where(tmp22, tmp32, tmp33)
    tmp35 = tl.load(in_ptr0 + (2 + 3*(triton_helpers.div_floor_integer((-2) + x0,  3))), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.load(in_ptr1 + (0))
    tmp37 = tl.broadcast_to(tmp36, [XBLOCK])
    tmp38 = tl.where(tmp9, tmp37, 0)
    tmp39 = tl.load(in_ptr2 + (0))
    tmp40 = tl.broadcast_to(tmp39, [XBLOCK])
    tmp41 = tl.where(tmp9, tmp40, 0)
    tmp42 = tmp38 + tmp41
    tmp43 = tmp42.to(tl.float32)
    tmp44 = tmp35 * tmp43
    tmp45 = tl.where(tmp21, tmp34, tmp44)
    tmp46 = tl.where(tmp12, tmp45, tmp44)
    tmp47 = tl.full(tmp46.shape, 0.0, tmp46.dtype)
    tmp48 = tl.where(tmp9, tmp46, tmp47)
    tmp49 = tl.full([1], 0, tl.int32)
    tmp50 = tmp49 == tmp49
    tmp51 = tl.full([1], 1, tl.int64)
    tmp52 = tmp0 >= tmp51
    tmp53 = (((-1) + x0) % 3)
    tmp54 = tmp53 == tmp6
    tmp55 = tmp52 & tmp4
    tmp56 = tmp55 & tmp54
    tmp57 = tl.load(in_ptr0 + (1 + 3*(triton_helpers.div_floor_integer((-1) + x0,  3))), tmp56 & xmask, eviction_policy='evict_last', other=0.0)
    tmp58 = tl.load(in_ptr1 + (0))
    tmp59 = tl.broadcast_to(tmp58, [XBLOCK])
    tmp60 = tl.where(tmp56, tmp59, 0)
    tmp61 = tl.load(in_ptr2 + (0))
    tmp62 = tl.broadcast_to(tmp61, [XBLOCK])
    tmp63 = tl.where(tmp56, tmp62, 0)
    tmp64 = tmp60 + tmp63
    tmp65 = tmp64.to(tl.float32)
    tmp66 = tmp57 * tmp65
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp56, tmp66, tmp67)
    tmp74 = tmp71 + tmp73
    tmp75 = tmp74.to(tl.float32)
    tmp76 = tmp69 * tmp75
    tmp77 = tl.where(tmp56, tmp68, tmp76)
    tmp78 = tl.where(tmp50, tmp77, tmp76)
    tmp79 = tl.where(tmp9, tmp48, tmp78)
    tl.store(out_ptr0 + (x0), tmp79, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/sm/csmnotbln63yadoop3paki4jbeogso6tm7yy25ogaoetfgh56mky.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => slice_scatter_default_3
# Graph fragment:
#   %slice_scatter_default_3 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%select_int_3, %slice_176, 2, 2, 60, 3), kwargs = {})
triton_poi_fused_2 = async_compile.triton('triton_poi_fused_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*i64', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'AA6F4099AE74BF369D45388847321E195BFBEE5A86FCB9B181180FAA5E25C7E8', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 768}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp74 = tl.load(in_ptr0 + (x0), xmask)
    tmp75 = tl.load(in_ptr1 + (0))
    tmp76 = tl.broadcast_to(tmp75, [XBLOCK])
    tmp78 = tl.load(in_ptr2 + (0))
    tmp79 = tl.broadcast_to(tmp78, [XBLOCK])
    tmp0 = x0
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 60, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (((-2) + x0) % 3)
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp5 == tmp6
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp7
    tmp10 = tl.full([1], 2, tl.int32)
    tmp11 = tl.full([1], 0, tl.int32)
    tmp12 = tmp10 == tmp11
    tmp13 = 2 + 3*(triton_helpers.div_floor_integer((-2) + x0,  3))
    tmp14 = tl.full([1], 1, tl.int64)
    tmp15 = tmp13 >= tmp14
    tmp16 = tl.full([1], 60, tl.int64)
    tmp17 = tmp13 < tmp16
    tmp18 = tl.full([1], 0, tl.int64)
    tmp19 = tmp14 == tmp18
    tmp20 = tmp15 & tmp17
    tmp21 = tmp20 & tmp19
    tmp22 = tmp21 & tmp9
    tmp23 = tl.load(in_ptr0 + (1 + 3*(triton_helpers.div_floor_integer((-2) + x0,  3))), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.load(in_ptr1 + (0))
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK])
    tmp26 = tl.where(tmp22, tmp25, 0)
    tmp27 = tl.full([1], 1, tl.int64)
    tmp28 = tmp26 + tmp27
    tmp29 = tl.load(in_ptr2 + (0))
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK])
    tmp31 = tl.where(tmp22, tmp30, 0)
    tmp32 = tmp28 + tmp31
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tmp23 * tmp33
    tmp35 = tl.full(tmp34.shape, 0.0, tmp34.dtype)
    tmp36 = tl.where(tmp22, tmp34, tmp35)
    tmp37 = tl.load(in_ptr0 + (2 + 3*(triton_helpers.div_floor_integer((-2) + x0,  3))), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.load(in_ptr1 + (0))
    tmp39 = tl.broadcast_to(tmp38, [XBLOCK])
    tmp40 = tl.where(tmp9, tmp39, 0)
    tmp41 = tmp40 + tmp14
    tmp42 = tl.load(in_ptr2 + (0))
    tmp43 = tl.broadcast_to(tmp42, [XBLOCK])
    tmp44 = tl.where(tmp9, tmp43, 0)
    tmp45 = tmp41 + tmp44
    tmp46 = tmp45.to(tl.float32)
    tmp47 = tmp37 * tmp46
    tmp48 = tl.where(tmp21, tmp36, tmp47)
    tmp49 = tl.where(tmp12, tmp48, tmp47)
    tmp50 = tl.full(tmp49.shape, 0.0, tmp49.dtype)
    tmp51 = tl.where(tmp9, tmp49, tmp50)
    tmp52 = tl.full([1], 0, tl.int32)
    tmp53 = tmp52 == tmp52
    tmp54 = tl.full([1], 1, tl.int64)
    tmp55 = tmp0 >= tmp54
    tmp56 = (((-1) + x0) % 3)
    tmp57 = tmp56 == tmp6
    tmp58 = tmp55 & tmp4
    tmp59 = tmp58 & tmp57
    tmp60 = tl.load(in_ptr0 + (1 + 3*(triton_helpers.div_floor_integer((-1) + x0,  3))), tmp59 & xmask, eviction_policy='evict_last', other=0.0)
    tmp61 = tl.load(in_ptr1 + (0))
    tmp62 = tl.broadcast_to(tmp61, [XBLOCK])
    tmp63 = tl.where(tmp59, tmp62, 0)
    tmp64 = tl.full([1], 1, tl.int64)
    tmp65 = tmp63 + tmp64
    tmp66 = tl.load(in_ptr2 + (0))
    tmp67 = tl.broadcast_to(tmp66, [XBLOCK])
    tmp68 = tl.where(tmp59, tmp67, 0)
    tmp69 = tmp65 + tmp68
    tmp70 = tmp69.to(tl.float32)
    tmp71 = tmp60 * tmp70
    tmp72 = tl.full(tmp71.shape, 0.0, tmp71.dtype)
    tmp73 = tl.where(tmp59, tmp71, tmp72)
    tmp77 = tmp76 + tmp54
    tmp80 = tmp77 + tmp79
    tmp81 = tmp80.to(tl.float32)
    tmp82 = tmp74 * tmp81
    tmp83 = tl.where(tmp59, tmp73, tmp82)
    tmp84 = tl.where(tmp53, tmp83, tmp82)
    tmp85 = tl.where(tmp9, tmp51, tmp84)
    tl.store(out_ptr0 + (x0), tmp85, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/f7/cf7fzbc6vhfwyu4b6q5ga5dssryexprqc3p3vzqupubpf65j5vba.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => slice_scatter_default_5
# Graph fragment:
#   %slice_scatter_default_5 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%select_int_5, %slice_348, 2, 2, 60, 3), kwargs = {})
triton_poi_fused_3 = async_compile.triton('triton_poi_fused_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*i64', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'AA6F4099AE74BF369D45388847321E195BFBEE5A86FCB9B181180FAA5E25C7E8', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 768}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_3(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp77 = tl.load(in_ptr0 + (x0), xmask)
    tmp78 = tl.load(in_ptr1 + (0))
    tmp79 = tl.broadcast_to(tmp78, [XBLOCK])
    tmp82 = tl.load(in_ptr2 + (0))
    tmp83 = tl.broadcast_to(tmp82, [XBLOCK])
    tmp0 = x0
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 60, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (((-2) + x0) % 3)
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp5 == tmp6
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp7
    tmp10 = tl.full([1], 2, tl.int32)
    tmp11 = tl.full([1], 0, tl.int32)
    tmp12 = tmp10 == tmp11
    tmp13 = 2 + 3*(triton_helpers.div_floor_integer((-2) + x0,  3))
    tmp14 = tl.full([1], 1, tl.int64)
    tmp15 = tmp13 >= tmp14
    tmp16 = tl.full([1], 60, tl.int64)
    tmp17 = tmp13 < tmp16
    tmp18 = tl.full([1], 0, tl.int64)
    tmp19 = tmp14 == tmp18
    tmp20 = tmp15 & tmp17
    tmp21 = tmp20 & tmp19
    tmp22 = tmp21 & tmp9
    tmp23 = tl.load(in_ptr0 + (1 + 3*(triton_helpers.div_floor_integer((-2) + x0,  3))), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.load(in_ptr1 + (0))
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK])
    tmp26 = tl.where(tmp22, tmp25, 0)
    tmp27 = tl.full([1], 1, tl.int64)
    tmp28 = tmp26 + tmp27
    tmp29 = tmp28 + tmp27
    tmp30 = tl.load(in_ptr2 + (0))
    tmp31 = tl.broadcast_to(tmp30, [XBLOCK])
    tmp32 = tl.where(tmp22, tmp31, 0)
    tmp33 = tmp29 + tmp32
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tmp23 * tmp34
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp22, tmp35, tmp36)
    tmp38 = tl.load(in_ptr0 + (2 + 3*(triton_helpers.div_floor_integer((-2) + x0,  3))), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp39 = tl.load(in_ptr1 + (0))
    tmp40 = tl.broadcast_to(tmp39, [XBLOCK])
    tmp41 = tl.where(tmp9, tmp40, 0)
    tmp42 = tmp41 + tmp14
    tmp43 = tmp42 + tmp14
    tmp44 = tl.load(in_ptr2 + (0))
    tmp45 = tl.broadcast_to(tmp44, [XBLOCK])
    tmp46 = tl.where(tmp9, tmp45, 0)
    tmp47 = tmp43 + tmp46
    tmp48 = tmp47.to(tl.float32)
    tmp49 = tmp38 * tmp48
    tmp50 = tl.where(tmp21, tmp37, tmp49)
    tmp51 = tl.where(tmp12, tmp50, tmp49)
    tmp52 = tl.full(tmp51.shape, 0.0, tmp51.dtype)
    tmp53 = tl.where(tmp9, tmp51, tmp52)
    tmp54 = tl.full([1], 0, tl.int32)
    tmp55 = tmp54 == tmp54
    tmp56 = tl.full([1], 1, tl.int64)
    tmp57 = tmp0 >= tmp56
    tmp58 = (((-1) + x0) % 3)
    tmp59 = tmp58 == tmp6
    tmp60 = tmp57 & tmp4
    tmp61 = tmp60 & tmp59
    tmp62 = tl.load(in_ptr0 + (1 + 3*(triton_helpers.div_floor_integer((-1) + x0,  3))), tmp61 & xmask, eviction_policy='evict_last', other=0.0)
    tmp63 = tl.load(in_ptr1 + (0))
    tmp64 = tl.broadcast_to(tmp63, [XBLOCK])
    tmp65 = tl.where(tmp61, tmp64, 0)
    tmp66 = tl.full([1], 1, tl.int64)
    tmp67 = tmp65 + tmp66
    tmp68 = tmp67 + tmp66
    tmp69 = tl.load(in_ptr2 + (0))
    tmp70 = tl.broadcast_to(tmp69, [XBLOCK])
    tmp71 = tl.where(tmp61, tmp70, 0)
    tmp72 = tmp68 + tmp71
    tmp73 = tmp72.to(tl.float32)
    tmp74 = tmp62 * tmp73
    tmp75 = tl.full(tmp74.shape, 0.0, tmp74.dtype)
    tmp76 = tl.where(tmp61, tmp74, tmp75)
    tmp80 = tmp79 + tmp56
    tmp81 = tmp80 + tmp56
    tmp84 = tmp81 + tmp83
    tmp85 = tmp84.to(tl.float32)
    tmp86 = tmp77 * tmp85
    tmp87 = tl.where(tmp61, tmp76, tmp86)
    tmp88 = tl.where(tmp55, tmp87, tmp86)
    tmp89 = tl.where(tmp9, tmp53, tmp88)
    tl.store(out_ptr0 + (x0), tmp89, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/h6/ch6tqb6dtrcejou3jvildswqv5xclnnz4bljb7endawx5be2oefv.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => slice_scatter_default_7
# Graph fragment:
#   %slice_scatter_default_7 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%select_int_7, %slice_520, 2, 2, 60, 3), kwargs = {})
triton_poi_fused_4 = async_compile.triton('triton_poi_fused_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*i64', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'AA6F4099AE74BF369D45388847321E195BFBEE5A86FCB9B181180FAA5E25C7E8', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 768}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_4(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp80 = tl.load(in_ptr0 + (x0), xmask)
    tmp81 = tl.load(in_ptr1 + (0))
    tmp82 = tl.broadcast_to(tmp81, [XBLOCK])
    tmp86 = tl.load(in_ptr2 + (0))
    tmp87 = tl.broadcast_to(tmp86, [XBLOCK])
    tmp0 = x0
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 60, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (((-2) + x0) % 3)
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp5 == tmp6
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp7
    tmp10 = tl.full([1], 2, tl.int32)
    tmp11 = tl.full([1], 0, tl.int32)
    tmp12 = tmp10 == tmp11
    tmp13 = 2 + 3*(triton_helpers.div_floor_integer((-2) + x0,  3))
    tmp14 = tl.full([1], 1, tl.int64)
    tmp15 = tmp13 >= tmp14
    tmp16 = tl.full([1], 60, tl.int64)
    tmp17 = tmp13 < tmp16
    tmp18 = tl.full([1], 0, tl.int64)
    tmp19 = tmp14 == tmp18
    tmp20 = tmp15 & tmp17
    tmp21 = tmp20 & tmp19
    tmp22 = tmp21 & tmp9
    tmp23 = tl.load(in_ptr0 + (1 + 3*(triton_helpers.div_floor_integer((-2) + x0,  3))), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.load(in_ptr1 + (0))
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK])
    tmp26 = tl.where(tmp22, tmp25, 0)
    tmp27 = tl.full([1], 1, tl.int64)
    tmp28 = tmp26 + tmp27
    tmp29 = tmp28 + tmp27
    tmp30 = tmp29 + tmp27
    tmp31 = tl.load(in_ptr2 + (0))
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK])
    tmp33 = tl.where(tmp22, tmp32, 0)
    tmp34 = tmp30 + tmp33
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tmp23 * tmp35
    tmp37 = tl.full(tmp36.shape, 0.0, tmp36.dtype)
    tmp38 = tl.where(tmp22, tmp36, tmp37)
    tmp39 = tl.load(in_ptr0 + (2 + 3*(triton_helpers.div_floor_integer((-2) + x0,  3))), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp40 = tl.load(in_ptr1 + (0))
    tmp41 = tl.broadcast_to(tmp40, [XBLOCK])
    tmp42 = tl.where(tmp9, tmp41, 0)
    tmp43 = tmp42 + tmp14
    tmp44 = tmp43 + tmp14
    tmp45 = tmp44 + tmp14
    tmp46 = tl.load(in_ptr2 + (0))
    tmp47 = tl.broadcast_to(tmp46, [XBLOCK])
    tmp48 = tl.where(tmp9, tmp47, 0)
    tmp49 = tmp45 + tmp48
    tmp50 = tmp49.to(tl.float32)
    tmp51 = tmp39 * tmp50
    tmp52 = tl.where(tmp21, tmp38, tmp51)
    tmp53 = tl.where(tmp12, tmp52, tmp51)
    tmp54 = tl.full(tmp53.shape, 0.0, tmp53.dtype)
    tmp55 = tl.where(tmp9, tmp53, tmp54)
    tmp56 = tl.full([1], 0, tl.int32)
    tmp57 = tmp56 == tmp56
    tmp58 = tl.full([1], 1, tl.int64)
    tmp59 = tmp0 >= tmp58
    tmp60 = (((-1) + x0) % 3)
    tmp61 = tmp60 == tmp6
    tmp62 = tmp59 & tmp4
    tmp63 = tmp62 & tmp61
    tmp64 = tl.load(in_ptr0 + (1 + 3*(triton_helpers.div_floor_integer((-1) + x0,  3))), tmp63 & xmask, eviction_policy='evict_last', other=0.0)
    tmp65 = tl.load(in_ptr1 + (0))
    tmp66 = tl.broadcast_to(tmp65, [XBLOCK])
    tmp67 = tl.where(tmp63, tmp66, 0)
    tmp68 = tl.full([1], 1, tl.int64)
    tmp69 = tmp67 + tmp68
    tmp70 = tmp69 + tmp68
    tmp71 = tmp70 + tmp68
    tmp72 = tl.load(in_ptr2 + (0))
    tmp73 = tl.broadcast_to(tmp72, [XBLOCK])
    tmp74 = tl.where(tmp63, tmp73, 0)
    tmp75 = tmp71 + tmp74
    tmp76 = tmp75.to(tl.float32)
    tmp77 = tmp64 * tmp76
    tmp78 = tl.full(tmp77.shape, 0.0, tmp77.dtype)
    tmp79 = tl.where(tmp63, tmp77, tmp78)
    tmp83 = tmp82 + tmp58
    tmp84 = tmp83 + tmp58
    tmp85 = tmp84 + tmp58
    tmp88 = tmp85 + tmp87
    tmp89 = tmp88.to(tl.float32)
    tmp90 = tmp80 * tmp89
    tmp91 = tl.where(tmp63, tmp79, tmp90)
    tmp92 = tl.where(tmp57, tmp91, tmp90)
    tmp93 = tl.where(tmp9, tmp55, tmp92)
    tl.store(out_ptr0 + (x0), tmp93, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/m7/cm76r6bbs5auascjr6eqesz3vfv4n6luexdpxfwd2i7oxeufhahk.py
# Topologically Sorted Source Nodes: [mul_5, sum_2], Original ATen: [aten.mul, aten.sum]
# Source node to ATen node mapping:
#   mul_5 => mul_5
#   sum_2 => sum_2
# Graph fragment:
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_14, %unsqueeze_15), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_5, [1]), kwargs = {})
triton_red_fused_mul_sum_5 = async_compile.triton('triton_red_fused_mul_sum_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 2048, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*i64', 'in_ptr2': '*fp16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sum_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': 'AA6F4099AE74BF369D45388847321E195BFBEE5A86FCB9B181180FAA5E25C7E8', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_red_fused_mul_sum_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2048
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp10 = tl.load(in_ptr3 + (0))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    x0 = xindex
    _tmp25 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp21 = tl.load(in_ptr4 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.full([XBLOCK, R0_BLOCK], 151936, tl.int32)
        tmp4 = tmp2 + tmp3
        tmp5 = tmp2 < 0
        tmp6 = tl.where(tmp5, tmp4, tmp2)
        tl.device_assert((0 <= tmp6) & (tmp6 < 151936), "index out of bounds: 0 <= tmp6 < 151936")
        tmp8 = tl.load(in_ptr2 + (r0_1 + 2048*tmp6), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp9 = tmp8.to(tl.float32)
        tmp12 = 2048.0
        tmp13 = (tmp11 / tmp12)
        tmp14 = 1e-06
        tmp15 = tmp13 + tmp14
        tmp16 = libdevice.rsqrt(tmp15)
        tmp17 = tmp9 * tmp16
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tmp0 * tmp18
        tmp20 = tmp19.to(tl.float32)
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp20 * tmp22
        tmp24 = tl.broadcast_to(tmp23, [XBLOCK, R0_BLOCK])
        tmp26 = _tmp25 + tmp24
        _tmp25 = tl.where(r0_mask & xmask, tmp26, _tmp25)
    tmp25 = tl.sum(_tmp25, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp25, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/bg/cbgwgnkwsoitvidokcbq6ou6gnwvuvcnpnvkw6jctyqwpvbxwjl7.py
# Topologically Sorted Source Nodes: [mul_8, sum_3, mul_11, sum_4, index_put_1], Original ATen: [aten.mul, aten.sum, aten.index_put]
# Source node to ATen node mapping:
#   index_put_1 => index_put_1
#   mul_11 => mul_11
#   mul_8 => mul_8
#   sum_3 => sum_3
#   sum_4 => sum_4
# Graph fragment:
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_16, %unsqueeze_17), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_8, [1]), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_18, %unsqueeze_19), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_11, [1]), kwargs = {})
#   %index_put_1 : [num_users=2] = call_function[target=torch.ops.aten.index_put_.default](args = (%arg291_1, [None, None, %arg1_1], %permute_13), kwargs = {})
triton_red_fused_index_put_mul_sum_6 = async_compile.triton('triton_red_fused_index_put_mul_sum_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*i64', 'in_ptr2': '*fp16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp16', 'in_ptr5': '*fp16', 'in_ptr6': '*i64', 'out_ptr0': '*fp32', 'out_ptr2': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_index_put_mul_sum_6', 'mutated_arg_names': ['out_ptr2'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 2, 'backend_hash': 'AA6F4099AE74BF369D45388847321E195BFBEE5A86FCB9B181180FAA5E25C7E8', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_red_fused_index_put_mul_sum_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp10 = tl.load(in_ptr3 + (0))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    x0 = xindex
    _tmp25 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp31 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp21 = tl.load(in_ptr4 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp27 = tl.load(in_ptr5 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.full([XBLOCK, R0_BLOCK], 151936, tl.int32)
        tmp4 = tmp2 + tmp3
        tmp5 = tmp2 < 0
        tmp6 = tl.where(tmp5, tmp4, tmp2)
        tl.device_assert((0 <= tmp6) & (tmp6 < 151936), "index out of bounds: 0 <= tmp6 < 151936")
        tmp8 = tl.load(in_ptr2 + (r0_1 + 2048*tmp6), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp9 = tmp8.to(tl.float32)
        tmp12 = 2048.0
        tmp13 = (tmp11 / tmp12)
        tmp14 = 1e-06
        tmp15 = tmp13 + tmp14
        tmp16 = libdevice.rsqrt(tmp15)
        tmp17 = tmp9 * tmp16
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tmp0 * tmp18
        tmp20 = tmp19.to(tl.float32)
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp20 * tmp22
        tmp24 = tl.broadcast_to(tmp23, [XBLOCK, R0_BLOCK])
        tmp26 = _tmp25 + tmp24
        _tmp25 = tl.where(r0_mask & xmask, tmp26, _tmp25)
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tmp20 * tmp28
        tmp30 = tl.broadcast_to(tmp29, [XBLOCK, R0_BLOCK])
        tmp32 = _tmp31 + tmp30
        _tmp31 = tl.where(r0_mask & xmask, tmp32, _tmp31)
    tmp25 = tl.sum(_tmp25, 1)[:, None]
    tmp31 = tl.sum(_tmp31, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp25, xmask)
    x2 = (xindex % 128)
    x3 = xindex // 128
    tmp33 = tl.load(in_ptr6 + (0))
    tmp34 = tl.broadcast_to(tmp33, [XBLOCK, 1])
    tmp35 = tl.full([XBLOCK, 1], 960, tl.int32)
    tmp36 = tmp34 + tmp35
    tmp37 = tmp34 < 0
    tmp38 = tl.where(tmp37, tmp36, tmp34)
    tl.device_assert((0 <= tmp38) & (tmp38 < 960), "index out of bounds: 0 <= tmp38 < 960")
    tmp40 = tmp31.to(tl.float32)
    tl.store(out_ptr2 + (x2 + 128*tmp38 + 122880*x3), tmp40, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/s2/cs2amvtmuxo37je5pi6pbzqq6jf2wz7reskrhfr5pvwdb2pjimda.py
# Topologically Sorted Source Nodes: [_generalized_scatter_3], Original ATen: []
# Source node to ATen node mapping:
#   _generalized_scatter_3 => select_scatter_default_1
# Graph fragment:
#   %select_scatter_default_1 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%permute_5, %slice_scatter_default_1, 0, 0), kwargs = {})
triton_poi_fused_7 = async_compile.triton('triton_poi_fused_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*i64', 'in_ptr3': '*i64', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'AA6F4099AE74BF369D45388847321E195BFBEE5A86FCB9B181180FAA5E25C7E8', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 2048}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 64
    x0 = (xindex % 64)
    x2 = xindex
    tmp3 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr2 + (0))
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK])
    tmp29 = tl.load(in_ptr3 + (0))
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK])
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = x0
    tmp5 = tl.full([1], 1, tl.int64)
    tmp6 = tmp4 >= tmp5
    tmp7 = tl.full([1], 60, tl.int64)
    tmp8 = tmp4 < tmp7
    tmp9 = (((-1) + x0) % 3)
    tmp10 = tl.full([1], 0, tl.int64)
    tmp11 = tmp9 == tmp10
    tmp12 = tmp6 & tmp8
    tmp13 = tmp12 & tmp11
    tmp14 = tl.load(in_ptr1 + (1 + 3*(triton_helpers.div_floor_integer((-1) + x0,  3))), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(in_ptr2 + (0))
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK])
    tmp17 = tl.where(tmp13, tmp16, 0)
    tmp18 = tl.load(in_ptr3 + (0))
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK])
    tmp20 = tl.where(tmp13, tmp19, 0)
    tmp21 = tmp17 + tmp20
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp14 * tmp22
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp13, tmp23, tmp24)
    tmp31 = tmp28 + tmp30
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tmp26 * tmp32
    tmp34 = tl.where(tmp13, tmp25, tmp33)
    tmp35 = tl.where(tmp2, tmp34, tmp33)
    tmp36 = tl.where(tmp2, tmp3, tmp35)
    tl.store(out_ptr0 + (x2), tmp36, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/ot/cothueqrkn65adzmtsdxdq2v2ajev6htswqjuptoycb4xowys4l7.py
# Topologically Sorted Source Nodes: [_generalized_scatter_7], Original ATen: []
# Source node to ATen node mapping:
#   _generalized_scatter_7 => select_scatter_default_3
# Graph fragment:
#   %select_scatter_default_3 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%permute_322, %slice_scatter_default_3, 0, 0), kwargs = {})
triton_poi_fused_8 = async_compile.triton('triton_poi_fused_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*i64', 'in_ptr3': '*i64', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'AA6F4099AE74BF369D45388847321E195BFBEE5A86FCB9B181180FAA5E25C7E8', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 2048}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 64
    x0 = (xindex % 64)
    x2 = xindex
    tmp3 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr2 + (0))
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK])
    tmp32 = tl.load(in_ptr3 + (0))
    tmp33 = tl.broadcast_to(tmp32, [XBLOCK])
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = x0
    tmp5 = tl.full([1], 1, tl.int64)
    tmp6 = tmp4 >= tmp5
    tmp7 = tl.full([1], 60, tl.int64)
    tmp8 = tmp4 < tmp7
    tmp9 = (((-1) + x0) % 3)
    tmp10 = tl.full([1], 0, tl.int64)
    tmp11 = tmp9 == tmp10
    tmp12 = tmp6 & tmp8
    tmp13 = tmp12 & tmp11
    tmp14 = tl.load(in_ptr1 + (1 + 3*(triton_helpers.div_floor_integer((-1) + x0,  3))), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(in_ptr2 + (0))
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK])
    tmp17 = tl.where(tmp13, tmp16, 0)
    tmp18 = tl.full([1], 1, tl.int64)
    tmp19 = tmp17 + tmp18
    tmp20 = tl.load(in_ptr3 + (0))
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK])
    tmp22 = tl.where(tmp13, tmp21, 0)
    tmp23 = tmp19 + tmp22
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp14 * tmp24
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp13, tmp25, tmp26)
    tmp31 = tmp30 + tmp5
    tmp34 = tmp31 + tmp33
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tmp28 * tmp35
    tmp37 = tl.where(tmp13, tmp27, tmp36)
    tmp38 = tl.where(tmp2, tmp37, tmp36)
    tmp39 = tl.where(tmp2, tmp3, tmp38)
    tl.store(out_ptr0 + (x2), tmp39, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/th/cth2njqcspvy7us2ncyfa2ziuuzjpezwoowy5esetxwafsickung.py
# Topologically Sorted Source Nodes: [_generalized_scatter_11], Original ATen: []
# Source node to ATen node mapping:
#   _generalized_scatter_11 => select_scatter_default_5
# Graph fragment:
#   %select_scatter_default_5 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%permute_639, %slice_scatter_default_5, 0, 0), kwargs = {})
triton_poi_fused_9 = async_compile.triton('triton_poi_fused_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*i64', 'in_ptr3': '*i64', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'AA6F4099AE74BF369D45388847321E195BFBEE5A86FCB9B181180FAA5E25C7E8', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 2048}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 64
    x0 = (xindex % 64)
    x2 = xindex
    tmp3 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr2 + (0))
    tmp31 = tl.broadcast_to(tmp30, [XBLOCK])
    tmp34 = tl.load(in_ptr3 + (0))
    tmp35 = tl.broadcast_to(tmp34, [XBLOCK])
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = x0
    tmp5 = tl.full([1], 1, tl.int64)
    tmp6 = tmp4 >= tmp5
    tmp7 = tl.full([1], 60, tl.int64)
    tmp8 = tmp4 < tmp7
    tmp9 = (((-1) + x0) % 3)
    tmp10 = tl.full([1], 0, tl.int64)
    tmp11 = tmp9 == tmp10
    tmp12 = tmp6 & tmp8
    tmp13 = tmp12 & tmp11
    tmp14 = tl.load(in_ptr1 + (1 + 3*(triton_helpers.div_floor_integer((-1) + x0,  3))), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(in_ptr2 + (0))
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK])
    tmp17 = tl.where(tmp13, tmp16, 0)
    tmp18 = tl.full([1], 1, tl.int64)
    tmp19 = tmp17 + tmp18
    tmp20 = tmp19 + tmp18
    tmp21 = tl.load(in_ptr3 + (0))
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK])
    tmp23 = tl.where(tmp13, tmp22, 0)
    tmp24 = tmp20 + tmp23
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp14 * tmp25
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp13, tmp26, tmp27)
    tmp32 = tmp31 + tmp5
    tmp33 = tmp32 + tmp5
    tmp36 = tmp33 + tmp35
    tmp37 = tmp36.to(tl.float32)
    tmp38 = tmp29 * tmp37
    tmp39 = tl.where(tmp13, tmp28, tmp38)
    tmp40 = tl.where(tmp2, tmp39, tmp38)
    tmp41 = tl.where(tmp2, tmp3, tmp40)
    tl.store(out_ptr0 + (x2), tmp41, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/27/c27rz3q5xe6z4hu5vuj3mphan3xibalqzgzlwtlw7kcyz3dtlfwf.py
# Topologically Sorted Source Nodes: [_generalized_scatter_15], Original ATen: []
# Source node to ATen node mapping:
#   _generalized_scatter_15 => select_scatter_default_7
# Graph fragment:
#   %select_scatter_default_7 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%permute_956, %slice_scatter_default_7, 0, 0), kwargs = {})
triton_poi_fused_10 = async_compile.triton('triton_poi_fused_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*i64', 'in_ptr3': '*i64', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'AA6F4099AE74BF369D45388847321E195BFBEE5A86FCB9B181180FAA5E25C7E8', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 2048}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 64
    x0 = (xindex % 64)
    x2 = xindex
    tmp3 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr2 + (0))
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK])
    tmp36 = tl.load(in_ptr3 + (0))
    tmp37 = tl.broadcast_to(tmp36, [XBLOCK])
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = x0
    tmp5 = tl.full([1], 1, tl.int64)
    tmp6 = tmp4 >= tmp5
    tmp7 = tl.full([1], 60, tl.int64)
    tmp8 = tmp4 < tmp7
    tmp9 = (((-1) + x0) % 3)
    tmp10 = tl.full([1], 0, tl.int64)
    tmp11 = tmp9 == tmp10
    tmp12 = tmp6 & tmp8
    tmp13 = tmp12 & tmp11
    tmp14 = tl.load(in_ptr1 + (1 + 3*(triton_helpers.div_floor_integer((-1) + x0,  3))), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(in_ptr2 + (0))
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK])
    tmp17 = tl.where(tmp13, tmp16, 0)
    tmp18 = tl.full([1], 1, tl.int64)
    tmp19 = tmp17 + tmp18
    tmp20 = tmp19 + tmp18
    tmp21 = tmp20 + tmp18
    tmp22 = tl.load(in_ptr3 + (0))
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK])
    tmp24 = tl.where(tmp13, tmp23, 0)
    tmp25 = tmp21 + tmp24
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp14 * tmp26
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp13, tmp27, tmp28)
    tmp33 = tmp32 + tmp5
    tmp34 = tmp33 + tmp5
    tmp35 = tmp34 + tmp5
    tmp38 = tmp35 + tmp37
    tmp39 = tmp38.to(tl.float32)
    tmp40 = tmp30 * tmp39
    tmp41 = tl.where(tmp13, tmp29, tmp40)
    tmp42 = tl.where(tmp2, tmp41, tmp40)
    tmp43 = tl.where(tmp2, tmp3, tmp42)
    tl.store(out_ptr0 + (x2), tmp43, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/xh/cxhl4jmg7kji5oswaqy3paghfivt4yelsbtgcqzodikcj6445rhi.py
# Topologically Sorted Source Nodes: [convert_element_type_8, pow_2, mean_1, mul_12, cat, mul_13, add_5, convert_element_type_18, mul_16], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
# Source node to ATen node mapping:
#   add_5 => add_5
#   cat => cat
#   convert_element_type_18 => convert_element_type_18
#   convert_element_type_8 => convert_element_type_8
#   mean_1 => mean_1
#   mul_12 => mul_12
#   mul_13 => mul_13
#   mul_16 => mul_16
#   pow_2 => pow_2
# Graph fragment:
#   %convert_element_type_8 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_12, torch.float32), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_8, 2), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_2, [-1], True), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_9, %unsqueeze_20), kwargs = {})
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%neg, %slice_5], -1), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat, %unsqueeze_21), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_12, %mul_13), kwargs = {})
#   %convert_element_type_18 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_5, torch.float32), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%convert_element_type_18, 0.29730177875068026), kwargs = {})
triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11 = async_compile.triton('triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp16', 'in_ptr2': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 1, 'backend_hash': 'AA6F4099AE74BF369D45388847321E195BFBEE5A86FCB9B181180FAA5E25C7E8', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 0, 'r0_': 42240}}
)
@triton.jit
def triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 128*x0), xmask, other=0.0)
    tmp46 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last').to(tl.float32)
    tmp55 = tl.load(in_ptr2 + ((r0_1 % 64)), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp2 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp6 = tl.where(xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = r0_1
    tmp9 = tl.full([1, 1], 0, tl.int64)
    tmp10 = tmp8 >= tmp9
    tmp11 = tl.full([1, 1], 64, tl.int64)
    tmp12 = tmp8 < tmp11
    tmp13 = tl.load(in_ptr1 + (tl.broadcast_to(64 + (r0_1), [XBLOCK, R0_BLOCK])), tmp12 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp14 = tl.load(in_ptr0 + (64 + 128*x0 + (r0_1)), tmp12 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = 128.0
    tmp18 = (tmp7 / tmp17)
    tmp19 = 1e-06
    tmp20 = tmp18 + tmp19
    tmp21 = libdevice.rsqrt(tmp20)
    tmp22 = tmp16 * tmp21
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp13 * tmp23
    tmp25 = -tmp24
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp12, tmp25, tmp26)
    tmp28 = tmp8 >= tmp11
    tmp29 = tl.full([1, 1], 128, tl.int64)
    tmp30 = tmp8 < tmp29
    tmp31 = tl.load(in_ptr1 + (tl.broadcast_to((-64) + r0_1, [XBLOCK, R0_BLOCK])), tmp28 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp32 = tl.load(in_ptr0 + (128*x0 + ((-64) + r0_1)), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tmp33.to(tl.float32)
    tmp35 = 128.0
    tmp36 = (tmp7 / tmp35)
    tmp37 = 1e-06
    tmp38 = tmp36 + tmp37
    tmp39 = libdevice.rsqrt(tmp38)
    tmp40 = tmp34 * tmp39
    tmp41 = tmp40.to(tl.float32)
    tmp42 = tmp31 * tmp41
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp28, tmp42, tmp43)
    tmp45 = tl.where(tmp12, tmp27, tmp44)
    tmp47 = 128.0
    tmp48 = (tmp7 / tmp47)
    tmp49 = 1e-06
    tmp50 = tmp48 + tmp49
    tmp51 = libdevice.rsqrt(tmp50)
    tmp52 = tmp2 * tmp51
    tmp53 = tmp52.to(tl.float32)
    tmp54 = tmp46 * tmp53
    tmp56 = tl_math.cos(tmp55)
    tmp57 = 1.0
    tmp58 = tmp56 * tmp57
    tmp59 = tmp58.to(tl.float32)
    tmp60 = tmp54 * tmp59
    tmp61 = tl_math.sin(tmp55)
    tmp62 = tmp61 * tmp57
    tmp63 = tmp62.to(tl.float32)
    tmp64 = tmp45 * tmp63
    tmp65 = tmp60 + tmp64
    tmp66 = tmp65.to(tl.float32)
    tmp67 = 0.29730177875068026
    tmp68 = tmp66 * tmp67
    tl.store(out_ptr2 + (r0_1 + 128*x0), tmp68, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/zw/czwusfu75jfrpf2f3gghh34o7wgoulnyy7cfldxb5kjcum6ejvpo.py
# Topologically Sorted Source Nodes: [convert_element_type_13, pow_3, mean_2, mul_14, cat_1, mul_15, add_6, index_put], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
# Source node to ATen node mapping:
#   add_6 => add_6
#   cat_1 => cat_1
#   convert_element_type_13 => convert_element_type_13
#   index_put => index_put
#   mean_2 => mean_2
#   mul_14 => mul_14
#   mul_15 => mul_15
#   pow_3 => pow_3
# Graph fragment:
#   %convert_element_type_13 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_15, torch.float32), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_13, 2), kwargs = {})
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_3, [-1], True), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_11, %unsqueeze_20), kwargs = {})
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%neg_1, %slice_7], -1), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat_1, %unsqueeze_21), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %mul_15), kwargs = {})
#   %index_put : [num_users=2] = call_function[target=torch.ops.aten.index_put_.default](args = (%arg290_1, [None, None, %arg1_1], %add_6), kwargs = {})
triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_12 = async_compile.triton('triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp16', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'out_ptr2': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_12', 'mutated_arg_names': ['out_ptr2'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 1, 'backend_hash': 'AA6F4099AE74BF369D45388847321E195BFBEE5A86FCB9B181180FAA5E25C7E8', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 8
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 128*x0), xmask, other=0.0)
    tmp46 = tl.load(in_ptr2 + (0))
    tmp47 = tl.broadcast_to(tmp46, [XBLOCK, R0_BLOCK])
    tmp53 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last').to(tl.float32)
    tmp62 = tl.load(in_ptr3 + ((r0_1 % 64)), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp2 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp6 = tl.where(xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = r0_1
    tmp9 = tl.full([1, 1], 0, tl.int64)
    tmp10 = tmp8 >= tmp9
    tmp11 = tl.full([1, 1], 64, tl.int64)
    tmp12 = tmp8 < tmp11
    tmp13 = tl.load(in_ptr1 + (tl.broadcast_to(64 + (r0_1), [XBLOCK, R0_BLOCK])), tmp12 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp14 = tl.load(in_ptr0 + (64 + 128*x0 + (r0_1)), tmp12 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = 128.0
    tmp18 = (tmp7 / tmp17)
    tmp19 = 1e-06
    tmp20 = tmp18 + tmp19
    tmp21 = libdevice.rsqrt(tmp20)
    tmp22 = tmp16 * tmp21
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp13 * tmp23
    tmp25 = -tmp24
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp12, tmp25, tmp26)
    tmp28 = tmp8 >= tmp11
    tmp29 = tl.full([1, 1], 128, tl.int64)
    tmp30 = tmp8 < tmp29
    tmp31 = tl.load(in_ptr1 + (tl.broadcast_to((-64) + r0_1, [XBLOCK, R0_BLOCK])), tmp28 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp32 = tl.load(in_ptr0 + (128*x0 + ((-64) + r0_1)), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tmp33.to(tl.float32)
    tmp35 = 128.0
    tmp36 = (tmp7 / tmp35)
    tmp37 = 1e-06
    tmp38 = tmp36 + tmp37
    tmp39 = libdevice.rsqrt(tmp38)
    tmp40 = tmp34 * tmp39
    tmp41 = tmp40.to(tl.float32)
    tmp42 = tmp31 * tmp41
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp28, tmp42, tmp43)
    tmp45 = tl.where(tmp12, tmp27, tmp44)
    tmp48 = tl.full([XBLOCK, R0_BLOCK], 960, tl.int32)
    tmp49 = tmp47 + tmp48
    tmp50 = tmp47 < 0
    tmp51 = tl.where(tmp50, tmp49, tmp47)
    tl.device_assert((0 <= tmp51) & (tmp51 < 960), "index out of bounds: 0 <= tmp51 < 960")
    tmp54 = 128.0
    tmp55 = (tmp7 / tmp54)
    tmp56 = 1e-06
    tmp57 = tmp55 + tmp56
    tmp58 = libdevice.rsqrt(tmp57)
    tmp59 = tmp2 * tmp58
    tmp60 = tmp59.to(tl.float32)
    tmp61 = tmp53 * tmp60
    tmp63 = tl_math.cos(tmp62)
    tmp64 = 1.0
    tmp65 = tmp63 * tmp64
    tmp66 = tmp65.to(tl.float32)
    tmp67 = tmp61 * tmp66
    tmp68 = tl_math.sin(tmp62)
    tmp69 = tmp68 * tmp64
    tmp70 = tmp69.to(tl.float32)
    tmp71 = tmp45 * tmp70
    tmp72 = tmp67 + tmp71
    tl.store(out_ptr2 + (r0_1 + 128*tmp51 + 122880*x0), tmp72, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/uj/cujthoxihzykgrro3fmpdwy7d5dm2hgoxshwgd5mmdup57ny2izl.py
# Topologically Sorted Source Nodes: [mul_18, sum_5], Original ATen: [aten.mul, aten.sum]
# Source node to ATen node mapping:
#   mul_18 => mul_18
#   sum_5 => sum_5
# Graph fragment:
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_24, %unsqueeze_25), kwargs = {})
#   %sum_5 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_18, [2]), kwargs = {})
triton_red_fused_mul_sum_13 = async_compile.triton('triton_red_fused_mul_sum_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r0_': 128},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sum_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'AA6F4099AE74BF369D45388847321E195BFBEE5A86FCB9B181180FAA5E25C7E8', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 122880, 'r0_': 8192}}
)
@triton.jit
def triton_red_fused_mul_sum_13(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 15360
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x1 = xindex // 960
    x0 = (xindex % 960)
    _tmp7 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_2 + 128*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_2 + 128*x0 + 122880*(x1 // 2)), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tmp1.to(tl.float32)
        tmp3 = 0.29730177875068026
        tmp4 = tmp2 * tmp3
        tmp5 = tmp0 * tmp4
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(r0_mask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/l4/cl46klebwxtwwgthhfdvmnx5kjhulwkdckax6qnkb3jg2qbqilw2.py
# Topologically Sorted Source Nodes: [full_default_1, full_default, where, add_7, eq, logical_not, any_1, logical_not_1, full_default_2, , sub, exp, div, where_1], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
# Source node to ATen node mapping:
#    => prepare_softmax_online_default_111
#   add_7 => add_7
#   any_1 => any_1
#   div => div
#   eq => eq
#   exp => exp_default_111
#   full_default => full_default
#   full_default_1 => full_default_1
#   full_default_2 => full_default_2
#   logical_not => logical_not
#   logical_not_1 => logical_not_1
#   sub => sub_tensor_111
#   where => where
#   where_1 => where_1
# Graph fragment:
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%expand_1, %full_default_1, %full_default), kwargs = {})
#   %add_7 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_23, %where), kwargs = {})
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%add_7, -inf), kwargs = {})
#   %logical_not : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%eq,), kwargs = {})
#   %any_1 : [num_users=1] = call_function[target=torch.ops.aten.any.dim](args = (%logical_not, -1, True), kwargs = {})
#   %logical_not_1 : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%any_1,), kwargs = {})
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 16, 1, 960], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %prepare_softmax_online_default_111 : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%add_7, -1), kwargs = {})
#   %sub_tensor_111 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_7, %getitem_222), kwargs = {})
#   %exp_default_111 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_111,), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_111, %getitem_223), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_not_1, %full_default_2, %div), kwargs = {})
triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_14 = async_compile.triton('triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 5, 'backend_hash': 'AA6F4099AE74BF369D45388847321E195BFBEE5A86FCB9B181180FAA5E25C7E8', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 0, 'r0_': 184320}}
)
@triton.jit
def triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_14(in_ptr0, in_ptr1, out_ptr3, xnumel, r0_numel):
    xnumel = 16
    XBLOCK: tl.constexpr = 1
    r0_numel = 960
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 960*x0), r0_mask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [R0_BLOCK])
    tmp3 = r0_1
    tmp4 = tmp3 <= tmp2
    tmp5 = 0.0
    tmp6 = float("-inf")
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp0 + tmp8
    tmp10 = tmp9 == tmp6
    tmp11 = tmp10 == 0
    tmp12 = tmp11.to(tl.int64)
    tmp13 = (tmp12 != 0)
    tmp14 = tl.broadcast_to(tmp13, [R0_BLOCK])
    tmp16 = tl.where(r0_mask, tmp14, False)
    tmp17 = triton_helpers.promote_to_tensor(triton_helpers.any(tmp16, 0))
    tmp18 = tl.broadcast_to(tmp9, [R0_BLOCK])
    tmp20 = tl.broadcast_to(tmp18, [R0_BLOCK])
    tmp22 = tl.where(r0_mask, tmp20, float("-inf"))
    tmp23 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp22, 0))
    tmp24 = tmp18 - tmp23
    tmp25 = tl_math.exp(tmp24)
    tmp26 = tl.broadcast_to(tmp25, [R0_BLOCK])
    tmp28 = tl.where(r0_mask, tmp26, 0)
    tmp29 = triton_helpers.promote_to_tensor(tl.sum(tmp28, 0))
    tmp30 = tmp17 == 0
    tmp31 = tmp9 - tmp23
    tmp32 = tl_math.exp(tmp31)
    tmp33 = (tmp32 / tmp29)
    tmp34 = tl.where(tmp30, tmp5, tmp33)
    tl.store(out_ptr3 + (r0_1 + 960*x0), tmp34, r0_mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/x3/cx3tsed3bso54oqdtlqmtywiqgb7xrio7pzhtfafqs2ax3dnxec3.py
# Topologically Sorted Source Nodes: [mul_19, sum_7], Original ATen: [aten.mul, aten.sum]
# Source node to ATen node mapping:
#   mul_19 => mul_19
#   sum_7 => sum_7
# Graph fragment:
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_26, %unsqueeze_27), kwargs = {})
#   %sum_7 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_19, [2]), kwargs = {})
triton_red_fused_mul_sum_15 = async_compile.triton('triton_red_fused_mul_sum_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sum_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'AA6F4099AE74BF369D45388847321E195BFBEE5A86FCB9B181180FAA5E25C7E8', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 131072, 'r0_': 61440}}
)
@triton.jit
def triton_red_fused_mul_sum_15(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 16384
    r0_numel = 120
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x4 = xindex // 128
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 8)
    x2 = xindex // 1024
    _tmp5 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x5 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_3 + 120*x4), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + 128*r0_3 + 15360*x1 + 122880*(x2 // 2)), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tmp1.to(tl.float32)
        tmp3 = tmp0 * tmp2
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(r0_mask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x5), tmp5, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/jr/cjr5hbt7bzbs46rnxssghosw6r7zxu3ld2djz75fyqlqjvbowf4v.py
# Topologically Sorted Source Nodes: [mul_19, sum_7], Original ATen: [aten.mul, aten.sum]
# Source node to ATen node mapping:
#   mul_19 => mul_19
#   sum_7 => sum_7
# Graph fragment:
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_26, %unsqueeze_27), kwargs = {})
#   %sum_7 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_19, [2]), kwargs = {})
triton_per_fused_mul_sum_16 = async_compile.triton('triton_per_fused_mul_sum_16', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2048, 'r0_': 8},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sum_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'AA6F4099AE74BF369D45388847321E195BFBEE5A86FCB9B181180FAA5E25C7E8', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 81920, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_mul_sum_16(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 2048
    r0_numel = 8
    R0_BLOCK: tl.constexpr = 8
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x0 = (xindex % 128)
    x1 = xindex // 128
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*r0_2 + 1024*x1), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/ks/ckskweq7jktbkfet5yog2h7i7cyo3ljhteiftq3agnf674vjiuoo.py
# Topologically Sorted Source Nodes: [mul_20, sum_8], Original ATen: [aten.mul, aten.sum]
# Source node to ATen node mapping:
#   mul_20 => mul_20
#   sum_8 => sum_8
# Graph fragment:
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_28, %unsqueeze_29), kwargs = {})
#   %sum_8 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_20, [1]), kwargs = {})
triton_red_fused_mul_sum_17 = async_compile.triton('triton_red_fused_mul_sum_17', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 2048, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sum_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'AA6F4099AE74BF369D45388847321E195BFBEE5A86FCB9B181180FAA5E25C7E8', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 16384, 'r0_': 8396800}}
)
@triton.jit
def triton_red_fused_mul_sum_17(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2048
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp7 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tmp1.to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tmp2 * tmp4
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(r0_mask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp7, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/pf/cpfgiuu5h2d4su5lp5qjc3gebkb6mhgan5drze7ps5p4esvsxacp.py
# Topologically Sorted Source Nodes: [embedding, add_8, convert_element_type_26, pow_4, mean_3, convert_element_type_28], Original ATen: [aten.embedding, aten.add, prims.convert_element_type, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   add_8 => add_8
#   convert_element_type_26 => convert_element_type_25
#   convert_element_type_28 => convert_element_type_27
#   embedding => embedding
#   mean_3 => mean_3
#   pow_4 => pow_4
# Graph fragment:
#   %embedding : [num_users=2] = call_function[target=torch.ops.aten.embedding.default](args = (%arg282_1, %arg0_1), kwargs = {})
#   %add_8 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %view_29), kwargs = {})
#   %convert_element_type_25 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_8, torch.float32), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_25, 2), kwargs = {})
#   %mean_3 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_4, [-1], True), kwargs = {})
#   %convert_element_type_27 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_30, torch.float32), kwargs = {})
triton_red_fused_add_convert_element_type_embedding_mean_pow_18 = async_compile.triton('triton_red_fused_add_convert_element_type_embedding_mean_pow_18', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp16', 'out_ptr1': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_convert_element_type_embedding_mean_pow_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 1, 'backend_hash': 'AA6F4099AE74BF369D45388847321E195BFBEE5A86FCB9B181180FAA5E25C7E8', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_red_fused_add_convert_element_type_embedding_mean_pow_18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    _tmp14 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp8 = tl.load(in_ptr2 + (r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.full([XBLOCK, R0_BLOCK], 151936, tl.int32)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp1 < 0
        tmp5 = tl.where(tmp4, tmp3, tmp1)
        tl.device_assert((0 <= tmp5) & (tmp5 < 151936), "index out of bounds: 0 <= tmp5 < 151936")
        tmp7 = tl.load(in_ptr1 + (r0_0 + 2048*tmp5), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp7 + tmp9
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tmp11 * tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(r0_mask, tmp15, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tmp17 = tl.load(in_ptr0 + (0))
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp16 = tl.load(in_ptr3 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp25 = tl.load(in_ptr2 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.full([XBLOCK, R0_BLOCK], 151936, tl.int32)
        tmp20 = tmp18 + tmp19
        tmp21 = tmp18 < 0
        tmp22 = tl.where(tmp21, tmp20, tmp18)
        tl.device_assert((0 <= tmp22) & (tmp22 < 151936), "index out of bounds: 0 <= tmp22 < 151936")
        tmp24 = tl.load(in_ptr1 + (r0_0 + 2048*tmp22), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp26 = tmp25.to(tl.float32)
        tmp27 = tmp24 + tmp26
        tmp28 = tmp27.to(tl.float32)
        tmp29 = 2048.0
        tmp30 = (tmp14 / tmp29)
        tmp31 = 1e-06
        tmp32 = tmp30 + tmp31
        tmp33 = libdevice.rsqrt(tmp32)
        tmp34 = tmp28 * tmp33
        tmp35 = tmp34.to(tl.float32)
        tmp36 = tmp16 * tmp35
        tmp37 = tmp36.to(tl.float32)
        tl.store(out_ptr1 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp37, r0_mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/j3/cj36x2d4huprntlvqk4abdcyydi7hy5ry3oqeco3owhjjgrlrxdj.py
# Topologically Sorted Source Nodes: [mul_23, sum_9], Original ATen: [aten.mul, aten.sum]
# Source node to ATen node mapping:
#   mul_23 => mul_23
#   sum_9 => sum_9
# Graph fragment:
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_30, %unsqueeze_31), kwargs = {})
#   %sum_9 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_23, [1]), kwargs = {})
triton_red_fused_mul_sum_19 = async_compile.triton('triton_red_fused_mul_sum_19', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sum_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'AA6F4099AE74BF369D45388847321E195BFBEE5A86FCB9B181180FAA5E25C7E8', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 98304, 'r0_': 50339840}}
)
@triton.jit
def triton_red_fused_mul_sum_19(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 12288
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp5 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_1 + 2048*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tmp1.to(tl.float32)
        tmp3 = tmp0 * tmp2
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(r0_mask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp5, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/tp/ctpsiysdbr32pdm2zjs45huzsx2x3rojp4kz7pwxn53v6slsrde6.py
# Topologically Sorted Source Nodes: [mul_26, sum_10], Original ATen: [aten.mul, aten.sum]
# Source node to ATen node mapping:
#   mul_26 => mul_26
#   sum_10 => sum_10
# Graph fragment:
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_32, %unsqueeze_33), kwargs = {})
#   %sum_10 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_26, [1]), kwargs = {})
triton_red_fused_mul_sum_20 = async_compile.triton('triton_red_fused_mul_sum_20', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 2048, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sum_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'AA6F4099AE74BF369D45388847321E195BFBEE5A86FCB9B181180FAA5E25C7E8', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 16384, 'r0_': 25214976}}
)
@triton.jit
def triton_red_fused_mul_sum_20(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2048
    r0_numel = 6144
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp14 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr0 + (6144 + r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.load(in_ptr1 + (r0_1 + 6144*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tmp1.to(tl.float32)
        tmp3 = tl.sigmoid(tmp2)
        tmp4 = tmp2 * tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tmp5 * tmp7
        tmp9 = tmp8.to(tl.float32)
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tmp9 * tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(r0_mask & xmask, tmp15, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/ay/cayvntsywk5ku24senibhradoms2d5v33waybh7byutt5jtr73mn.py
# Topologically Sorted Source Nodes: [embedding, add_8, add_10, convert_element_type_36, pow_5, mean_4, add_11, rsqrt_4, mul_27, convert_element_type_37, mul_28], Original ATen: [aten.embedding, aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_10 => add_10
#   add_11 => add_11
#   add_8 => add_8
#   convert_element_type_36 => convert_element_type_35
#   convert_element_type_37 => convert_element_type_36
#   embedding => embedding
#   mean_4 => mean_4
#   mul_27 => mul_27
#   mul_28 => mul_28
#   pow_5 => pow_5
#   rsqrt_4 => rsqrt_4
# Graph fragment:
#   %embedding : [num_users=2] = call_function[target=torch.ops.aten.embedding.default](args = (%arg282_1, %arg0_1), kwargs = {})
#   %add_8 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %view_29), kwargs = {})
#   %add_10 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_8, %view_33), kwargs = {})
#   %convert_element_type_35 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_10, torch.float32), kwargs = {})
#   %pow_5 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_35, 2), kwargs = {})
#   %mean_4 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_5, [-1], True), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_4, 1e-06), kwargs = {})
#   %rsqrt_4 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_11,), kwargs = {})
#   %mul_27 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_35, %rsqrt_4), kwargs = {})
#   %convert_element_type_36 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_27, torch.float16), kwargs = {})
#   %mul_28 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg296_1, %convert_element_type_36), kwargs = {})
triton_red_fused_add_convert_element_type_embedding_mean_mul_pow_rsqrt_21 = async_compile.triton('triton_red_fused_add_convert_element_type_embedding_mean_mul_pow_rsqrt_21', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp16', 'out_ptr1': '*fp16', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_convert_element_type_embedding_mean_mul_pow_rsqrt_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 1, 'backend_hash': 'AA6F4099AE74BF369D45388847321E195BFBEE5A86FCB9B181180FAA5E25C7E8', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_red_fused_add_convert_element_type_embedding_mean_mul_pow_rsqrt_21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    _tmp17 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp8 = tl.load(in_ptr2 + (r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp11 = tl.load(in_ptr3 + (r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.full([XBLOCK, R0_BLOCK], 151936, tl.int32)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp1 < 0
        tmp5 = tl.where(tmp4, tmp3, tmp1)
        tl.device_assert((0 <= tmp5) & (tmp5 < 151936), "index out of bounds: 0 <= tmp5 < 151936")
        tmp7 = tl.load(in_ptr1 + (r0_0 + 2048*tmp5), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp7 + tmp9
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp10 + tmp12
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tmp14 * tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, R0_BLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(r0_mask, tmp18, _tmp17)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tmp20 = tl.load(in_ptr0 + (0))
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, R0_BLOCK])
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp19 = tl.load(in_ptr4 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp28 = tl.load(in_ptr2 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp31 = tl.load(in_ptr3 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp22 = tl.full([XBLOCK, R0_BLOCK], 151936, tl.int32)
        tmp23 = tmp21 + tmp22
        tmp24 = tmp21 < 0
        tmp25 = tl.where(tmp24, tmp23, tmp21)
        tl.device_assert((0 <= tmp25) & (tmp25 < 151936), "index out of bounds: 0 <= tmp25 < 151936")
        tmp27 = tl.load(in_ptr1 + (r0_0 + 2048*tmp25), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp29 = tmp28.to(tl.float32)
        tmp30 = tmp27 + tmp29
        tmp32 = tmp31.to(tl.float32)
        tmp33 = tmp30 + tmp32
        tmp34 = tmp33.to(tl.float32)
        tmp35 = 2048.0
        tmp36 = (tmp17 / tmp35)
        tmp37 = 1e-06
        tmp38 = tmp36 + tmp37
        tmp39 = libdevice.rsqrt(tmp38)
        tmp40 = tmp34 * tmp39
        tmp41 = tmp40.to(tl.float32)
        tmp42 = tmp19 * tmp41
        tl.store(out_ptr1 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp42, r0_mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/ro/croiy5tvbjemk6mch4prspuurk5ddalqiob5zbuqaf6c5z375ztc.py
# Topologically Sorted Source Nodes: [mul_29, sum_11], Original ATen: [aten.mul, aten.sum]
# Source node to ATen node mapping:
#   mul_29 => mul_29
#   sum_11 => sum_11
# Graph fragment:
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_34, %unsqueeze_35), kwargs = {})
#   %sum_11 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_29, [1]), kwargs = {})
triton_red_fused_mul_sum_22 = async_compile.triton('triton_red_fused_mul_sum_22', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 2048, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sum_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'AA6F4099AE74BF369D45388847321E195BFBEE5A86FCB9B181180FAA5E25C7E8', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 16384, 'r0_': 8392704}}
)
@triton.jit
def triton_red_fused_mul_sum_22(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2048
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(r0_mask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/hx/chxvx56lseveddshszz6ql7ezcakfnrhe2vtyap7lgs2xuo5sgmg.py
# Topologically Sorted Source Nodes: [mul_32, sum_12, mul_35, sum_13, index_put_3], Original ATen: [aten.mul, aten.sum, aten.index_put]
# Source node to ATen node mapping:
#   index_put_3 => index_put_3
#   mul_32 => mul_32
#   mul_35 => mul_35
#   sum_12 => sum_12
#   sum_13 => sum_13
# Graph fragment:
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_36, %unsqueeze_37), kwargs = {})
#   %sum_12 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_32, [1]), kwargs = {})
#   %mul_35 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_38, %unsqueeze_39), kwargs = {})
#   %sum_13 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_35, [1]), kwargs = {})
#   %index_put_3 : [num_users=2] = call_function[target=torch.ops.aten.index_put_.default](args = (%arg303_1, [None, None, %arg1_1], %permute_24), kwargs = {})
triton_red_fused_index_put_mul_sum_23 = async_compile.triton('triton_red_fused_index_put_mul_sum_23', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*i64', 'out_ptr0': '*fp32', 'out_ptr2': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_index_put_mul_sum_23', 'mutated_arg_names': ['out_ptr2'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'AA6F4099AE74BF369D45388847321E195BFBEE5A86FCB9B181180FAA5E25C7E8', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_red_fused_index_put_mul_sum_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp12 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp8 = tl.load(in_ptr2 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(r0_mask & xmask, tmp7, _tmp6)
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp1 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(r0_mask & xmask, tmp13, _tmp12)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    x2 = (xindex % 128)
    x3 = xindex // 128
    tmp14 = tl.load(in_ptr3 + (0))
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, 1])
    tmp16 = tl.full([XBLOCK, 1], 960, tl.int32)
    tmp17 = tmp15 + tmp16
    tmp18 = tmp15 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp15)
    tl.device_assert((0 <= tmp19) & (tmp19 < 960), "index out of bounds: 0 <= tmp19 < 960")
    tmp21 = tmp12.to(tl.float32)
    tl.store(out_ptr2 + (x2 + 128*tmp19 + 122880*x3), tmp21, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/3o/c3ozoitxebdcetf3znznxxjdek6545sutxiavhgpwcjkehfr3mus.py
# Topologically Sorted Source Nodes: [embedding, add_8, add_10, mul_44, sum_17, add_17], Original ATen: [aten.embedding, aten.add, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   add_10 => add_10
#   add_17 => add_17
#   add_8 => add_8
#   embedding => embedding
#   mul_44 => mul_44
#   sum_17 => sum_17
# Graph fragment:
#   %embedding : [num_users=2] = call_function[target=torch.ops.aten.embedding.default](args = (%arg282_1, %arg0_1), kwargs = {})
#   %add_8 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %view_29), kwargs = {})
#   %add_10 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_8, %view_33), kwargs = {})
#   %mul_44 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_48, %unsqueeze_49), kwargs = {})
#   %sum_17 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_44, [1]), kwargs = {})
#   %add_17 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_10, %view_53), kwargs = {})
triton_red_fused_add_embedding_mul_sum_24 = async_compile.triton('triton_red_fused_add_embedding_mul_sum_24', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 2048, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp16', 'in_ptr2': '*i64', 'in_ptr3': '*fp16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr1': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_embedding_mul_sum_24', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 1, 'backend_hash': 'AA6F4099AE74BF369D45388847321E195BFBEE5A86FCB9B181180FAA5E25C7E8', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_red_fused_add_embedding_mul_sum_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2048
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp7 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tmp1.to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tmp2 * tmp4
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(r0_mask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tmp9 = tl.load(in_ptr2 + (0))
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, 1])
    tmp17 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.full([XBLOCK, 1], 151936, tl.int32)
    tmp12 = tmp10 + tmp11
    tmp13 = tmp10 < 0
    tmp14 = tl.where(tmp13, tmp12, tmp10)
    tl.device_assert((0 <= tmp14) & (tmp14 < 151936), "index out of bounds: 0 <= tmp14 < 151936")
    tmp16 = tl.load(in_ptr3 + (x0 + 2048*tmp14), xmask).to(tl.float32)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 + tmp18
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp19 + tmp21
    tmp23 = tmp7.to(tl.float32)
    tmp24 = tmp22 + tmp23
    tl.store(out_ptr1 + (x0), tmp24, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/wt/cwtqdvohz3hd6rooalecqwsofasluocr6wiwzsya3jrbo2uskna3.py
# Topologically Sorted Source Nodes: [convert_element_type_59, pow_8, mean_7], Original ATen: [prims.convert_element_type, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   convert_element_type_59 => convert_element_type_57
#   mean_7 => mean_7
#   pow_8 => pow_8
# Graph fragment:
#   %convert_element_type_57 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_17, torch.float32), kwargs = {})
#   %pow_8 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_57, 2), kwargs = {})
#   %mean_7 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_8, [-1], True), kwargs = {})
triton_red_fused_convert_element_type_mean_pow_25 = async_compile.triton('triton_red_fused_convert_element_type_mean_pow_25', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convert_element_type_mean_pow_25', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'AA6F4099AE74BF369D45388847321E195BFBEE5A86FCB9B181180FAA5E25C7E8', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'r0_': 4096}}
)
@triton.jit
def triton_red_fused_convert_element_type_mean_pow_25(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tmp1 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(r0_mask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/xj/cxjr4ta6x3teuzpqzwewxd36cxil6dm43ar2tefiamaotg57tt46.py
# Topologically Sorted Source Nodes: [mul_47, sum_18], Original ATen: [aten.mul, aten.sum]
# Source node to ATen node mapping:
#   mul_47 => mul_47
#   sum_18 => sum_18
# Graph fragment:
#   %mul_47 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_50, %unsqueeze_51), kwargs = {})
#   %sum_18 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_47, [1]), kwargs = {})
triton_red_fused_mul_sum_26 = async_compile.triton('triton_red_fused_mul_sum_26', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sum_26', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': 'AA6F4099AE74BF369D45388847321E195BFBEE5A86FCB9B181180FAA5E25C7E8', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 98304, 'r0_': 50339840}}
)
@triton.jit
def triton_red_fused_mul_sum_26(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 12288
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    tmp3 = tl.load(in_ptr2 + (0))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    x0 = xindex
    _tmp18 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp14 = tl.load(in_ptr3 + (r0_1 + 2048*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tmp1.to(tl.float32)
        tmp5 = 2048.0
        tmp6 = (tmp4 / tmp5)
        tmp7 = 1e-06
        tmp8 = tmp6 + tmp7
        tmp9 = libdevice.rsqrt(tmp8)
        tmp10 = tmp2 * tmp9
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tmp0 * tmp11
        tmp13 = tmp12.to(tl.float32)
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tmp13 * tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(r0_mask, tmp19, _tmp18)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp18, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/3x/c3xwtmotmloyads5cfddflmc7iki3c4sx2evrp3vzifz5peaac6t.py
# Topologically Sorted Source Nodes: [add_19, convert_element_type_69, pow_9, mean_8], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   add_19 => add_19
#   convert_element_type_69 => convert_element_type_67
#   mean_8 => mean_8
#   pow_9 => pow_9
# Graph fragment:
#   %add_19 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_17, %view_57), kwargs = {})
#   %convert_element_type_67 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_19, torch.float32), kwargs = {})
#   %pow_9 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_67, 2), kwargs = {})
#   %mean_8 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_9, [-1], True), kwargs = {})
triton_red_fused_add_convert_element_type_mean_pow_27 = async_compile.triton('triton_red_fused_add_convert_element_type_mean_pow_27', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_convert_element_type_mean_pow_27', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'AA6F4099AE74BF369D45388847321E195BFBEE5A86FCB9B181180FAA5E25C7E8', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'r0_': 12288}}
)
@triton.jit
def triton_red_fused_add_convert_element_type_mean_pow_27(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp7 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp1.to(tl.float32)
        tmp3 = tmp0 + tmp2
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tmp4 * tmp4
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(r0_mask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp7, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/ox/coxm7gnbtjf4wulz4vbarhwr3n4kfej6zfqgeqoybffictahyhwd.py
# Topologically Sorted Source Nodes: [mul_53, sum_20], Original ATen: [aten.mul, aten.sum]
# Source node to ATen node mapping:
#   mul_53 => mul_53
#   sum_20 => sum_20
# Graph fragment:
#   %mul_53 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_54, %unsqueeze_55), kwargs = {})
#   %sum_20 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_53, [1]), kwargs = {})
triton_red_fused_mul_sum_28 = async_compile.triton('triton_red_fused_mul_sum_28', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 2048, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sum_28', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 1, 'backend_hash': 'AA6F4099AE74BF369D45388847321E195BFBEE5A86FCB9B181180FAA5E25C7E8', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 16384, 'r0_': 8404992}}
)
@triton.jit
def triton_red_fused_mul_sum_28(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2048
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    tmp6 = tl.load(in_ptr3 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    x0 = xindex
    _tmp21 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.load(in_ptr4 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 + tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp8 = 2048.0
        tmp9 = (tmp7 / tmp8)
        tmp10 = 1e-06
        tmp11 = tmp9 + tmp10
        tmp12 = libdevice.rsqrt(tmp11)
        tmp13 = tmp5 * tmp12
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tmp0 * tmp14
        tmp16 = tmp15.to(tl.float32)
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tmp16 * tmp18
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(r0_mask & xmask, tmp22, _tmp21)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp21, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/ke/ckeab6zygjjnjaqtiqc5cjwoywiszhssqr2maytgbigaz75lgv24.py
# Topologically Sorted Source Nodes: [mul_56, sum_21, mul_59, sum_22, index_put_5], Original ATen: [aten.mul, aten.sum, aten.index_put]
# Source node to ATen node mapping:
#   index_put_5 => index_put_5
#   mul_56 => mul_56
#   mul_59 => mul_59
#   sum_21 => sum_21
#   sum_22 => sum_22
# Graph fragment:
#   %mul_56 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_56, %unsqueeze_57), kwargs = {})
#   %sum_21 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_56, [1]), kwargs = {})
#   %mul_59 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_58, %unsqueeze_59), kwargs = {})
#   %sum_22 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_59, [1]), kwargs = {})
#   %index_put_5 : [num_users=2] = call_function[target=torch.ops.aten.index_put_.default](args = (%arg315_1, [None, None, %arg1_1], %permute_35), kwargs = {})
triton_red_fused_index_put_mul_sum_29 = async_compile.triton('triton_red_fused_index_put_mul_sum_29', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp16', 'in_ptr5': '*fp16', 'in_ptr6': '*i64', 'out_ptr0': '*fp32', 'out_ptr2': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_index_put_mul_sum_29', 'mutated_arg_names': ['out_ptr2'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 2, 'backend_hash': 'AA6F4099AE74BF369D45388847321E195BFBEE5A86FCB9B181180FAA5E25C7E8', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_red_fused_index_put_mul_sum_29(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    tmp6 = tl.load(in_ptr3 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    x0 = xindex
    _tmp21 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp27 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.load(in_ptr4 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp23 = tl.load(in_ptr5 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 + tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp8 = 2048.0
        tmp9 = (tmp7 / tmp8)
        tmp10 = 1e-06
        tmp11 = tmp9 + tmp10
        tmp12 = libdevice.rsqrt(tmp11)
        tmp13 = tmp5 * tmp12
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tmp0 * tmp14
        tmp16 = tmp15.to(tl.float32)
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tmp16 * tmp18
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(r0_mask & xmask, tmp22, _tmp21)
        tmp24 = tmp23.to(tl.float32)
        tmp25 = tmp16 * tmp24
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, R0_BLOCK])
        tmp28 = _tmp27 + tmp26
        _tmp27 = tl.where(r0_mask & xmask, tmp28, _tmp27)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tmp27 = tl.sum(_tmp27, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp21, xmask)
    x2 = (xindex % 128)
    x3 = xindex // 128
    tmp29 = tl.load(in_ptr6 + (0))
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK, 1])
    tmp31 = tl.full([XBLOCK, 1], 960, tl.int32)
    tmp32 = tmp30 + tmp31
    tmp33 = tmp30 < 0
    tmp34 = tl.where(tmp33, tmp32, tmp30)
    tl.device_assert((0 <= tmp34) & (tmp34 < 960), "index out of bounds: 0 <= tmp34 < 960")
    tmp36 = tmp27.to(tl.float32)
    tl.store(out_ptr2 + (x2 + 128*tmp34 + 122880*x3), tmp36, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/p6/cp6lheyym6myiuyqakw24yo6ikm6guyvsvinayyffavejts3a3ka.py
# Topologically Sorted Source Nodes: [add_19, add_26, convert_element_type_92, pow_12, mean_11, convert_element_type_94], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   add_19 => add_19
#   add_26 => add_26
#   convert_element_type_92 => convert_element_type_89
#   convert_element_type_94 => convert_element_type_91
#   mean_11 => mean_11
#   pow_12 => pow_12
# Graph fragment:
#   %add_19 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_17, %view_57), kwargs = {})
#   %add_26 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_19, %view_77), kwargs = {})
#   %convert_element_type_89 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_26, torch.float32), kwargs = {})
#   %pow_12 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_89, 2), kwargs = {})
#   %mean_11 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_12, [-1], True), kwargs = {})
#   %convert_element_type_91 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_78, torch.float32), kwargs = {})
triton_red_fused_add_convert_element_type_mean_pow_30 = async_compile.triton('triton_red_fused_add_convert_element_type_mean_pow_30', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp16', 'out_ptr1': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_convert_element_type_mean_pow_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 1, 'backend_hash': 'AA6F4099AE74BF369D45388847321E195BFBEE5A86FCB9B181180FAA5E25C7E8', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'r0_': 40960}}
)
@triton.jit
def triton_red_fused_add_convert_element_type_mean_pow_30(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp1.to(tl.float32)
        tmp3 = tmp0 + tmp2
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp3 + tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tmp7 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(r0_mask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp12 = tl.load(in_ptr3 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp13 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp14 = tl.load(in_ptr1 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp17 = tl.load(in_ptr2 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tmp13 + tmp15
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tmp16 + tmp18
        tmp20 = tmp19.to(tl.float32)
        tmp21 = 2048.0
        tmp22 = (tmp10 / tmp21)
        tmp23 = 1e-06
        tmp24 = tmp22 + tmp23
        tmp25 = libdevice.rsqrt(tmp24)
        tmp26 = tmp20 * tmp25
        tmp27 = tmp26.to(tl.float32)
        tmp28 = tmp12 * tmp27
        tmp29 = tmp28.to(tl.float32)
        tl.store(out_ptr1 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp29, r0_mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/w6/cw6vt24oyizshz4bfr6veufff64stds5pgniulggifxa6nsyrt5g.py
# Topologically Sorted Source Nodes: [add_19, add_26, add_28, convert_element_type_102, pow_13, mean_12, add_29, rsqrt_12, mul_75, convert_element_type_103, mul_76], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_19 => add_19
#   add_26 => add_26
#   add_28 => add_28
#   add_29 => add_29
#   convert_element_type_102 => convert_element_type_99
#   convert_element_type_103 => convert_element_type_100
#   mean_12 => mean_12
#   mul_75 => mul_75
#   mul_76 => mul_76
#   pow_13 => pow_13
#   rsqrt_12 => rsqrt_12
# Graph fragment:
#   %add_19 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_17, %view_57), kwargs = {})
#   %add_26 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_19, %view_77), kwargs = {})
#   %add_28 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_26, %view_81), kwargs = {})
#   %convert_element_type_99 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_28, torch.float32), kwargs = {})
#   %pow_13 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_99, 2), kwargs = {})
#   %mean_12 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_13, [-1], True), kwargs = {})
#   %add_29 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_12, 1e-06), kwargs = {})
#   %rsqrt_12 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_29,), kwargs = {})
#   %mul_75 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_99, %rsqrt_12), kwargs = {})
#   %convert_element_type_100 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_75, torch.float16), kwargs = {})
#   %mul_76 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg320_1, %convert_element_type_100), kwargs = {})
triton_red_fused_add_convert_element_type_mean_mul_pow_rsqrt_31 = async_compile.triton('triton_red_fused_add_convert_element_type_mean_mul_pow_rsqrt_31', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp16', 'out_ptr1': '*fp16', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_convert_element_type_mean_mul_pow_rsqrt_31', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 1, 'backend_hash': 'AA6F4099AE74BF369D45388847321E195BFBEE5A86FCB9B181180FAA5E25C7E8', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'r0_': 40960}}
)
@triton.jit
def triton_red_fused_add_convert_element_type_mean_mul_pow_rsqrt_31(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp13 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr3 + (r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp1.to(tl.float32)
        tmp3 = tmp0 + tmp2
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp3 + tmp5
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp6 + tmp8
        tmp10 = tmp9.to(tl.float32)
        tmp11 = tmp10 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(r0_mask, tmp14, _tmp13)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp15 = tl.load(in_ptr4 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp16 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp17 = tl.load(in_ptr1 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp20 = tl.load(in_ptr2 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp23 = tl.load(in_ptr3 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tmp16 + tmp18
        tmp21 = tmp20.to(tl.float32)
        tmp22 = tmp19 + tmp21
        tmp24 = tmp23.to(tl.float32)
        tmp25 = tmp22 + tmp24
        tmp26 = tmp25.to(tl.float32)
        tmp27 = 2048.0
        tmp28 = (tmp13 / tmp27)
        tmp29 = 1e-06
        tmp30 = tmp28 + tmp29
        tmp31 = libdevice.rsqrt(tmp30)
        tmp32 = tmp26 * tmp31
        tmp33 = tmp32.to(tl.float32)
        tmp34 = tmp15 * tmp33
        tl.store(out_ptr1 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp34, r0_mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/r3/cr32hfvgvhl2fablj33jdpohkyik63zwuir4suwgh3wcdqqrknqv.py
# Topologically Sorted Source Nodes: [add_19, add_26, add_28, mul_92, sum_35, add_35], Original ATen: [aten.add, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   add_19 => add_19
#   add_26 => add_26
#   add_28 => add_28
#   add_35 => add_35
#   mul_92 => mul_92
#   sum_35 => sum_35
# Graph fragment:
#   %add_19 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_17, %view_57), kwargs = {})
#   %add_26 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_19, %view_77), kwargs = {})
#   %add_28 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_26, %view_81), kwargs = {})
#   %mul_92 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_88, %unsqueeze_89), kwargs = {})
#   %sum_35 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_92, [1]), kwargs = {})
#   %add_35 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_28, %view_101), kwargs = {})
triton_red_fused_add_mul_sum_32 = async_compile.triton('triton_red_fused_add_mul_sum_32', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 2048, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_ptr0': '*fp32', 'in_ptr1': '*fp16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_sum_32', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 1, 'backend_hash': 'AA6F4099AE74BF369D45388847321E195BFBEE5A86FCB9B181180FAA5E25C7E8', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 36864, 'r0_': 8396800}}
)
@triton.jit
def triton_red_fused_add_mul_sum_32(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2048
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp7 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tmp1.to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tmp2 * tmp4
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(r0_mask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tmp9 = tl.load(in_out_ptr0 + (x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp10 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 + tmp11
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 + tmp14
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 + tmp17
    tmp19 = tmp7.to(tl.float32)
    tmp20 = tmp18 + tmp19
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp20, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/eh/ceharl4fukk4xv4iv4zz5nrzjlb5wbr7srkzwt7tgdjsfaiuylqr.py
# Topologically Sorted Source Nodes: [logits_0], Original ATen: [aten.mm]
# Source node to ATen node mapping:
#   logits_0 => mul_677, sum_254
# Graph fragment:
#   %mul_677 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_574, %unsqueeze_575), kwargs = {})
#   %sum_254 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_677, [1]), kwargs = {})
triton_red_fused_mm_33 = async_compile.triton('triton_red_fused_mm_33', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 262144, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mm_33', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 1, 'backend_hash': 'AA6F4099AE74BF369D45388847321E195BFBEE5A86FCB9B181180FAA5E25C7E8', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 1215488, 'r0_': 622346240}}
)
@triton.jit
def triton_red_fused_mm_33(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 151936
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    tmp6 = tl.load(in_ptr3 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    x0 = xindex
    _tmp21 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.load(in_ptr4 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 + tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp8 = 2048.0
        tmp9 = (tmp7 / tmp8)
        tmp10 = 1e-06
        tmp11 = tmp9 + tmp10
        tmp12 = libdevice.rsqrt(tmp11)
        tmp13 = tmp5 * tmp12
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tmp0 * tmp14
        tmp16 = tmp15.to(tl.float32)
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tmp16 * tmp18
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(r0_mask & xmask, tmp22, _tmp21)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp21, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/3c/c3cp633lnuekdxcci3o5typ3asluzrbvrmkdxrpjihnnkrljxxrf.py
# Topologically Sorted Source Nodes: [logits_0, argmax, cat_224], Original ATen: [aten.mm, aten.argmax, aten.cat]
# Source node to ATen node mapping:
#   argmax => argmax
#   cat_224 => cat_224
#   logits_0 => convert_element_type_903
# Graph fragment:
#   %convert_element_type_903 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_254, torch.float16), kwargs = {})
#   %argmax : [num_users=2] = call_function[target=torch.ops.aten.argmax.default](args = (%convert_element_type_903, -1), kwargs = {})
#   %cat_224 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_2731, %view_2732, %view_2733, %view_2734], 1), kwargs = {})
triton_red_fused_argmax_cat_mm_34 = async_compile.triton('triton_red_fused_argmax_cat_mm_34', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r0_': 262144},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*i64', 'out_ptr1': '*i64', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_argmax_cat_mm_34', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'AA6F4099AE74BF369D45388847321E195BFBEE5A86FCB9B181180FAA5E25C7E8', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'r0_': 607744}}
)
@triton.jit
def triton_red_fused_argmax_cat_mm_34(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 151936
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp3 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    _tmp3_index = tl.full([XBLOCK, R0_BLOCK], 2147483647, tl.int32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        _tmp3_next, _tmp3_index_next = triton_helpers.maximum_with_index(
            _tmp3, _tmp3_index, tmp2, rindex
        )
        _tmp3 = tl.where(r0_mask, _tmp3_next, _tmp3)
        _tmp3_index = tl.where(r0_mask, _tmp3_index_next, _tmp3_index)
    tmp3_val, tmp3_idx = triton_helpers.max_with_index(_tmp3, _tmp3_index, 1)
    tmp3 = tmp3_idx[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp3, None)
    tl.store(out_ptr1 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp3, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/uo/cuoo777ezhhg5plstncax763agrnrfjqqj7cte3p23ln6e6uq7yg.py
# Topologically Sorted Source Nodes: [mul_685, sum_256, mul_688, sum_257, index_put_57], Original ATen: [aten.mul, aten.sum, aten.index_put]
# Source node to ATen node mapping:
#   index_put_57 => index_put_57
#   mul_685 => mul_686
#   mul_688 => mul_689
#   sum_256 => sum_257
#   sum_257 => sum_258
# Graph fragment:
#   %mul_686 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_592, %unsqueeze_593), kwargs = {})
#   %sum_257 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_686, [1]), kwargs = {})
#   %mul_689 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_594, %unsqueeze_595), kwargs = {})
#   %sum_258 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_689, [1]), kwargs = {})
#   %index_put_57 : [num_users=2] = call_function[target=torch.ops.aten.index_put_.default](args = (%index_put_1, [None, None, %add_255], %permute_330), kwargs = {})
triton_red_fused_index_put_mul_sum_35 = async_compile.triton('triton_red_fused_index_put_mul_sum_35', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*i64', 'in_ptr2': '*fp16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp16', 'in_ptr5': '*fp16', 'in_ptr6': '*i64', 'out_ptr0': '*fp32', 'out_ptr2': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_index_put_mul_sum_35', 'mutated_arg_names': ['out_ptr2'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 2, 'backend_hash': 'AA6F4099AE74BF369D45388847321E195BFBEE5A86FCB9B181180FAA5E25C7E8', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_red_fused_index_put_mul_sum_35(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp10 = tl.load(in_ptr3 + (0))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    x0 = xindex
    _tmp25 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp31 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp21 = tl.load(in_ptr4 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp27 = tl.load(in_ptr5 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.full([XBLOCK, R0_BLOCK], 151936, tl.int32)
        tmp4 = tmp2 + tmp3
        tmp5 = tmp2 < 0
        tmp6 = tl.where(tmp5, tmp4, tmp2)
        tl.device_assert((0 <= tmp6) & (tmp6 < 151936), "index out of bounds: 0 <= tmp6 < 151936")
        tmp8 = tl.load(in_ptr2 + (r0_1 + 2048*tmp6), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp9 = tmp8.to(tl.float32)
        tmp12 = 2048.0
        tmp13 = (tmp11 / tmp12)
        tmp14 = 1e-06
        tmp15 = tmp13 + tmp14
        tmp16 = libdevice.rsqrt(tmp15)
        tmp17 = tmp9 * tmp16
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tmp0 * tmp18
        tmp20 = tmp19.to(tl.float32)
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp20 * tmp22
        tmp24 = tl.broadcast_to(tmp23, [XBLOCK, R0_BLOCK])
        tmp26 = _tmp25 + tmp24
        _tmp25 = tl.where(r0_mask & xmask, tmp26, _tmp25)
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tmp20 * tmp28
        tmp30 = tl.broadcast_to(tmp29, [XBLOCK, R0_BLOCK])
        tmp32 = _tmp31 + tmp30
        _tmp31 = tl.where(r0_mask & xmask, tmp32, _tmp31)
    tmp25 = tl.sum(_tmp25, 1)[:, None]
    tmp31 = tl.sum(_tmp31, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp25, xmask)
    x2 = (xindex % 128)
    x3 = xindex // 128
    tmp33 = tl.load(in_ptr6 + (0))
    tmp34 = tl.broadcast_to(tmp33, [XBLOCK, 1])
    tmp35 = tl.full([1, 1], 1, tl.int64)
    tmp36 = tmp34 + tmp35
    tmp37 = tl.full([XBLOCK, 1], 960, tl.int32)
    tmp38 = tmp36 + tmp37
    tmp39 = tmp36 < 0
    tmp40 = tl.where(tmp39, tmp38, tmp36)
    tl.device_assert((0 <= tmp40) & (tmp40 < 960), "index out of bounds: 0 <= tmp40 < 960")
    tmp42 = tmp31.to(tl.float32)
    tl.store(out_ptr2 + (x2 + 128*tmp40 + 122880*x3), tmp42, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/p6/cp65z6a2fuxvvf7enn5yptx53yalplcge7ybqiu3cyud3ydtq46x.py
# Topologically Sorted Source Nodes: [convert_element_type_417, pow_116, mean_115, mul_691, cat_57, mul_692, add_262, index_put_56], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
# Source node to ATen node mapping:
#   add_262 => add_262
#   cat_57 => cat_57
#   convert_element_type_417 => convert_element_type_917
#   index_put_56 => index_put_56
#   mean_115 => mean_115
#   mul_691 => mul_692
#   mul_692 => mul_693
#   pow_116 => pow_116
# Graph fragment:
#   %convert_element_type_917 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_698, torch.float32), kwargs = {})
#   %pow_116 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_917, 2), kwargs = {})
#   %mean_115 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_116, [-1], True), kwargs = {})
#   %mul_692 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_328, %unsqueeze_596), kwargs = {})
#   %cat_57 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%neg_57, %slice_179], -1), kwargs = {})
#   %mul_693 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat_57, %unsqueeze_597), kwargs = {})
#   %add_262 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_692, %mul_693), kwargs = {})
#   %index_put_56 : [num_users=2] = call_function[target=torch.ops.aten.index_put_.default](args = (%index_put, [None, None, %add_255], %add_262), kwargs = {})
triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_36 = async_compile.triton('triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_36', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp16', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'out_ptr2': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_36', 'mutated_arg_names': ['out_ptr2'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 1, 'backend_hash': 'AA6F4099AE74BF369D45388847321E195BFBEE5A86FCB9B181180FAA5E25C7E8', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_36(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 8
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 128*x0), xmask, other=0.0)
    tmp46 = tl.load(in_ptr2 + (0))
    tmp47 = tl.broadcast_to(tmp46, [XBLOCK, R0_BLOCK])
    tmp55 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last').to(tl.float32)
    tmp64 = tl.load(in_ptr3 + ((r0_1 % 64)), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp2 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp6 = tl.where(xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = r0_1
    tmp9 = tl.full([1, 1], 0, tl.int64)
    tmp10 = tmp8 >= tmp9
    tmp11 = tl.full([1, 1], 64, tl.int64)
    tmp12 = tmp8 < tmp11
    tmp13 = tl.load(in_ptr1 + (tl.broadcast_to(64 + (r0_1), [XBLOCK, R0_BLOCK])), tmp12 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp14 = tl.load(in_ptr0 + (64 + 128*x0 + (r0_1)), tmp12 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = 128.0
    tmp18 = (tmp7 / tmp17)
    tmp19 = 1e-06
    tmp20 = tmp18 + tmp19
    tmp21 = libdevice.rsqrt(tmp20)
    tmp22 = tmp16 * tmp21
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp13 * tmp23
    tmp25 = -tmp24
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp12, tmp25, tmp26)
    tmp28 = tmp8 >= tmp11
    tmp29 = tl.full([1, 1], 128, tl.int64)
    tmp30 = tmp8 < tmp29
    tmp31 = tl.load(in_ptr1 + (tl.broadcast_to((-64) + r0_1, [XBLOCK, R0_BLOCK])), tmp28 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp32 = tl.load(in_ptr0 + (128*x0 + ((-64) + r0_1)), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tmp33.to(tl.float32)
    tmp35 = 128.0
    tmp36 = (tmp7 / tmp35)
    tmp37 = 1e-06
    tmp38 = tmp36 + tmp37
    tmp39 = libdevice.rsqrt(tmp38)
    tmp40 = tmp34 * tmp39
    tmp41 = tmp40.to(tl.float32)
    tmp42 = tmp31 * tmp41
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp28, tmp42, tmp43)
    tmp45 = tl.where(tmp12, tmp27, tmp44)
    tmp48 = tl.full([1, 1], 1, tl.int64)
    tmp49 = tmp47 + tmp48
    tmp50 = tl.full([XBLOCK, R0_BLOCK], 960, tl.int32)
    tmp51 = tmp49 + tmp50
    tmp52 = tmp49 < 0
    tmp53 = tl.where(tmp52, tmp51, tmp49)
    tl.device_assert((0 <= tmp53) & (tmp53 < 960), "index out of bounds: 0 <= tmp53 < 960")
    tmp56 = 128.0
    tmp57 = (tmp7 / tmp56)
    tmp58 = 1e-06
    tmp59 = tmp57 + tmp58
    tmp60 = libdevice.rsqrt(tmp59)
    tmp61 = tmp2 * tmp60
    tmp62 = tmp61.to(tl.float32)
    tmp63 = tmp55 * tmp62
    tmp65 = tl_math.cos(tmp64)
    tmp66 = 1.0
    tmp67 = tmp65 * tmp66
    tmp68 = tmp67.to(tl.float32)
    tmp69 = tmp63 * tmp68
    tmp70 = tl_math.sin(tmp64)
    tmp71 = tmp70 * tmp66
    tmp72 = tmp71.to(tl.float32)
    tmp73 = tmp45 * tmp72
    tmp74 = tmp69 + tmp73
    tl.store(out_ptr2 + (r0_1 + 128*tmp53 + 122880*x0), tmp74, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/lh/clhw6naeo4rc7dpgqisczicqwk2ompqrq5gvo73jtrisum5ejkni.py
# Topologically Sorted Source Nodes: [full_default_87, full_default_85, where_56, add_263, eq_28, logical_not_56, any_29, logical_not_57, full_default_89, , sub, exp, div_28, where_57], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
# Source node to ATen node mapping:
#    => prepare_softmax_online_default_83
#   add_263 => add_263
#   any_29 => any_29
#   div_28 => div_28
#   eq_28 => eq_28
#   exp => exp_default_83
#   full_default_85 => full_default_84
#   full_default_87 => full_default_85
#   full_default_89 => full_default_86
#   logical_not_56 => logical_not_56
#   logical_not_57 => logical_not_57
#   sub => sub_tensor_83
#   where_56 => where_56
#   where_57 => where_57
# Graph fragment:
#   %full_default_85 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_84 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_56 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%expand_175, %full_default_85, %full_default_84), kwargs = {})
#   %add_263 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_706, %where_56), kwargs = {})
#   %eq_28 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%add_263, -inf), kwargs = {})
#   %logical_not_56 : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%eq_28,), kwargs = {})
#   %any_29 : [num_users=1] = call_function[target=torch.ops.aten.any.dim](args = (%logical_not_56, -1, True), kwargs = {})
#   %logical_not_57 : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%any_29,), kwargs = {})
#   %full_default_86 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 16, 1, 960], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %prepare_softmax_online_default_83 : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%add_263, -1), kwargs = {})
#   %sub_tensor_83 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_263, %getitem_166), kwargs = {})
#   %exp_default_83 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_83,), kwargs = {})
#   %div_28 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_83, %getitem_167), kwargs = {})
#   %where_57 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_not_57, %full_default_86, %div_28), kwargs = {})
triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_37 = async_compile.triton('triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_37', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_37', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 5, 'backend_hash': 'AA6F4099AE74BF369D45388847321E195BFBEE5A86FCB9B181180FAA5E25C7E8', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 0, 'r0_': 184320}}
)
@triton.jit
def triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_37(in_ptr0, in_ptr1, out_ptr3, xnumel, r0_numel):
    xnumel = 16
    XBLOCK: tl.constexpr = 1
    r0_numel = 960
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 960*x0), r0_mask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [R0_BLOCK])
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp2 + tmp3
    tmp5 = r0_1
    tmp6 = tmp5 <= tmp4
    tmp7 = 0.0
    tmp8 = float("-inf")
    tmp9 = tl.where(tmp6, tmp7, tmp8)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp0 + tmp10
    tmp12 = tmp11 == tmp8
    tmp13 = tmp12 == 0
    tmp14 = tmp13.to(tl.int64)
    tmp15 = (tmp14 != 0)
    tmp16 = tl.broadcast_to(tmp15, [R0_BLOCK])
    tmp18 = tl.where(r0_mask, tmp16, False)
    tmp19 = triton_helpers.promote_to_tensor(triton_helpers.any(tmp18, 0))
    tmp20 = tl.broadcast_to(tmp11, [R0_BLOCK])
    tmp22 = tl.broadcast_to(tmp20, [R0_BLOCK])
    tmp24 = tl.where(r0_mask, tmp22, float("-inf"))
    tmp25 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp24, 0))
    tmp26 = tmp20 - tmp25
    tmp27 = tl_math.exp(tmp26)
    tmp28 = tl.broadcast_to(tmp27, [R0_BLOCK])
    tmp30 = tl.where(r0_mask, tmp28, 0)
    tmp31 = triton_helpers.promote_to_tensor(tl.sum(tmp30, 0))
    tmp32 = tmp19 == 0
    tmp33 = tmp11 - tmp25
    tmp34 = tl_math.exp(tmp33)
    tmp35 = (tmp34 / tmp31)
    tmp36 = tl.where(tmp32, tmp7, tmp35)
    tl.store(out_ptr3 + (r0_1 + 960*x0), tmp36, r0_mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/lu/clusgmgzwfvce2uefjbucfarzbmafsj75xfud2vewb7bl7sw6hsb.py
# Topologically Sorted Source Nodes: [mul_709, sum_265, mul_712, sum_266, index_put_59], Original ATen: [aten.mul, aten.sum, aten.index_put]
# Source node to ATen node mapping:
#   index_put_59 => index_put_59
#   mul_709 => mul_710
#   mul_712 => mul_713
#   sum_265 => sum_266
#   sum_266 => sum_267
# Graph fragment:
#   %mul_710 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_612, %unsqueeze_613), kwargs = {})
#   %sum_266 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_710, [1]), kwargs = {})
#   %mul_713 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_614, %unsqueeze_615), kwargs = {})
#   %sum_267 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_713, [1]), kwargs = {})
#   %index_put_59 : [num_users=2] = call_function[target=torch.ops.aten.index_put_.default](args = (%index_put_3, [None, None, %add_255], %permute_341), kwargs = {})
triton_red_fused_index_put_mul_sum_38 = async_compile.triton('triton_red_fused_index_put_mul_sum_38', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*i64', 'out_ptr0': '*fp32', 'out_ptr2': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_index_put_mul_sum_38', 'mutated_arg_names': ['out_ptr2'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'AA6F4099AE74BF369D45388847321E195BFBEE5A86FCB9B181180FAA5E25C7E8', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_red_fused_index_put_mul_sum_38(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp12 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp8 = tl.load(in_ptr2 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(r0_mask & xmask, tmp7, _tmp6)
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp1 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(r0_mask & xmask, tmp13, _tmp12)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    x2 = (xindex % 128)
    x3 = xindex // 128
    tmp14 = tl.load(in_ptr3 + (0))
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, 1])
    tmp16 = tl.full([1, 1], 1, tl.int64)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([XBLOCK, 1], 960, tl.int32)
    tmp19 = tmp17 + tmp18
    tmp20 = tmp17 < 0
    tmp21 = tl.where(tmp20, tmp19, tmp17)
    tl.device_assert((0 <= tmp21) & (tmp21 < 960), "index out of bounds: 0 <= tmp21 < 960")
    tmp23 = tmp12.to(tl.float32)
    tl.store(out_ptr2 + (x2 + 128*tmp21 + 122880*x3), tmp23, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/tf/ctfwbdp6kwswqlxpr77srvvglbuyge6k3tojn3uadagrp4j4shju.py
# Topologically Sorted Source Nodes: [mul_733, sum_274, mul_736, sum_275, index_put_61], Original ATen: [aten.mul, aten.sum, aten.index_put]
# Source node to ATen node mapping:
#   index_put_61 => index_put_61
#   mul_733 => mul_734
#   mul_736 => mul_737
#   sum_274 => sum_275
#   sum_275 => sum_276
# Graph fragment:
#   %mul_734 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_632, %unsqueeze_633), kwargs = {})
#   %sum_275 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_734, [1]), kwargs = {})
#   %mul_737 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_634, %unsqueeze_635), kwargs = {})
#   %sum_276 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_737, [1]), kwargs = {})
#   %index_put_61 : [num_users=2] = call_function[target=torch.ops.aten.index_put_.default](args = (%index_put_5, [None, None, %add_255], %permute_352), kwargs = {})
triton_red_fused_index_put_mul_sum_39 = async_compile.triton('triton_red_fused_index_put_mul_sum_39', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp16', 'in_ptr5': '*fp16', 'in_ptr6': '*i64', 'out_ptr0': '*fp32', 'out_ptr2': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_index_put_mul_sum_39', 'mutated_arg_names': ['out_ptr2'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 2, 'backend_hash': 'AA6F4099AE74BF369D45388847321E195BFBEE5A86FCB9B181180FAA5E25C7E8', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_red_fused_index_put_mul_sum_39(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    tmp6 = tl.load(in_ptr3 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    x0 = xindex
    _tmp21 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp27 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.load(in_ptr4 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp23 = tl.load(in_ptr5 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 + tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp8 = 2048.0
        tmp9 = (tmp7 / tmp8)
        tmp10 = 1e-06
        tmp11 = tmp9 + tmp10
        tmp12 = libdevice.rsqrt(tmp11)
        tmp13 = tmp5 * tmp12
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tmp0 * tmp14
        tmp16 = tmp15.to(tl.float32)
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tmp16 * tmp18
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(r0_mask & xmask, tmp22, _tmp21)
        tmp24 = tmp23.to(tl.float32)
        tmp25 = tmp16 * tmp24
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, R0_BLOCK])
        tmp28 = _tmp27 + tmp26
        _tmp27 = tl.where(r0_mask & xmask, tmp28, _tmp27)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tmp27 = tl.sum(_tmp27, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp21, xmask)
    x2 = (xindex % 128)
    x3 = xindex // 128
    tmp29 = tl.load(in_ptr6 + (0))
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK, 1])
    tmp31 = tl.full([1, 1], 1, tl.int64)
    tmp32 = tmp30 + tmp31
    tmp33 = tl.full([XBLOCK, 1], 960, tl.int32)
    tmp34 = tmp32 + tmp33
    tmp35 = tmp32 < 0
    tmp36 = tl.where(tmp35, tmp34, tmp32)
    tl.device_assert((0 <= tmp36) & (tmp36 < 960), "index out of bounds: 0 <= tmp36 < 960")
    tmp38 = tmp27.to(tl.float32)
    tl.store(out_ptr2 + (x2 + 128*tmp36 + 122880*x3), tmp38, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/dy/cdypgw5xn2ggjh76dbs36y2wh2uxs76gfw63p3sm5fl3mo62grwy.py
# Topologically Sorted Source Nodes: [logits_1, argmax_1, cat_224], Original ATen: [aten.mm, aten.argmax, aten.cat]
# Source node to ATen node mapping:
#   argmax_1 => argmax_1
#   cat_224 => cat_224
#   logits_1 => convert_element_type_1807
# Graph fragment:
#   %convert_element_type_1807 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_508, torch.float16), kwargs = {})
#   %argmax_1 : [num_users=2] = call_function[target=torch.ops.aten.argmax.default](args = (%convert_element_type_1807, -1), kwargs = {})
#   %cat_224 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_2731, %view_2732, %view_2733, %view_2734], 1), kwargs = {})
triton_red_fused_argmax_cat_mm_40 = async_compile.triton('triton_red_fused_argmax_cat_mm_40', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r0_': 262144},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*i64', 'out_ptr1': '*i64', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_argmax_cat_mm_40', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'AA6F4099AE74BF369D45388847321E195BFBEE5A86FCB9B181180FAA5E25C7E8', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'r0_': 607744}}
)
@triton.jit
def triton_red_fused_argmax_cat_mm_40(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 151936
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp3 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    _tmp3_index = tl.full([XBLOCK, R0_BLOCK], 2147483647, tl.int32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        _tmp3_next, _tmp3_index_next = triton_helpers.maximum_with_index(
            _tmp3, _tmp3_index, tmp2, rindex
        )
        _tmp3 = tl.where(r0_mask, _tmp3_next, _tmp3)
        _tmp3_index = tl.where(r0_mask, _tmp3_index_next, _tmp3_index)
    tmp3_val, tmp3_idx = triton_helpers.max_with_index(_tmp3, _tmp3_index, 1)
    tmp3 = tmp3_idx[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp3, None)
    tl.store(out_ptr1 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp3, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/56/c56ijvwaqz2qtstgnol6rqynsan4xtbsmf7sncvp2ncnmzbgvaox.py
# Topologically Sorted Source Nodes: [mul_1362, sum_509, mul_1365, sum_510, index_put_113], Original ATen: [aten.mul, aten.sum, aten.index_put]
# Source node to ATen node mapping:
#   index_put_113 => index_put_113
#   mul_1362 => mul_1364
#   mul_1365 => mul_1367
#   sum_509 => sum_511
#   sum_510 => sum_512
# Graph fragment:
#   %mul_1364 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1168, %unsqueeze_1169), kwargs = {})
#   %sum_511 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_1364, [1]), kwargs = {})
#   %mul_1367 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1170, %unsqueeze_1171), kwargs = {})
#   %sum_512 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_1367, [1]), kwargs = {})
#   %index_put_113 : [num_users=2] = call_function[target=torch.ops.aten.index_put_.default](args = (%index_put_57, [None, None, %add_511], %permute_647), kwargs = {})
triton_red_fused_index_put_mul_sum_41 = async_compile.triton('triton_red_fused_index_put_mul_sum_41', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*i64', 'in_ptr2': '*fp16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp16', 'in_ptr5': '*fp16', 'in_ptr6': '*i64', 'out_ptr0': '*fp32', 'out_ptr2': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_index_put_mul_sum_41', 'mutated_arg_names': ['out_ptr2'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 2, 'backend_hash': 'AA6F4099AE74BF369D45388847321E195BFBEE5A86FCB9B181180FAA5E25C7E8', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_red_fused_index_put_mul_sum_41(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp10 = tl.load(in_ptr3 + (0))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    x0 = xindex
    _tmp25 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp31 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp21 = tl.load(in_ptr4 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp27 = tl.load(in_ptr5 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.full([XBLOCK, R0_BLOCK], 151936, tl.int32)
        tmp4 = tmp2 + tmp3
        tmp5 = tmp2 < 0
        tmp6 = tl.where(tmp5, tmp4, tmp2)
        tl.device_assert((0 <= tmp6) & (tmp6 < 151936), "index out of bounds: 0 <= tmp6 < 151936")
        tmp8 = tl.load(in_ptr2 + (r0_1 + 2048*tmp6), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp9 = tmp8.to(tl.float32)
        tmp12 = 2048.0
        tmp13 = (tmp11 / tmp12)
        tmp14 = 1e-06
        tmp15 = tmp13 + tmp14
        tmp16 = libdevice.rsqrt(tmp15)
        tmp17 = tmp9 * tmp16
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tmp0 * tmp18
        tmp20 = tmp19.to(tl.float32)
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp20 * tmp22
        tmp24 = tl.broadcast_to(tmp23, [XBLOCK, R0_BLOCK])
        tmp26 = _tmp25 + tmp24
        _tmp25 = tl.where(r0_mask & xmask, tmp26, _tmp25)
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tmp20 * tmp28
        tmp30 = tl.broadcast_to(tmp29, [XBLOCK, R0_BLOCK])
        tmp32 = _tmp31 + tmp30
        _tmp31 = tl.where(r0_mask & xmask, tmp32, _tmp31)
    tmp25 = tl.sum(_tmp25, 1)[:, None]
    tmp31 = tl.sum(_tmp31, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp25, xmask)
    x2 = (xindex % 128)
    x3 = xindex // 128
    tmp33 = tl.load(in_ptr6 + (0))
    tmp34 = tl.broadcast_to(tmp33, [XBLOCK, 1])
    tmp35 = tl.full([1, 1], 1, tl.int64)
    tmp36 = tmp34 + tmp35
    tmp37 = tmp36 + tmp35
    tmp38 = tl.full([XBLOCK, 1], 960, tl.int32)
    tmp39 = tmp37 + tmp38
    tmp40 = tmp37 < 0
    tmp41 = tl.where(tmp40, tmp39, tmp37)
    tl.device_assert((0 <= tmp41) & (tmp41 < 960), "index out of bounds: 0 <= tmp41 < 960")
    tmp43 = tmp31.to(tl.float32)
    tl.store(out_ptr2 + (x2 + 128*tmp41 + 122880*x3), tmp43, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/c5/cc5qa2lhw5hksisgyz7gyxbfrql2uzlkcep4r3fxixyktfcpkwqq.py
# Topologically Sorted Source Nodes: [convert_element_type_1815, pow_229, mean_228, mul_1368, cat_113, mul_1369, add_518, index_put_112], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
# Source node to ATen node mapping:
#   add_518 => add_518
#   cat_113 => cat_113
#   convert_element_type_1815 => convert_element_type_1821
#   index_put_112 => index_put_112
#   mean_228 => mean_228
#   mul_1368 => mul_1370
#   mul_1369 => mul_1371
#   pow_229 => pow_229
# Graph fragment:
#   %convert_element_type_1821 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_1381, torch.float32), kwargs = {})
#   %pow_229 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_1821, 2), kwargs = {})
#   %mean_228 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_229, [-1], True), kwargs = {})
#   %mul_1370 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_645, %unsqueeze_1172), kwargs = {})
#   %cat_113 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%neg_113, %slice_351], -1), kwargs = {})
#   %mul_1371 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat_113, %unsqueeze_1173), kwargs = {})
#   %add_518 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1370, %mul_1371), kwargs = {})
#   %index_put_112 : [num_users=2] = call_function[target=torch.ops.aten.index_put_.default](args = (%index_put_56, [None, None, %add_511], %add_518), kwargs = {})
triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_42 = async_compile.triton('triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_42', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp16', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'out_ptr2': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_42', 'mutated_arg_names': ['out_ptr2'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 1, 'backend_hash': 'AA6F4099AE74BF369D45388847321E195BFBEE5A86FCB9B181180FAA5E25C7E8', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_42(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 8
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 128*x0), xmask, other=0.0)
    tmp46 = tl.load(in_ptr2 + (0))
    tmp47 = tl.broadcast_to(tmp46, [XBLOCK, R0_BLOCK])
    tmp56 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last').to(tl.float32)
    tmp65 = tl.load(in_ptr3 + ((r0_1 % 64)), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp2 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp6 = tl.where(xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = r0_1
    tmp9 = tl.full([1, 1], 0, tl.int64)
    tmp10 = tmp8 >= tmp9
    tmp11 = tl.full([1, 1], 64, tl.int64)
    tmp12 = tmp8 < tmp11
    tmp13 = tl.load(in_ptr1 + (tl.broadcast_to(64 + (r0_1), [XBLOCK, R0_BLOCK])), tmp12 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp14 = tl.load(in_ptr0 + (64 + 128*x0 + (r0_1)), tmp12 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = 128.0
    tmp18 = (tmp7 / tmp17)
    tmp19 = 1e-06
    tmp20 = tmp18 + tmp19
    tmp21 = libdevice.rsqrt(tmp20)
    tmp22 = tmp16 * tmp21
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp13 * tmp23
    tmp25 = -tmp24
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp12, tmp25, tmp26)
    tmp28 = tmp8 >= tmp11
    tmp29 = tl.full([1, 1], 128, tl.int64)
    tmp30 = tmp8 < tmp29
    tmp31 = tl.load(in_ptr1 + (tl.broadcast_to((-64) + r0_1, [XBLOCK, R0_BLOCK])), tmp28 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp32 = tl.load(in_ptr0 + (128*x0 + ((-64) + r0_1)), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tmp33.to(tl.float32)
    tmp35 = 128.0
    tmp36 = (tmp7 / tmp35)
    tmp37 = 1e-06
    tmp38 = tmp36 + tmp37
    tmp39 = libdevice.rsqrt(tmp38)
    tmp40 = tmp34 * tmp39
    tmp41 = tmp40.to(tl.float32)
    tmp42 = tmp31 * tmp41
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp28, tmp42, tmp43)
    tmp45 = tl.where(tmp12, tmp27, tmp44)
    tmp48 = tl.full([1, 1], 1, tl.int64)
    tmp49 = tmp47 + tmp48
    tmp50 = tmp49 + tmp48
    tmp51 = tl.full([XBLOCK, R0_BLOCK], 960, tl.int32)
    tmp52 = tmp50 + tmp51
    tmp53 = tmp50 < 0
    tmp54 = tl.where(tmp53, tmp52, tmp50)
    tl.device_assert((0 <= tmp54) & (tmp54 < 960), "index out of bounds: 0 <= tmp54 < 960")
    tmp57 = 128.0
    tmp58 = (tmp7 / tmp57)
    tmp59 = 1e-06
    tmp60 = tmp58 + tmp59
    tmp61 = libdevice.rsqrt(tmp60)
    tmp62 = tmp2 * tmp61
    tmp63 = tmp62.to(tl.float32)
    tmp64 = tmp56 * tmp63
    tmp66 = tl_math.cos(tmp65)
    tmp67 = 1.0
    tmp68 = tmp66 * tmp67
    tmp69 = tmp68.to(tl.float32)
    tmp70 = tmp64 * tmp69
    tmp71 = tl_math.sin(tmp65)
    tmp72 = tmp71 * tmp67
    tmp73 = tmp72.to(tl.float32)
    tmp74 = tmp45 * tmp73
    tmp75 = tmp70 + tmp74
    tl.store(out_ptr2 + (r0_1 + 128*tmp54 + 122880*x0), tmp75, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/hz/chzgboap52p32uc6ce7yazkv43dwmdiagxdoai6dsvv4lqjssgsx.py
# Topologically Sorted Source Nodes: [full_default_255, full_default_253, where_112, add_519, eq_56, logical_not_112, any_57, logical_not_113, full_default_257, , sub, exp, div_56, where_113], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
# Source node to ATen node mapping:
#    => prepare_softmax_online_default_55
#   add_519 => add_519
#   any_57 => any_57
#   div_56 => div_56
#   eq_56 => eq_56
#   exp => exp_default_55
#   full_default_253 => full_default_168
#   full_default_255 => full_default_169
#   full_default_257 => full_default_170
#   logical_not_112 => logical_not_112
#   logical_not_113 => logical_not_113
#   sub => sub_tensor_55
#   where_112 => where_112
#   where_113 => where_113
# Graph fragment:
#   %full_default_169 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_168 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_112 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%expand_349, %full_default_169, %full_default_168), kwargs = {})
#   %add_519 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1389, %where_112), kwargs = {})
#   %eq_56 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%add_519, -inf), kwargs = {})
#   %logical_not_112 : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%eq_56,), kwargs = {})
#   %any_57 : [num_users=1] = call_function[target=torch.ops.aten.any.dim](args = (%logical_not_112, -1, True), kwargs = {})
#   %logical_not_113 : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%any_57,), kwargs = {})
#   %full_default_170 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 16, 1, 960], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %prepare_softmax_online_default_55 : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%add_519, -1), kwargs = {})
#   %sub_tensor_55 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_519, %getitem_110), kwargs = {})
#   %exp_default_55 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_55,), kwargs = {})
#   %div_56 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_55, %getitem_111), kwargs = {})
#   %where_113 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_not_113, %full_default_170, %div_56), kwargs = {})
triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_43 = async_compile.triton('triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_43', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_43', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 5, 'backend_hash': 'AA6F4099AE74BF369D45388847321E195BFBEE5A86FCB9B181180FAA5E25C7E8', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 0, 'r0_': 184320}}
)
@triton.jit
def triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_43(in_ptr0, in_ptr1, out_ptr3, xnumel, r0_numel):
    xnumel = 16
    XBLOCK: tl.constexpr = 1
    r0_numel = 960
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 960*x0), r0_mask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [R0_BLOCK])
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4 + tmp3
    tmp6 = r0_1
    tmp7 = tmp6 <= tmp5
    tmp8 = 0.0
    tmp9 = float("-inf")
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp0 + tmp11
    tmp13 = tmp12 == tmp9
    tmp14 = tmp13 == 0
    tmp15 = tmp14.to(tl.int64)
    tmp16 = (tmp15 != 0)
    tmp17 = tl.broadcast_to(tmp16, [R0_BLOCK])
    tmp19 = tl.where(r0_mask, tmp17, False)
    tmp20 = triton_helpers.promote_to_tensor(triton_helpers.any(tmp19, 0))
    tmp21 = tl.broadcast_to(tmp12, [R0_BLOCK])
    tmp23 = tl.broadcast_to(tmp21, [R0_BLOCK])
    tmp25 = tl.where(r0_mask, tmp23, float("-inf"))
    tmp26 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp25, 0))
    tmp27 = tmp21 - tmp26
    tmp28 = tl_math.exp(tmp27)
    tmp29 = tl.broadcast_to(tmp28, [R0_BLOCK])
    tmp31 = tl.where(r0_mask, tmp29, 0)
    tmp32 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp33 = tmp20 == 0
    tmp34 = tmp12 - tmp26
    tmp35 = tl_math.exp(tmp34)
    tmp36 = (tmp35 / tmp32)
    tmp37 = tl.where(tmp33, tmp8, tmp36)
    tl.store(out_ptr3 + (r0_1 + 960*x0), tmp37, r0_mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/qc/cqcklexw3nbp3whr7xgyv3j7mfsvrzwqmaltjqbek4cwx3r67rxh.py
# Topologically Sorted Source Nodes: [mul_1386, sum_518, mul_1389, sum_519, index_put_115], Original ATen: [aten.mul, aten.sum, aten.index_put]
# Source node to ATen node mapping:
#   index_put_115 => index_put_115
#   mul_1386 => mul_1388
#   mul_1389 => mul_1391
#   sum_518 => sum_520
#   sum_519 => sum_521
# Graph fragment:
#   %mul_1388 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1188, %unsqueeze_1189), kwargs = {})
#   %sum_520 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_1388, [1]), kwargs = {})
#   %mul_1391 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1190, %unsqueeze_1191), kwargs = {})
#   %sum_521 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_1391, [1]), kwargs = {})
#   %index_put_115 : [num_users=2] = call_function[target=torch.ops.aten.index_put_.default](args = (%index_put_59, [None, None, %add_511], %permute_658), kwargs = {})
triton_red_fused_index_put_mul_sum_44 = async_compile.triton('triton_red_fused_index_put_mul_sum_44', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*i64', 'out_ptr0': '*fp32', 'out_ptr2': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_index_put_mul_sum_44', 'mutated_arg_names': ['out_ptr2'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'AA6F4099AE74BF369D45388847321E195BFBEE5A86FCB9B181180FAA5E25C7E8', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_red_fused_index_put_mul_sum_44(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp12 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp8 = tl.load(in_ptr2 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(r0_mask & xmask, tmp7, _tmp6)
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp1 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(r0_mask & xmask, tmp13, _tmp12)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    x2 = (xindex % 128)
    x3 = xindex // 128
    tmp14 = tl.load(in_ptr3 + (0))
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, 1])
    tmp16 = tl.full([1, 1], 1, tl.int64)
    tmp17 = tmp15 + tmp16
    tmp18 = tmp17 + tmp16
    tmp19 = tl.full([XBLOCK, 1], 960, tl.int32)
    tmp20 = tmp18 + tmp19
    tmp21 = tmp18 < 0
    tmp22 = tl.where(tmp21, tmp20, tmp18)
    tl.device_assert((0 <= tmp22) & (tmp22 < 960), "index out of bounds: 0 <= tmp22 < 960")
    tmp24 = tmp12.to(tl.float32)
    tl.store(out_ptr2 + (x2 + 128*tmp22 + 122880*x3), tmp24, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/bz/cbzpgt3rpdgp7dtunew2vvbmd4suej3u3opeuxoeh6v3536sup4f.py
# Topologically Sorted Source Nodes: [mul_1410, sum_527, mul_1413, sum_528, index_put_117], Original ATen: [aten.mul, aten.sum, aten.index_put]
# Source node to ATen node mapping:
#   index_put_117 => index_put_117
#   mul_1410 => mul_1412
#   mul_1413 => mul_1415
#   sum_527 => sum_529
#   sum_528 => sum_530
# Graph fragment:
#   %mul_1412 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1208, %unsqueeze_1209), kwargs = {})
#   %sum_529 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_1412, [1]), kwargs = {})
#   %mul_1415 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1210, %unsqueeze_1211), kwargs = {})
#   %sum_530 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_1415, [1]), kwargs = {})
#   %index_put_117 : [num_users=2] = call_function[target=torch.ops.aten.index_put_.default](args = (%index_put_61, [None, None, %add_511], %permute_669), kwargs = {})
triton_red_fused_index_put_mul_sum_45 = async_compile.triton('triton_red_fused_index_put_mul_sum_45', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp16', 'in_ptr5': '*fp16', 'in_ptr6': '*i64', 'out_ptr0': '*fp32', 'out_ptr2': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_index_put_mul_sum_45', 'mutated_arg_names': ['out_ptr2'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 2, 'backend_hash': 'AA6F4099AE74BF369D45388847321E195BFBEE5A86FCB9B181180FAA5E25C7E8', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_red_fused_index_put_mul_sum_45(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    tmp6 = tl.load(in_ptr3 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    x0 = xindex
    _tmp21 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp27 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.load(in_ptr4 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp23 = tl.load(in_ptr5 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 + tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp8 = 2048.0
        tmp9 = (tmp7 / tmp8)
        tmp10 = 1e-06
        tmp11 = tmp9 + tmp10
        tmp12 = libdevice.rsqrt(tmp11)
        tmp13 = tmp5 * tmp12
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tmp0 * tmp14
        tmp16 = tmp15.to(tl.float32)
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tmp16 * tmp18
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(r0_mask & xmask, tmp22, _tmp21)
        tmp24 = tmp23.to(tl.float32)
        tmp25 = tmp16 * tmp24
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, R0_BLOCK])
        tmp28 = _tmp27 + tmp26
        _tmp27 = tl.where(r0_mask & xmask, tmp28, _tmp27)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tmp27 = tl.sum(_tmp27, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp21, xmask)
    x2 = (xindex % 128)
    x3 = xindex // 128
    tmp29 = tl.load(in_ptr6 + (0))
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK, 1])
    tmp31 = tl.full([1, 1], 1, tl.int64)
    tmp32 = tmp30 + tmp31
    tmp33 = tmp32 + tmp31
    tmp34 = tl.full([XBLOCK, 1], 960, tl.int32)
    tmp35 = tmp33 + tmp34
    tmp36 = tmp33 < 0
    tmp37 = tl.where(tmp36, tmp35, tmp33)
    tl.device_assert((0 <= tmp37) & (tmp37 < 960), "index out of bounds: 0 <= tmp37 < 960")
    tmp39 = tmp27.to(tl.float32)
    tl.store(out_ptr2 + (x2 + 128*tmp37 + 122880*x3), tmp39, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/qh/cqhcwbkoc7jqdnvx7w6tusjo5vqzqkjjxvvqfebjqatao7l3nepo.py
# Topologically Sorted Source Nodes: [mul_2039, sum_762, mul_2042, sum_763, index_put_169], Original ATen: [aten.mul, aten.sum, aten.index_put]
# Source node to ATen node mapping:
#   index_put_169 => index_put_169
#   mul_2039 => mul_2042
#   mul_2042 => mul_2045
#   sum_762 => sum_765
#   sum_763 => sum_766
# Graph fragment:
#   %mul_2042 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1744, %unsqueeze_1745), kwargs = {})
#   %sum_765 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_2042, [1]), kwargs = {})
#   %mul_2045 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1746, %unsqueeze_1747), kwargs = {})
#   %sum_766 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_2045, [1]), kwargs = {})
#   %index_put_169 : [num_users=2] = call_function[target=torch.ops.aten.index_put_.default](args = (%index_put_113, [None, None, %add_767], %permute_964), kwargs = {})
triton_red_fused_index_put_mul_sum_46 = async_compile.triton('triton_red_fused_index_put_mul_sum_46', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*i64', 'in_ptr2': '*fp16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp16', 'in_ptr5': '*fp16', 'in_ptr6': '*i64', 'out_ptr0': '*fp32', 'out_ptr2': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_index_put_mul_sum_46', 'mutated_arg_names': ['out_ptr2'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 2, 'backend_hash': 'AA6F4099AE74BF369D45388847321E195BFBEE5A86FCB9B181180FAA5E25C7E8', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_red_fused_index_put_mul_sum_46(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp10 = tl.load(in_ptr3 + (0))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    x0 = xindex
    _tmp25 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp31 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp21 = tl.load(in_ptr4 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp27 = tl.load(in_ptr5 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.full([XBLOCK, R0_BLOCK], 151936, tl.int32)
        tmp4 = tmp2 + tmp3
        tmp5 = tmp2 < 0
        tmp6 = tl.where(tmp5, tmp4, tmp2)
        tl.device_assert((0 <= tmp6) & (tmp6 < 151936), "index out of bounds: 0 <= tmp6 < 151936")
        tmp8 = tl.load(in_ptr2 + (r0_1 + 2048*tmp6), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp9 = tmp8.to(tl.float32)
        tmp12 = 2048.0
        tmp13 = (tmp11 / tmp12)
        tmp14 = 1e-06
        tmp15 = tmp13 + tmp14
        tmp16 = libdevice.rsqrt(tmp15)
        tmp17 = tmp9 * tmp16
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tmp0 * tmp18
        tmp20 = tmp19.to(tl.float32)
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp20 * tmp22
        tmp24 = tl.broadcast_to(tmp23, [XBLOCK, R0_BLOCK])
        tmp26 = _tmp25 + tmp24
        _tmp25 = tl.where(r0_mask & xmask, tmp26, _tmp25)
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tmp20 * tmp28
        tmp30 = tl.broadcast_to(tmp29, [XBLOCK, R0_BLOCK])
        tmp32 = _tmp31 + tmp30
        _tmp31 = tl.where(r0_mask & xmask, tmp32, _tmp31)
    tmp25 = tl.sum(_tmp25, 1)[:, None]
    tmp31 = tl.sum(_tmp31, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp25, xmask)
    x2 = (xindex % 128)
    x3 = xindex // 128
    tmp33 = tl.load(in_ptr6 + (0))
    tmp34 = tl.broadcast_to(tmp33, [XBLOCK, 1])
    tmp35 = tl.full([1, 1], 1, tl.int64)
    tmp36 = tmp34 + tmp35
    tmp37 = tmp36 + tmp35
    tmp38 = tmp37 + tmp35
    tmp39 = tl.full([XBLOCK, 1], 960, tl.int32)
    tmp40 = tmp38 + tmp39
    tmp41 = tmp38 < 0
    tmp42 = tl.where(tmp41, tmp40, tmp38)
    tl.device_assert((0 <= tmp42) & (tmp42 < 960), "index out of bounds: 0 <= tmp42 < 960")
    tmp44 = tmp31.to(tl.float32)
    tl.store(out_ptr2 + (x2 + 128*tmp42 + 122880*x3), tmp44, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/na/cnaahw4ktiiwf2leoo4co5hxb57gpqkawjeb7kw6soxnrfrrzwnq.py
# Topologically Sorted Source Nodes: [convert_element_type_2716, pow_342, mean_341, mul_2045, cat_169, mul_2046, add_774, index_put_168], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
# Source node to ATen node mapping:
#   add_774 => add_774
#   cat_169 => cat_169
#   convert_element_type_2716 => convert_element_type_2725
#   index_put_168 => index_put_168
#   mean_341 => mean_341
#   mul_2045 => mul_2048
#   mul_2046 => mul_2049
#   pow_342 => pow_342
# Graph fragment:
#   %convert_element_type_2725 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_2064, torch.float32), kwargs = {})
#   %pow_342 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_2725, 2), kwargs = {})
#   %mean_341 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_342, [-1], True), kwargs = {})
#   %mul_2048 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_962, %unsqueeze_1748), kwargs = {})
#   %cat_169 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%neg_169, %slice_523], -1), kwargs = {})
#   %mul_2049 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat_169, %unsqueeze_1749), kwargs = {})
#   %add_774 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2048, %mul_2049), kwargs = {})
#   %index_put_168 : [num_users=2] = call_function[target=torch.ops.aten.index_put_.default](args = (%index_put_112, [None, None, %add_767], %add_774), kwargs = {})
triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_47 = async_compile.triton('triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_47', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp16', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'out_ptr2': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_47', 'mutated_arg_names': ['out_ptr2'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 1, 'backend_hash': 'AA6F4099AE74BF369D45388847321E195BFBEE5A86FCB9B181180FAA5E25C7E8', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_47(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 8
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 128*x0), xmask, other=0.0)
    tmp46 = tl.load(in_ptr2 + (0))
    tmp47 = tl.broadcast_to(tmp46, [XBLOCK, R0_BLOCK])
    tmp57 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last').to(tl.float32)
    tmp66 = tl.load(in_ptr3 + ((r0_1 % 64)), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp2 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp6 = tl.where(xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = r0_1
    tmp9 = tl.full([1, 1], 0, tl.int64)
    tmp10 = tmp8 >= tmp9
    tmp11 = tl.full([1, 1], 64, tl.int64)
    tmp12 = tmp8 < tmp11
    tmp13 = tl.load(in_ptr1 + (tl.broadcast_to(64 + (r0_1), [XBLOCK, R0_BLOCK])), tmp12 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp14 = tl.load(in_ptr0 + (64 + 128*x0 + (r0_1)), tmp12 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = 128.0
    tmp18 = (tmp7 / tmp17)
    tmp19 = 1e-06
    tmp20 = tmp18 + tmp19
    tmp21 = libdevice.rsqrt(tmp20)
    tmp22 = tmp16 * tmp21
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp13 * tmp23
    tmp25 = -tmp24
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp12, tmp25, tmp26)
    tmp28 = tmp8 >= tmp11
    tmp29 = tl.full([1, 1], 128, tl.int64)
    tmp30 = tmp8 < tmp29
    tmp31 = tl.load(in_ptr1 + (tl.broadcast_to((-64) + r0_1, [XBLOCK, R0_BLOCK])), tmp28 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp32 = tl.load(in_ptr0 + (128*x0 + ((-64) + r0_1)), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tmp33.to(tl.float32)
    tmp35 = 128.0
    tmp36 = (tmp7 / tmp35)
    tmp37 = 1e-06
    tmp38 = tmp36 + tmp37
    tmp39 = libdevice.rsqrt(tmp38)
    tmp40 = tmp34 * tmp39
    tmp41 = tmp40.to(tl.float32)
    tmp42 = tmp31 * tmp41
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp28, tmp42, tmp43)
    tmp45 = tl.where(tmp12, tmp27, tmp44)
    tmp48 = tl.full([1, 1], 1, tl.int64)
    tmp49 = tmp47 + tmp48
    tmp50 = tmp49 + tmp48
    tmp51 = tmp50 + tmp48
    tmp52 = tl.full([XBLOCK, R0_BLOCK], 960, tl.int32)
    tmp53 = tmp51 + tmp52
    tmp54 = tmp51 < 0
    tmp55 = tl.where(tmp54, tmp53, tmp51)
    tl.device_assert((0 <= tmp55) & (tmp55 < 960), "index out of bounds: 0 <= tmp55 < 960")
    tmp58 = 128.0
    tmp59 = (tmp7 / tmp58)
    tmp60 = 1e-06
    tmp61 = tmp59 + tmp60
    tmp62 = libdevice.rsqrt(tmp61)
    tmp63 = tmp2 * tmp62
    tmp64 = tmp63.to(tl.float32)
    tmp65 = tmp57 * tmp64
    tmp67 = tl_math.cos(tmp66)
    tmp68 = 1.0
    tmp69 = tmp67 * tmp68
    tmp70 = tmp69.to(tl.float32)
    tmp71 = tmp65 * tmp70
    tmp72 = tl_math.sin(tmp66)
    tmp73 = tmp72 * tmp68
    tmp74 = tmp73.to(tl.float32)
    tmp75 = tmp45 * tmp74
    tmp76 = tmp71 + tmp75
    tl.store(out_ptr2 + (r0_1 + 128*tmp55 + 122880*x0), tmp76, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/jk/cjk3m7tahvy2s2plg6qb57cymxgmg4jrgeuglfwc2o5zdb266zvo.py
# Topologically Sorted Source Nodes: [full_default_423, full_default_421, where_168, add_775, eq_84, logical_not_168, any_85, logical_not_169, full_default_425, , sub, exp, div_84, where_169], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
# Source node to ATen node mapping:
#    => prepare_softmax_online_default_27
#   add_775 => add_775
#   any_85 => any_85
#   div_84 => div_84
#   eq_84 => eq_84
#   exp => exp_default_27
#   full_default_421 => full_default_252
#   full_default_423 => full_default_253
#   full_default_425 => full_default_254
#   logical_not_168 => logical_not_168
#   logical_not_169 => logical_not_169
#   sub => sub_tensor_27
#   where_168 => where_168
#   where_169 => where_169
# Graph fragment:
#   %full_default_253 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_252 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_168 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%expand_523, %full_default_253, %full_default_252), kwargs = {})
#   %add_775 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_2072, %where_168), kwargs = {})
#   %eq_84 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%add_775, -inf), kwargs = {})
#   %logical_not_168 : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%eq_84,), kwargs = {})
#   %any_85 : [num_users=1] = call_function[target=torch.ops.aten.any.dim](args = (%logical_not_168, -1, True), kwargs = {})
#   %logical_not_169 : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%any_85,), kwargs = {})
#   %full_default_254 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 16, 1, 960], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %prepare_softmax_online_default_27 : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%add_775, -1), kwargs = {})
#   %sub_tensor_27 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_775, %getitem_54), kwargs = {})
#   %exp_default_27 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_27,), kwargs = {})
#   %div_84 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_27, %getitem_55), kwargs = {})
#   %where_169 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_not_169, %full_default_254, %div_84), kwargs = {})
triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_48 = async_compile.triton('triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_48', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_48', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 5, 'backend_hash': 'AA6F4099AE74BF369D45388847321E195BFBEE5A86FCB9B181180FAA5E25C7E8', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 0, 'r0_': 184320}}
)
@triton.jit
def triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_48(in_out_ptr0, in_ptr0, xnumel, r0_numel):
    xnumel = 16
    XBLOCK: tl.constexpr = 1
    r0_numel = 960
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 960*x0), r0_mask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (0))
    tmp2 = tl.broadcast_to(tmp1, [R0_BLOCK])
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4 + tmp3
    tmp6 = tmp5 + tmp3
    tmp7 = r0_1
    tmp8 = tmp7 <= tmp6
    tmp9 = 0.0
    tmp10 = float("-inf")
    tmp11 = tl.where(tmp8, tmp9, tmp10)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp0 + tmp12
    tmp14 = tmp13 == tmp10
    tmp15 = tmp14 == 0
    tmp16 = tmp15.to(tl.int64)
    tmp17 = (tmp16 != 0)
    tmp18 = tl.broadcast_to(tmp17, [R0_BLOCK])
    tmp20 = tl.where(r0_mask, tmp18, False)
    tmp21 = triton_helpers.promote_to_tensor(triton_helpers.any(tmp20, 0))
    tmp22 = tl.broadcast_to(tmp13, [R0_BLOCK])
    tmp24 = tl.broadcast_to(tmp22, [R0_BLOCK])
    tmp26 = tl.where(r0_mask, tmp24, float("-inf"))
    tmp27 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp26, 0))
    tmp28 = tmp22 - tmp27
    tmp29 = tl_math.exp(tmp28)
    tmp30 = tl.broadcast_to(tmp29, [R0_BLOCK])
    tmp32 = tl.where(r0_mask, tmp30, 0)
    tmp33 = triton_helpers.promote_to_tensor(tl.sum(tmp32, 0))
    tmp34 = tmp21 == 0
    tmp35 = tmp13 - tmp27
    tmp36 = tl_math.exp(tmp35)
    tmp37 = (tmp36 / tmp33)
    tmp38 = tl.where(tmp34, tmp9, tmp37)
    tl.store(in_out_ptr0 + (r0_1 + 960*x0), tmp38, r0_mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/li/clinturh3ngbcxgiv73b3oygrgyu7pavyybhb3lp54iuqczvph6s.py
# Topologically Sorted Source Nodes: [mul_2063, sum_771, mul_2066, sum_772, index_put_171], Original ATen: [aten.mul, aten.sum, aten.index_put]
# Source node to ATen node mapping:
#   index_put_171 => index_put_171
#   mul_2063 => mul_2066
#   mul_2066 => mul_2069
#   sum_771 => sum_774
#   sum_772 => sum_775
# Graph fragment:
#   %mul_2066 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1764, %unsqueeze_1765), kwargs = {})
#   %sum_774 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_2066, [1]), kwargs = {})
#   %mul_2069 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1766, %unsqueeze_1767), kwargs = {})
#   %sum_775 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_2069, [1]), kwargs = {})
#   %index_put_171 : [num_users=2] = call_function[target=torch.ops.aten.index_put_.default](args = (%index_put_115, [None, None, %add_767], %permute_975), kwargs = {})
triton_red_fused_index_put_mul_sum_49 = async_compile.triton('triton_red_fused_index_put_mul_sum_49', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*i64', 'out_ptr0': '*fp32', 'out_ptr2': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_index_put_mul_sum_49', 'mutated_arg_names': ['out_ptr2'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'AA6F4099AE74BF369D45388847321E195BFBEE5A86FCB9B181180FAA5E25C7E8', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_red_fused_index_put_mul_sum_49(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp12 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp8 = tl.load(in_ptr2 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(r0_mask & xmask, tmp7, _tmp6)
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp1 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(r0_mask & xmask, tmp13, _tmp12)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    x2 = (xindex % 128)
    x3 = xindex // 128
    tmp14 = tl.load(in_ptr3 + (0))
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, 1])
    tmp16 = tl.full([1, 1], 1, tl.int64)
    tmp17 = tmp15 + tmp16
    tmp18 = tmp17 + tmp16
    tmp19 = tmp18 + tmp16
    tmp20 = tl.full([XBLOCK, 1], 960, tl.int32)
    tmp21 = tmp19 + tmp20
    tmp22 = tmp19 < 0
    tmp23 = tl.where(tmp22, tmp21, tmp19)
    tl.device_assert((0 <= tmp23) & (tmp23 < 960), "index out of bounds: 0 <= tmp23 < 960")
    tmp25 = tmp12.to(tl.float32)
    tl.store(out_ptr2 + (x2 + 128*tmp23 + 122880*x3), tmp25, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/wm/cwmphnkuqxmy6xdfvfd5lnhhp2hechx2bssn3kkdl7o56lvhuisg.py
# Topologically Sorted Source Nodes: [mul_2087, sum_780, mul_2090, sum_781, index_put_173], Original ATen: [aten.mul, aten.sum, aten.index_put]
# Source node to ATen node mapping:
#   index_put_173 => index_put_173
#   mul_2087 => mul_2090
#   mul_2090 => mul_2093
#   sum_780 => sum_783
#   sum_781 => sum_784
# Graph fragment:
#   %mul_2090 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1784, %unsqueeze_1785), kwargs = {})
#   %sum_783 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_2090, [1]), kwargs = {})
#   %mul_2093 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1786, %unsqueeze_1787), kwargs = {})
#   %sum_784 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_2093, [1]), kwargs = {})
#   %index_put_173 : [num_users=2] = call_function[target=torch.ops.aten.index_put_.default](args = (%index_put_117, [None, None, %add_767], %permute_986), kwargs = {})
triton_red_fused_index_put_mul_sum_50 = async_compile.triton('triton_red_fused_index_put_mul_sum_50', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp16', 'in_ptr5': '*fp16', 'in_ptr6': '*i64', 'out_ptr0': '*fp32', 'out_ptr2': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_index_put_mul_sum_50', 'mutated_arg_names': ['out_ptr2'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 2, 'backend_hash': 'AA6F4099AE74BF369D45388847321E195BFBEE5A86FCB9B181180FAA5E25C7E8', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_red_fused_index_put_mul_sum_50(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    tmp6 = tl.load(in_ptr3 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    x0 = xindex
    _tmp21 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp27 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.load(in_ptr4 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp23 = tl.load(in_ptr5 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 + tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp8 = 2048.0
        tmp9 = (tmp7 / tmp8)
        tmp10 = 1e-06
        tmp11 = tmp9 + tmp10
        tmp12 = libdevice.rsqrt(tmp11)
        tmp13 = tmp5 * tmp12
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tmp0 * tmp14
        tmp16 = tmp15.to(tl.float32)
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tmp16 * tmp18
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(r0_mask & xmask, tmp22, _tmp21)
        tmp24 = tmp23.to(tl.float32)
        tmp25 = tmp16 * tmp24
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, R0_BLOCK])
        tmp28 = _tmp27 + tmp26
        _tmp27 = tl.where(r0_mask & xmask, tmp28, _tmp27)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tmp27 = tl.sum(_tmp27, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp21, xmask)
    x2 = (xindex % 128)
    x3 = xindex // 128
    tmp29 = tl.load(in_ptr6 + (0))
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK, 1])
    tmp31 = tl.full([1, 1], 1, tl.int64)
    tmp32 = tmp30 + tmp31
    tmp33 = tmp32 + tmp31
    tmp34 = tmp33 + tmp31
    tmp35 = tl.full([XBLOCK, 1], 960, tl.int32)
    tmp36 = tmp34 + tmp35
    tmp37 = tmp34 < 0
    tmp38 = tl.where(tmp37, tmp36, tmp34)
    tl.device_assert((0 <= tmp38) & (tmp38 < 960), "index out of bounds: 0 <= tmp38 < 960")
    tmp40 = tmp27.to(tl.float32)
    tl.store(out_ptr2 + (x2 + 128*tmp38 + 122880*x3), tmp40, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/sh/cshsjmabth77hh3zsaevpq6vhx5qjjw4tdb5mqta7kwvkkufsqrb.py
# Topologically Sorted Source Nodes: [logits_3, argmax_3, cat_224], Original ATen: [aten.mm, aten.argmax, aten.cat]
# Source node to ATen node mapping:
#   argmax_3 => argmax_3
#   cat_224 => cat_224
#   logits_3 => convert_element_type_3615
# Graph fragment:
#   %convert_element_type_3615 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_1016, torch.float16), kwargs = {})
#   %argmax_3 : [num_users=1] = call_function[target=torch.ops.aten.argmax.default](args = (%convert_element_type_3615, -1), kwargs = {})
#   %cat_224 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_2731, %view_2732, %view_2733, %view_2734], 1), kwargs = {})
triton_red_fused_argmax_cat_mm_51 = async_compile.triton('triton_red_fused_argmax_cat_mm_51', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r0_': 262144},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*i64', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_argmax_cat_mm_51', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'AA6F4099AE74BF369D45388847321E195BFBEE5A86FCB9B181180FAA5E25C7E8', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'r0_': 607744}}
)
@triton.jit
def triton_red_fused_argmax_cat_mm_51(in_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 151936
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp3 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    _tmp3_index = tl.full([XBLOCK, R0_BLOCK], 2147483647, tl.int32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        _tmp3_next, _tmp3_index_next = triton_helpers.maximum_with_index(
            _tmp3, _tmp3_index, tmp2, rindex
        )
        _tmp3 = tl.where(r0_mask, _tmp3_next, _tmp3)
        _tmp3_index = tl.where(r0_mask, _tmp3_index_next, _tmp3_index)
    tmp3_val, tmp3_idx = triton_helpers.max_with_index(_tmp3, _tmp3_index, 1)
    tmp3 = tmp3_idx[:, None]
    tl.store(out_ptr1 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp3, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 1), (1, 1))
    assert_size_stride(arg1_1, (1, ), (1, ))
    assert_size_stride(arg2_1, (12288, 2048), (2048, 1))
    assert_size_stride(arg3_1, (2048, 6144), (6144, 1))
    assert_size_stride(arg4_1, (2048, ), (1, ))
    assert_size_stride(arg5_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg6_1, (128, ), (1, ))
    assert_size_stride(arg7_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg8_1, (128, ), (1, ))
    assert_size_stride(arg9_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg10_1, (1, 8, 960, 128), (983040, 122880, 128, 1))
    assert_size_stride(arg11_1, (1, 8, 960, 128), (983040, 122880, 128, 1))
    assert_size_stride(arg12_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg13_1, (2048, ), (1, ))
    assert_size_stride(arg14_1, (12288, 2048), (2048, 1))
    assert_size_stride(arg15_1, (2048, 6144), (6144, 1))
    assert_size_stride(arg16_1, (2048, ), (1, ))
    assert_size_stride(arg17_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg18_1, (128, ), (1, ))
    assert_size_stride(arg19_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg20_1, (128, ), (1, ))
    assert_size_stride(arg21_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg22_1, (1, 8, 960, 128), (983040, 122880, 128, 1))
    assert_size_stride(arg23_1, (1, 8, 960, 128), (983040, 122880, 128, 1))
    assert_size_stride(arg24_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg25_1, (2048, ), (1, ))
    assert_size_stride(arg26_1, (12288, 2048), (2048, 1))
    assert_size_stride(arg27_1, (2048, 6144), (6144, 1))
    assert_size_stride(arg28_1, (2048, ), (1, ))
    assert_size_stride(arg29_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg30_1, (128, ), (1, ))
    assert_size_stride(arg31_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg32_1, (128, ), (1, ))
    assert_size_stride(arg33_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg34_1, (1, 8, 960, 128), (983040, 122880, 128, 1))
    assert_size_stride(arg35_1, (1, 8, 960, 128), (983040, 122880, 128, 1))
    assert_size_stride(arg36_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg37_1, (2048, ), (1, ))
    assert_size_stride(arg38_1, (12288, 2048), (2048, 1))
    assert_size_stride(arg39_1, (2048, 6144), (6144, 1))
    assert_size_stride(arg40_1, (2048, ), (1, ))
    assert_size_stride(arg41_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg42_1, (128, ), (1, ))
    assert_size_stride(arg43_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg44_1, (128, ), (1, ))
    assert_size_stride(arg45_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg46_1, (1, 8, 960, 128), (983040, 122880, 128, 1))
    assert_size_stride(arg47_1, (1, 8, 960, 128), (983040, 122880, 128, 1))
    assert_size_stride(arg48_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg49_1, (2048, ), (1, ))
    assert_size_stride(arg50_1, (12288, 2048), (2048, 1))
    assert_size_stride(arg51_1, (2048, 6144), (6144, 1))
    assert_size_stride(arg52_1, (2048, ), (1, ))
    assert_size_stride(arg53_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg54_1, (128, ), (1, ))
    assert_size_stride(arg55_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg56_1, (128, ), (1, ))
    assert_size_stride(arg57_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg58_1, (1, 8, 960, 128), (983040, 122880, 128, 1))
    assert_size_stride(arg59_1, (1, 8, 960, 128), (983040, 122880, 128, 1))
    assert_size_stride(arg60_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg61_1, (2048, ), (1, ))
    assert_size_stride(arg62_1, (12288, 2048), (2048, 1))
    assert_size_stride(arg63_1, (2048, 6144), (6144, 1))
    assert_size_stride(arg64_1, (2048, ), (1, ))
    assert_size_stride(arg65_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg66_1, (128, ), (1, ))
    assert_size_stride(arg67_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg68_1, (128, ), (1, ))
    assert_size_stride(arg69_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg70_1, (1, 8, 960, 128), (983040, 122880, 128, 1))
    assert_size_stride(arg71_1, (1, 8, 960, 128), (983040, 122880, 128, 1))
    assert_size_stride(arg72_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg73_1, (2048, ), (1, ))
    assert_size_stride(arg74_1, (12288, 2048), (2048, 1))
    assert_size_stride(arg75_1, (2048, 6144), (6144, 1))
    assert_size_stride(arg76_1, (2048, ), (1, ))
    assert_size_stride(arg77_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg78_1, (128, ), (1, ))
    assert_size_stride(arg79_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg80_1, (128, ), (1, ))
    assert_size_stride(arg81_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg82_1, (1, 8, 960, 128), (983040, 122880, 128, 1))
    assert_size_stride(arg83_1, (1, 8, 960, 128), (983040, 122880, 128, 1))
    assert_size_stride(arg84_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg85_1, (2048, ), (1, ))
    assert_size_stride(arg86_1, (12288, 2048), (2048, 1))
    assert_size_stride(arg87_1, (2048, 6144), (6144, 1))
    assert_size_stride(arg88_1, (2048, ), (1, ))
    assert_size_stride(arg89_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg90_1, (128, ), (1, ))
    assert_size_stride(arg91_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg92_1, (128, ), (1, ))
    assert_size_stride(arg93_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg94_1, (1, 8, 960, 128), (983040, 122880, 128, 1))
    assert_size_stride(arg95_1, (1, 8, 960, 128), (983040, 122880, 128, 1))
    assert_size_stride(arg96_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg97_1, (2048, ), (1, ))
    assert_size_stride(arg98_1, (12288, 2048), (2048, 1))
    assert_size_stride(arg99_1, (2048, 6144), (6144, 1))
    assert_size_stride(arg100_1, (2048, ), (1, ))
    assert_size_stride(arg101_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg102_1, (128, ), (1, ))
    assert_size_stride(arg103_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg104_1, (128, ), (1, ))
    assert_size_stride(arg105_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg106_1, (1, 8, 960, 128), (983040, 122880, 128, 1))
    assert_size_stride(arg107_1, (1, 8, 960, 128), (983040, 122880, 128, 1))
    assert_size_stride(arg108_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg109_1, (2048, ), (1, ))
    assert_size_stride(arg110_1, (12288, 2048), (2048, 1))
    assert_size_stride(arg111_1, (2048, 6144), (6144, 1))
    assert_size_stride(arg112_1, (2048, ), (1, ))
    assert_size_stride(arg113_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg114_1, (128, ), (1, ))
    assert_size_stride(arg115_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg116_1, (128, ), (1, ))
    assert_size_stride(arg117_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg118_1, (1, 8, 960, 128), (983040, 122880, 128, 1))
    assert_size_stride(arg119_1, (1, 8, 960, 128), (983040, 122880, 128, 1))
    assert_size_stride(arg120_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg121_1, (2048, ), (1, ))
    assert_size_stride(arg122_1, (12288, 2048), (2048, 1))
    assert_size_stride(arg123_1, (2048, 6144), (6144, 1))
    assert_size_stride(arg124_1, (2048, ), (1, ))
    assert_size_stride(arg125_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg126_1, (128, ), (1, ))
    assert_size_stride(arg127_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg128_1, (128, ), (1, ))
    assert_size_stride(arg129_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg130_1, (1, 8, 960, 128), (983040, 122880, 128, 1))
    assert_size_stride(arg131_1, (1, 8, 960, 128), (983040, 122880, 128, 1))
    assert_size_stride(arg132_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg133_1, (2048, ), (1, ))
    assert_size_stride(arg134_1, (12288, 2048), (2048, 1))
    assert_size_stride(arg135_1, (2048, 6144), (6144, 1))
    assert_size_stride(arg136_1, (2048, ), (1, ))
    assert_size_stride(arg137_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg138_1, (128, ), (1, ))
    assert_size_stride(arg139_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg140_1, (128, ), (1, ))
    assert_size_stride(arg141_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg142_1, (1, 8, 960, 128), (983040, 122880, 128, 1))
    assert_size_stride(arg143_1, (1, 8, 960, 128), (983040, 122880, 128, 1))
    assert_size_stride(arg144_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg145_1, (2048, ), (1, ))
    assert_size_stride(arg146_1, (12288, 2048), (2048, 1))
    assert_size_stride(arg147_1, (2048, 6144), (6144, 1))
    assert_size_stride(arg148_1, (2048, ), (1, ))
    assert_size_stride(arg149_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg150_1, (128, ), (1, ))
    assert_size_stride(arg151_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg152_1, (128, ), (1, ))
    assert_size_stride(arg153_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg154_1, (1, 8, 960, 128), (983040, 122880, 128, 1))
    assert_size_stride(arg155_1, (1, 8, 960, 128), (983040, 122880, 128, 1))
    assert_size_stride(arg156_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg157_1, (2048, ), (1, ))
    assert_size_stride(arg158_1, (12288, 2048), (2048, 1))
    assert_size_stride(arg159_1, (2048, 6144), (6144, 1))
    assert_size_stride(arg160_1, (2048, ), (1, ))
    assert_size_stride(arg161_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg162_1, (128, ), (1, ))
    assert_size_stride(arg163_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg164_1, (128, ), (1, ))
    assert_size_stride(arg165_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg166_1, (1, 8, 960, 128), (983040, 122880, 128, 1))
    assert_size_stride(arg167_1, (1, 8, 960, 128), (983040, 122880, 128, 1))
    assert_size_stride(arg168_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg169_1, (2048, ), (1, ))
    assert_size_stride(arg170_1, (12288, 2048), (2048, 1))
    assert_size_stride(arg171_1, (2048, 6144), (6144, 1))
    assert_size_stride(arg172_1, (2048, ), (1, ))
    assert_size_stride(arg173_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg174_1, (128, ), (1, ))
    assert_size_stride(arg175_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg176_1, (128, ), (1, ))
    assert_size_stride(arg177_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg178_1, (1, 8, 960, 128), (983040, 122880, 128, 1))
    assert_size_stride(arg179_1, (1, 8, 960, 128), (983040, 122880, 128, 1))
    assert_size_stride(arg180_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg181_1, (2048, ), (1, ))
    assert_size_stride(arg182_1, (12288, 2048), (2048, 1))
    assert_size_stride(arg183_1, (2048, 6144), (6144, 1))
    assert_size_stride(arg184_1, (2048, ), (1, ))
    assert_size_stride(arg185_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg186_1, (128, ), (1, ))
    assert_size_stride(arg187_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg188_1, (128, ), (1, ))
    assert_size_stride(arg189_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg190_1, (1, 8, 960, 128), (983040, 122880, 128, 1))
    assert_size_stride(arg191_1, (1, 8, 960, 128), (983040, 122880, 128, 1))
    assert_size_stride(arg192_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg193_1, (2048, ), (1, ))
    assert_size_stride(arg194_1, (12288, 2048), (2048, 1))
    assert_size_stride(arg195_1, (2048, 6144), (6144, 1))
    assert_size_stride(arg196_1, (2048, ), (1, ))
    assert_size_stride(arg197_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg198_1, (128, ), (1, ))
    assert_size_stride(arg199_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg200_1, (128, ), (1, ))
    assert_size_stride(arg201_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg202_1, (1, 8, 960, 128), (983040, 122880, 128, 1))
    assert_size_stride(arg203_1, (1, 8, 960, 128), (983040, 122880, 128, 1))
    assert_size_stride(arg204_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg205_1, (2048, ), (1, ))
    assert_size_stride(arg206_1, (12288, 2048), (2048, 1))
    assert_size_stride(arg207_1, (2048, 6144), (6144, 1))
    assert_size_stride(arg208_1, (2048, ), (1, ))
    assert_size_stride(arg209_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg210_1, (128, ), (1, ))
    assert_size_stride(arg211_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg212_1, (128, ), (1, ))
    assert_size_stride(arg213_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg214_1, (1, 8, 960, 128), (983040, 122880, 128, 1))
    assert_size_stride(arg215_1, (1, 8, 960, 128), (983040, 122880, 128, 1))
    assert_size_stride(arg216_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg217_1, (2048, ), (1, ))
    assert_size_stride(arg218_1, (12288, 2048), (2048, 1))
    assert_size_stride(arg219_1, (2048, 6144), (6144, 1))
    assert_size_stride(arg220_1, (2048, ), (1, ))
    assert_size_stride(arg221_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg222_1, (128, ), (1, ))
    assert_size_stride(arg223_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg224_1, (128, ), (1, ))
    assert_size_stride(arg225_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg226_1, (1, 8, 960, 128), (983040, 122880, 128, 1))
    assert_size_stride(arg227_1, (1, 8, 960, 128), (983040, 122880, 128, 1))
    assert_size_stride(arg228_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg229_1, (2048, ), (1, ))
    assert_size_stride(arg230_1, (12288, 2048), (2048, 1))
    assert_size_stride(arg231_1, (2048, 6144), (6144, 1))
    assert_size_stride(arg232_1, (2048, ), (1, ))
    assert_size_stride(arg233_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg234_1, (128, ), (1, ))
    assert_size_stride(arg235_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg236_1, (128, ), (1, ))
    assert_size_stride(arg237_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg238_1, (1, 8, 960, 128), (983040, 122880, 128, 1))
    assert_size_stride(arg239_1, (1, 8, 960, 128), (983040, 122880, 128, 1))
    assert_size_stride(arg240_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg241_1, (2048, ), (1, ))
    assert_size_stride(arg242_1, (12288, 2048), (2048, 1))
    assert_size_stride(arg243_1, (2048, 6144), (6144, 1))
    assert_size_stride(arg244_1, (2048, ), (1, ))
    assert_size_stride(arg245_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg246_1, (128, ), (1, ))
    assert_size_stride(arg247_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg248_1, (128, ), (1, ))
    assert_size_stride(arg249_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg250_1, (1, 8, 960, 128), (983040, 122880, 128, 1))
    assert_size_stride(arg251_1, (1, 8, 960, 128), (983040, 122880, 128, 1))
    assert_size_stride(arg252_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg253_1, (2048, ), (1, ))
    assert_size_stride(arg254_1, (12288, 2048), (2048, 1))
    assert_size_stride(arg255_1, (2048, 6144), (6144, 1))
    assert_size_stride(arg256_1, (2048, ), (1, ))
    assert_size_stride(arg257_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg258_1, (128, ), (1, ))
    assert_size_stride(arg259_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg260_1, (128, ), (1, ))
    assert_size_stride(arg261_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg262_1, (1, 8, 960, 128), (983040, 122880, 128, 1))
    assert_size_stride(arg263_1, (1, 8, 960, 128), (983040, 122880, 128, 1))
    assert_size_stride(arg264_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg265_1, (2048, ), (1, ))
    assert_size_stride(arg266_1, (12288, 2048), (2048, 1))
    assert_size_stride(arg267_1, (2048, 6144), (6144, 1))
    assert_size_stride(arg268_1, (2048, ), (1, ))
    assert_size_stride(arg269_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg270_1, (128, ), (1, ))
    assert_size_stride(arg271_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg272_1, (128, ), (1, ))
    assert_size_stride(arg273_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg274_1, (1, 8, 960, 128), (983040, 122880, 128, 1))
    assert_size_stride(arg275_1, (1, 8, 960, 128), (983040, 122880, 128, 1))
    assert_size_stride(arg276_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg277_1, (2048, ), (1, ))
    assert_size_stride(arg278_1, (12288, 2048), (2048, 1))
    assert_size_stride(arg279_1, (2048, 6144), (6144, 1))
    assert_size_stride(arg280_1, (2048, ), (1, ))
    assert_size_stride(arg281_1, (1, 1), (1, 1))
    assert_size_stride(arg282_1, (151936, 2048), (2048, 1))
    assert_size_stride(arg283_1, (64, ), (1, ))
    assert_size_stride(arg284_1, (2048, ), (1, ))
    assert_size_stride(arg285_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg286_1, (128, ), (1, ))
    assert_size_stride(arg287_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg288_1, (128, ), (1, ))
    assert_size_stride(arg289_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg290_1, (1, 8, 960, 128), (983040, 122880, 128, 1))
    assert_size_stride(arg291_1, (1, 8, 960, 128), (983040, 122880, 128, 1))
    assert_size_stride(arg292_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg293_1, (2048, ), (1, ))
    assert_size_stride(arg294_1, (12288, 2048), (2048, 1))
    assert_size_stride(arg295_1, (2048, 6144), (6144, 1))
    assert_size_stride(arg296_1, (2048, ), (1, ))
    assert_size_stride(arg297_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg298_1, (128, ), (1, ))
    assert_size_stride(arg299_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg300_1, (128, ), (1, ))
    assert_size_stride(arg301_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg302_1, (1, 8, 960, 128), (983040, 122880, 128, 1))
    assert_size_stride(arg303_1, (1, 8, 960, 128), (983040, 122880, 128, 1))
    assert_size_stride(arg304_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg305_1, (2048, ), (1, ))
    assert_size_stride(arg306_1, (12288, 2048), (2048, 1))
    assert_size_stride(arg307_1, (2048, 6144), (6144, 1))
    assert_size_stride(arg308_1, (2048, ), (1, ))
    assert_size_stride(arg309_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg310_1, (128, ), (1, ))
    assert_size_stride(arg311_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg312_1, (128, ), (1, ))
    assert_size_stride(arg313_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg314_1, (1, 8, 960, 128), (983040, 122880, 128, 1))
    assert_size_stride(arg315_1, (1, 8, 960, 128), (983040, 122880, 128, 1))
    assert_size_stride(arg316_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg317_1, (2048, ), (1, ))
    assert_size_stride(arg318_1, (12288, 2048), (2048, 1))
    assert_size_stride(arg319_1, (2048, 6144), (6144, 1))
    assert_size_stride(arg320_1, (2048, ), (1, ))
    assert_size_stride(arg321_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg322_1, (128, ), (1, ))
    assert_size_stride(arg323_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg324_1, (128, ), (1, ))
    assert_size_stride(arg325_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg326_1, (1, 8, 960, 128), (983040, 122880, 128, 1))
    assert_size_stride(arg327_1, (1, 8, 960, 128), (983040, 122880, 128, 1))
    assert_size_stride(arg328_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg329_1, (2048, ), (1, ))
    assert_size_stride(arg330_1, (12288, 2048), (2048, 1))
    assert_size_stride(arg331_1, (2048, 6144), (6144, 1))
    assert_size_stride(arg332_1, (2048, ), (1, ))
    assert_size_stride(arg333_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg334_1, (128, ), (1, ))
    assert_size_stride(arg335_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg336_1, (128, ), (1, ))
    assert_size_stride(arg337_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg338_1, (1, 8, 960, 128), (983040, 122880, 128, 1))
    assert_size_stride(arg339_1, (1, 8, 960, 128), (983040, 122880, 128, 1))
    assert_size_stride(arg340_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg341_1, (2048, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 1, 1), (1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [embedding, convert_element_type_3, pow_1, mean], Original ATen: [aten.embedding, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_embedding_mean_pow_0.run(arg0_1, arg282_1, buf0, 1, 2048, stream=stream0)
        buf3 = empty_strided_cuda((1, 1, 64), (64, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(arg283_1, arg1_1, arg281_1, buf3, 64, stream=stream0)
        buf666 = empty_strided_cuda((1, 1, 64), (64, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(arg283_1, arg1_1, arg281_1, buf666, 64, stream=stream0)
        buf1525 = empty_strided_cuda((1, 1, 64), (64, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(arg283_1, arg1_1, arg281_1, buf1525, 64, stream=stream0)
        buf2384 = empty_strided_cuda((1, 1, 64), (64, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(arg283_1, arg1_1, arg281_1, buf2384, 64, stream=stream0)
        buf1 = empty_strided_cuda((1, 2048), (2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_5, sum_2], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_5.run(arg284_1, arg0_1, arg282_1, buf0, arg285_1, buf1, 2048, 2048, stream=stream0)
        buf6 = empty_strided_cuda((1, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_8, sum_3, mul_11, sum_4, index_put_1], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_6.run(arg284_1, arg0_1, arg282_1, buf0, arg287_1, arg289_1, arg1_1, buf6, arg291_1, 1024, 2048, stream=stream0)
        buf4 = empty_strided_cuda((3, 1, 1, 64), (64, 192, 192, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_generalized_scatter_3], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_7.run(buf3, arg283_1, arg1_1, arg281_1, buf4, 192, stream=stream0)
        del buf3
        buf667 = empty_strided_cuda((3, 1, 1, 64), (64, 192, 192, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_generalized_scatter_7], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(buf666, arg283_1, arg1_1, arg281_1, buf667, 192, stream=stream0)
        del buf666
        buf1526 = empty_strided_cuda((3, 1, 1, 64), (64, 192, 192, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_generalized_scatter_11], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_9.run(buf1525, arg283_1, arg1_1, arg281_1, buf1526, 192, stream=stream0)
        del buf1525
        buf2385 = empty_strided_cuda((3, 1, 1, 64), (64, 192, 192, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_generalized_scatter_15], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_10.run(buf2384, arg283_1, arg1_1, arg281_1, buf2385, 192, stream=stream0)
        del arg281_1
        del arg283_1
        del buf2384
        buf10 = empty_strided_cuda((1, 16, 1, 128), (2048, 128, 2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [convert_element_type_8, pow_2, mean_1, mul_12, cat, mul_13, add_5, convert_element_type_18, mul_16], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf1, arg286_1, buf4, buf10, 16, 128, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_13, pow_3, mean_2, mul_14, cat_1, mul_15, add_6, index_put], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_12.run(buf6, arg288_1, arg1_1, buf4, arg290_1, 8, 128, stream=stream0)
        buf11 = empty_strided_cuda((16, 1, 960), (960, 960, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_18, sum_5], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf10, arg290_1, buf11, 15360, 128, stream=stream0)
        buf17 = empty_strided_cuda((1, 16, 1, 960), (15360, 960, 15360, 1), torch.float32)
        # Topologically Sorted Source Nodes: [full_default_1, full_default, where, add_7, eq, logical_not, any_1, logical_not_1, full_default_2, , sub, exp, div, where_1], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_14.run(buf11, arg1_1, buf17, 16, 960, stream=stream0)
        buf18 = empty_strided_cuda((16, 1, 128, 8), (1024, 16384, 1, 128), torch.float32)
        # Topologically Sorted Source Nodes: [mul_19, sum_7], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf17, arg291_1, buf18, 16384, 120, stream=stream0)
        buf19 = reinterpret_tensor(buf10, (16, 1, 128), (128, 128, 1), 0); del buf10  # reuse
        # Topologically Sorted Source Nodes: [mul_19, sum_7], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf18, buf19, 2048, 8, stream=stream0)
        buf20 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [mul_20, sum_8], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_17.run(buf19, arg292_1, buf20, 2048, 2048, stream=stream0)
        buf22 = reinterpret_tensor(buf19, (1, 2048), (2048, 1), 0); del buf19  # reuse
        # Topologically Sorted Source Nodes: [embedding, add_8, convert_element_type_26, pow_4, mean_3, convert_element_type_28], Original ATen: [aten.embedding, aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_embedding_mean_pow_18.run(arg0_1, arg282_1, buf20, arg293_1, buf22, 1, 2048, stream=stream0)
        buf23 = empty_strided_cuda((1, 12288), (12288, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_23, sum_9], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_19.run(buf22, arg294_1, buf23, 12288, 2048, stream=stream0)
        buf24 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [mul_26, sum_10], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf23, arg295_1, buf24, 2048, 6144, stream=stream0)
        buf26 = empty_strided_cuda((1, 1, 2048), (2048, 2048, 1), torch.float16)
        # Topologically Sorted Source Nodes: [embedding, add_8, add_10, convert_element_type_36, pow_5, mean_4, add_11, rsqrt_4, mul_27, convert_element_type_37, mul_28], Original ATen: [aten.embedding, aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_embedding_mean_mul_pow_rsqrt_21.run(arg0_1, arg282_1, buf20, buf24, arg296_1, buf26, 1, 2048, stream=stream0)
        buf27 = empty_strided_cuda((1, 2048), (2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_29, sum_11], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_22.run(buf26, arg297_1, buf27, 2048, 2048, stream=stream0)
        buf34 = empty_strided_cuda((1, 16, 1, 128), (2048, 128, 2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [convert_element_type_41, pow_6, mean_5, mul_36, cat_2, mul_37, add_14, convert_element_type_51, mul_40], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf27, arg298_1, buf4, buf34, 16, 128, stream=stream0)
        buf30 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [mul_32, sum_12, mul_35, sum_13, index_put_3], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_23.run(buf26, arg299_1, arg301_1, arg1_1, buf30, arg303_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_46, pow_7, mean_6, mul_38, cat_3, mul_39, add_15, index_put_2], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_12.run(buf30, arg300_1, arg1_1, buf4, arg302_1, 8, 128, stream=stream0)
        buf35 = reinterpret_tensor(buf17, (16, 1, 960), (960, 960, 1), 0); del buf17  # reuse
        # Topologically Sorted Source Nodes: [mul_42, sum_14], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf34, arg302_1, buf35, 15360, 128, stream=stream0)
        buf41 = empty_strided_cuda((1, 16, 1, 960), (15360, 960, 15360, 1), torch.float32)
        # Topologically Sorted Source Nodes: [full_default_4, full_default_3, where_2, add_16, eq_1, logical_not_2, any_2, logical_not_3, full_default_5, , sub, exp, div_1, where_3], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_14.run(buf35, arg1_1, buf41, 16, 960, stream=stream0)
        buf42 = empty_strided_cuda((16, 1, 128, 8), (1024, 16384, 1, 128), torch.float32)
        # Topologically Sorted Source Nodes: [mul_43, sum_16], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf41, arg303_1, buf42, 16384, 120, stream=stream0)
        buf43 = reinterpret_tensor(buf34, (16, 1, 128), (128, 128, 1), 0); del buf34  # reuse
        # Topologically Sorted Source Nodes: [mul_43, sum_16], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf42, buf43, 2048, 8, stream=stream0)
        buf45 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [embedding, add_8, add_10, mul_44, sum_17, add_17], Original ATen: [aten.embedding, aten.add, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_embedding_mul_sum_24.run(buf43, arg304_1, arg0_1, arg282_1, buf20, buf24, buf45, 2048, 2048, stream=stream0)
        del arg0_1
        buf46 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_59, pow_8, mean_7], Original ATen: [prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_mean_pow_25.run(buf45, buf46, 1, 2048, stream=stream0)
        buf47 = buf23; del buf23  # reuse
        # Topologically Sorted Source Nodes: [mul_47, sum_18], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_26.run(arg305_1, buf45, buf46, arg306_1, buf47, 12288, 2048, stream=stream0)
        buf48 = reinterpret_tensor(buf43, (1, 2048), (2048, 1), 0); del buf43  # reuse
        # Topologically Sorted Source Nodes: [mul_50, sum_19], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf47, arg307_1, buf48, 2048, 6144, stream=stream0)
        buf49 = buf46; del buf46  # reuse
        # Topologically Sorted Source Nodes: [add_19, convert_element_type_69, pow_9, mean_8], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_27.run(buf45, buf48, buf49, 1, 2048, stream=stream0)
        buf50 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [mul_53, sum_20], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_28.run(arg308_1, buf45, buf48, buf49, arg309_1, buf50, 2048, 2048, stream=stream0)
        buf57 = reinterpret_tensor(buf20, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_74, pow_10, mean_9, mul_60, cat_4, mul_61, add_23, convert_element_type_84, mul_64], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf50, arg310_1, buf4, buf57, 16, 128, stream=stream0)
        buf53 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [mul_56, sum_21, mul_59, sum_22, index_put_5], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_29.run(arg308_1, buf45, buf48, buf49, arg311_1, arg313_1, arg1_1, buf53, arg315_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_79, pow_11, mean_10, mul_62, cat_5, mul_63, add_24, index_put_4], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_12.run(buf53, arg312_1, arg1_1, buf4, arg314_1, 8, 128, stream=stream0)
        buf58 = reinterpret_tensor(buf41, (16, 1, 960), (960, 960, 1), 0); del buf41  # reuse
        # Topologically Sorted Source Nodes: [mul_66, sum_23], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf57, arg314_1, buf58, 15360, 128, stream=stream0)
        buf64 = empty_strided_cuda((1, 16, 1, 960), (15360, 960, 15360, 1), torch.float32)
        # Topologically Sorted Source Nodes: [full_default_7, full_default_6, where_4, add_25, eq_2, logical_not_4, any_3, logical_not_5, full_default_8, , sub, exp, div_2, where_5], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_14.run(buf58, arg1_1, buf64, 16, 960, stream=stream0)
        buf65 = empty_strided_cuda((16, 1, 128, 8), (1024, 16384, 1, 128), torch.float32)
        # Topologically Sorted Source Nodes: [mul_67, sum_25], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf64, arg315_1, buf65, 16384, 120, stream=stream0)
        buf66 = reinterpret_tensor(buf57, (16, 1, 128), (128, 128, 1), 0); del buf57  # reuse
        # Topologically Sorted Source Nodes: [mul_67, sum_25], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf65, buf66, 2048, 8, stream=stream0)
        buf67 = buf50; del buf50  # reuse
        # Topologically Sorted Source Nodes: [mul_68, sum_26], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_17.run(buf66, arg316_1, buf67, 2048, 2048, stream=stream0)
        buf69 = reinterpret_tensor(buf66, (1, 2048), (2048, 1), 0); del buf66  # reuse
        # Topologically Sorted Source Nodes: [add_19, add_26, convert_element_type_92, pow_12, mean_11, convert_element_type_94], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_30.run(buf45, buf48, buf67, arg317_1, buf69, 1, 2048, stream=stream0)
        buf70 = buf47; del buf47  # reuse
        # Topologically Sorted Source Nodes: [mul_71, sum_27], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_19.run(buf69, arg318_1, buf70, 12288, 2048, stream=stream0)
        buf71 = buf69; del buf69  # reuse
        # Topologically Sorted Source Nodes: [mul_74, sum_28], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf70, arg319_1, buf71, 2048, 6144, stream=stream0)
        buf73 = empty_strided_cuda((1, 1, 2048), (2048, 2048, 1), torch.float16)
        # Topologically Sorted Source Nodes: [add_19, add_26, add_28, convert_element_type_102, pow_13, mean_12, add_29, rsqrt_12, mul_75, convert_element_type_103, mul_76], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_mul_pow_rsqrt_31.run(buf45, buf48, buf67, buf71, arg320_1, buf73, 1, 2048, stream=stream0)
        buf74 = buf27; del buf27  # reuse
        # Topologically Sorted Source Nodes: [mul_77, sum_29], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_22.run(buf73, arg321_1, buf74, 2048, 2048, stream=stream0)
        buf81 = empty_strided_cuda((1, 16, 1, 128), (2048, 128, 2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [convert_element_type_107, pow_14, mean_13, mul_84, cat_6, mul_85, add_32, convert_element_type_117, mul_88], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf74, arg322_1, buf4, buf81, 16, 128, stream=stream0)
        buf77 = buf53; del buf53  # reuse
        # Topologically Sorted Source Nodes: [mul_80, sum_30, mul_83, sum_31, index_put_7], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_23.run(buf73, arg323_1, arg325_1, arg1_1, buf77, arg327_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_112, pow_15, mean_14, mul_86, cat_7, mul_87, add_33, index_put_6], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_12.run(buf77, arg324_1, arg1_1, buf4, arg326_1, 8, 128, stream=stream0)
        buf82 = reinterpret_tensor(buf64, (16, 1, 960), (960, 960, 1), 0); del buf64  # reuse
        # Topologically Sorted Source Nodes: [mul_90, sum_32], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf81, arg326_1, buf82, 15360, 128, stream=stream0)
        buf88 = empty_strided_cuda((1, 16, 1, 960), (15360, 960, 15360, 1), torch.float32)
        # Topologically Sorted Source Nodes: [full_default_10, full_default_9, where_6, add_34, eq_3, logical_not_6, any_4, logical_not_7, full_default_11, , sub, exp, div_3, where_7], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_14.run(buf82, arg1_1, buf88, 16, 960, stream=stream0)
        buf89 = empty_strided_cuda((16, 1, 128, 8), (1024, 16384, 1, 128), torch.float32)
        # Topologically Sorted Source Nodes: [mul_91, sum_34], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf88, arg327_1, buf89, 16384, 120, stream=stream0)
        buf90 = reinterpret_tensor(buf81, (16, 1, 128), (128, 128, 1), 0); del buf81  # reuse
        # Topologically Sorted Source Nodes: [mul_91, sum_34], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf89, buf90, 2048, 8, stream=stream0)
        buf92 = buf45; del buf45  # reuse
        # Topologically Sorted Source Nodes: [add_19, add_26, add_28, mul_92, sum_35, add_35], Original ATen: [aten.add, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_sum_32.run(buf92, buf90, arg328_1, buf48, buf67, buf71, 2048, 2048, stream=stream0)
        buf93 = buf49; del buf49  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_125, pow_16, mean_15], Original ATen: [prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_mean_pow_25.run(buf92, buf93, 1, 2048, stream=stream0)
        buf94 = buf70; del buf70  # reuse
        # Topologically Sorted Source Nodes: [mul_95, sum_36], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_26.run(arg329_1, buf92, buf93, arg330_1, buf94, 12288, 2048, stream=stream0)
        buf95 = reinterpret_tensor(buf90, (1, 2048), (2048, 1), 0); del buf90  # reuse
        # Topologically Sorted Source Nodes: [mul_98, sum_37], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf94, arg331_1, buf95, 2048, 6144, stream=stream0)
        buf96 = buf93; del buf93  # reuse
        # Topologically Sorted Source Nodes: [add_37, convert_element_type_135, pow_17, mean_16], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_27.run(buf92, buf95, buf96, 1, 2048, stream=stream0)
        buf97 = buf71; del buf71  # reuse
        # Topologically Sorted Source Nodes: [mul_101, sum_38], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_28.run(arg332_1, buf92, buf95, buf96, arg333_1, buf97, 2048, 2048, stream=stream0)
        buf104 = reinterpret_tensor(buf67, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf67  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_140, pow_18, mean_17, mul_108, cat_8, mul_109, add_41, convert_element_type_150, mul_112], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf97, arg334_1, buf4, buf104, 16, 128, stream=stream0)
        buf100 = buf77; del buf77  # reuse
        # Topologically Sorted Source Nodes: [mul_104, sum_39, mul_107, sum_40, index_put_9], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_29.run(arg332_1, buf92, buf95, buf96, arg335_1, arg337_1, arg1_1, buf100, arg339_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_145, pow_19, mean_18, mul_110, cat_9, mul_111, add_42, index_put_8], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_12.run(buf100, arg336_1, arg1_1, buf4, arg338_1, 8, 128, stream=stream0)
        buf105 = reinterpret_tensor(buf88, (16, 1, 960), (960, 960, 1), 0); del buf88  # reuse
        # Topologically Sorted Source Nodes: [mul_114, sum_41], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf104, arg338_1, buf105, 15360, 128, stream=stream0)
        buf111 = empty_strided_cuda((1, 16, 1, 960), (15360, 960, 15360, 1), torch.float32)
        # Topologically Sorted Source Nodes: [full_default_13, full_default_12, where_8, add_43, eq_4, logical_not_8, any_5, logical_not_9, full_default_14, , sub, exp, div_4, where_9], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_14.run(buf105, arg1_1, buf111, 16, 960, stream=stream0)
        buf112 = empty_strided_cuda((16, 1, 128, 8), (1024, 16384, 1, 128), torch.float32)
        # Topologically Sorted Source Nodes: [mul_115, sum_43], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf111, arg339_1, buf112, 16384, 120, stream=stream0)
        buf113 = reinterpret_tensor(buf104, (16, 1, 128), (128, 128, 1), 0); del buf104  # reuse
        # Topologically Sorted Source Nodes: [mul_115, sum_43], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf112, buf113, 2048, 8, stream=stream0)
        buf114 = buf97; del buf97  # reuse
        # Topologically Sorted Source Nodes: [mul_116, sum_44], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_17.run(buf113, arg340_1, buf114, 2048, 2048, stream=stream0)
        buf116 = reinterpret_tensor(buf113, (1, 2048), (2048, 1), 0); del buf113  # reuse
        # Topologically Sorted Source Nodes: [add_37, add_44, convert_element_type_158, pow_20, mean_19, convert_element_type_160], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_30.run(buf92, buf95, buf114, arg341_1, buf116, 1, 2048, stream=stream0)
        buf117 = buf94; del buf94  # reuse
        # Topologically Sorted Source Nodes: [mul_119, sum_45], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_19.run(buf116, arg2_1, buf117, 12288, 2048, stream=stream0)
        buf118 = buf116; del buf116  # reuse
        # Topologically Sorted Source Nodes: [mul_122, sum_46], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf117, arg3_1, buf118, 2048, 6144, stream=stream0)
        buf120 = buf73; del buf73  # reuse
        # Topologically Sorted Source Nodes: [add_37, add_44, add_46, convert_element_type_168, pow_21, mean_20, add_47, rsqrt_20, mul_123, convert_element_type_169, mul_124], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_mul_pow_rsqrt_31.run(buf92, buf95, buf114, buf118, arg4_1, buf120, 1, 2048, stream=stream0)
        buf121 = buf48; del buf48  # reuse
        # Topologically Sorted Source Nodes: [mul_125, sum_47], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_22.run(buf120, arg5_1, buf121, 2048, 2048, stream=stream0)
        buf128 = reinterpret_tensor(buf74, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf74  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_173, pow_22, mean_21, mul_132, cat_10, mul_133, add_50, convert_element_type_183, mul_136], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf121, arg6_1, buf4, buf128, 16, 128, stream=stream0)
        buf124 = buf100; del buf100  # reuse
        # Topologically Sorted Source Nodes: [mul_128, sum_48, mul_131, sum_49, index_put_11], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_23.run(buf120, arg7_1, arg9_1, arg1_1, buf124, arg11_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_178, pow_23, mean_22, mul_134, cat_11, mul_135, add_51, index_put_10], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_12.run(buf124, arg8_1, arg1_1, buf4, arg10_1, 8, 128, stream=stream0)
        buf129 = reinterpret_tensor(buf111, (16, 1, 960), (960, 960, 1), 0); del buf111  # reuse
        # Topologically Sorted Source Nodes: [mul_138, sum_50], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf128, arg10_1, buf129, 15360, 128, stream=stream0)
        buf135 = empty_strided_cuda((1, 16, 1, 960), (15360, 960, 15360, 1), torch.float32)
        # Topologically Sorted Source Nodes: [full_default_16, full_default_15, where_10, add_52, eq_5, logical_not_10, any_6, logical_not_11, full_default_17, , sub, exp, div_5, where_11], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_14.run(buf129, arg1_1, buf135, 16, 960, stream=stream0)
        buf136 = empty_strided_cuda((16, 1, 128, 8), (1024, 16384, 1, 128), torch.float32)
        # Topologically Sorted Source Nodes: [mul_139, sum_52], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf135, arg11_1, buf136, 16384, 120, stream=stream0)
        buf137 = reinterpret_tensor(buf128, (16, 1, 128), (128, 128, 1), 0); del buf128  # reuse
        # Topologically Sorted Source Nodes: [mul_139, sum_52], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf136, buf137, 2048, 8, stream=stream0)
        buf139 = buf92; del buf92  # reuse
        # Topologically Sorted Source Nodes: [add_37, add_44, add_46, mul_140, sum_53, add_53], Original ATen: [aten.add, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_sum_32.run(buf139, buf137, arg12_1, buf95, buf114, buf118, 2048, 2048, stream=stream0)
        buf140 = buf96; del buf96  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_191, pow_24, mean_23], Original ATen: [prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_mean_pow_25.run(buf139, buf140, 1, 2048, stream=stream0)
        buf141 = buf117; del buf117  # reuse
        # Topologically Sorted Source Nodes: [mul_143, sum_54], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_26.run(arg13_1, buf139, buf140, arg14_1, buf141, 12288, 2048, stream=stream0)
        buf142 = buf95; del buf95  # reuse
        # Topologically Sorted Source Nodes: [mul_146, sum_55], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf141, arg15_1, buf142, 2048, 6144, stream=stream0)
        buf143 = buf140; del buf140  # reuse
        # Topologically Sorted Source Nodes: [add_55, convert_element_type_201, pow_25, mean_24], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_27.run(buf139, buf142, buf143, 1, 2048, stream=stream0)
        buf144 = reinterpret_tensor(buf137, (1, 2048), (2048, 1), 0); del buf137  # reuse
        # Topologically Sorted Source Nodes: [mul_149, sum_56], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_28.run(arg16_1, buf139, buf142, buf143, arg17_1, buf144, 2048, 2048, stream=stream0)
        buf151 = reinterpret_tensor(buf118, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf118  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_206, pow_26, mean_25, mul_156, cat_12, mul_157, add_59, convert_element_type_216, mul_160], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf144, arg18_1, buf4, buf151, 16, 128, stream=stream0)
        buf147 = buf124; del buf124  # reuse
        # Topologically Sorted Source Nodes: [mul_152, sum_57, mul_155, sum_58, index_put_13], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_29.run(arg16_1, buf139, buf142, buf143, arg19_1, arg21_1, arg1_1, buf147, arg23_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_211, pow_27, mean_26, mul_158, cat_13, mul_159, add_60, index_put_12], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_12.run(buf147, arg20_1, arg1_1, buf4, arg22_1, 8, 128, stream=stream0)
        buf152 = reinterpret_tensor(buf135, (16, 1, 960), (960, 960, 1), 0); del buf135  # reuse
        # Topologically Sorted Source Nodes: [mul_162, sum_59], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf151, arg22_1, buf152, 15360, 128, stream=stream0)
        buf158 = empty_strided_cuda((1, 16, 1, 960), (15360, 960, 15360, 1), torch.float32)
        # Topologically Sorted Source Nodes: [full_default_19, full_default_18, where_12, add_61, eq_6, logical_not_12, any_7, logical_not_13, full_default_20, , sub, exp, div_6, where_13], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_14.run(buf152, arg1_1, buf158, 16, 960, stream=stream0)
        buf159 = empty_strided_cuda((16, 1, 128, 8), (1024, 16384, 1, 128), torch.float32)
        # Topologically Sorted Source Nodes: [mul_163, sum_61], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf158, arg23_1, buf159, 16384, 120, stream=stream0)
        buf160 = reinterpret_tensor(buf151, (16, 1, 128), (128, 128, 1), 0); del buf151  # reuse
        # Topologically Sorted Source Nodes: [mul_163, sum_61], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf159, buf160, 2048, 8, stream=stream0)
        buf161 = buf144; del buf144  # reuse
        # Topologically Sorted Source Nodes: [mul_164, sum_62], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_17.run(buf160, arg24_1, buf161, 2048, 2048, stream=stream0)
        buf163 = reinterpret_tensor(buf160, (1, 2048), (2048, 1), 0); del buf160  # reuse
        # Topologically Sorted Source Nodes: [add_55, add_62, convert_element_type_224, pow_28, mean_27, convert_element_type_226], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_30.run(buf139, buf142, buf161, arg25_1, buf163, 1, 2048, stream=stream0)
        buf164 = buf141; del buf141  # reuse
        # Topologically Sorted Source Nodes: [mul_167, sum_63], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_19.run(buf163, arg26_1, buf164, 12288, 2048, stream=stream0)
        buf165 = buf163; del buf163  # reuse
        # Topologically Sorted Source Nodes: [mul_170, sum_64], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf164, arg27_1, buf165, 2048, 6144, stream=stream0)
        buf167 = buf120; del buf120  # reuse
        # Topologically Sorted Source Nodes: [add_55, add_62, add_64, convert_element_type_234, pow_29, mean_28, add_65, rsqrt_28, mul_171, convert_element_type_235, mul_172], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_mul_pow_rsqrt_31.run(buf139, buf142, buf161, buf165, arg28_1, buf167, 1, 2048, stream=stream0)
        buf168 = buf114; del buf114  # reuse
        # Topologically Sorted Source Nodes: [mul_173, sum_65], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_22.run(buf167, arg29_1, buf168, 2048, 2048, stream=stream0)
        buf175 = reinterpret_tensor(buf121, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf121  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_239, pow_30, mean_29, mul_180, cat_14, mul_181, add_68, convert_element_type_249, mul_184], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf168, arg30_1, buf4, buf175, 16, 128, stream=stream0)
        buf171 = buf147; del buf147  # reuse
        # Topologically Sorted Source Nodes: [mul_176, sum_66, mul_179, sum_67, index_put_15], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_23.run(buf167, arg31_1, arg33_1, arg1_1, buf171, arg35_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_244, pow_31, mean_30, mul_182, cat_15, mul_183, add_69, index_put_14], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_12.run(buf171, arg32_1, arg1_1, buf4, arg34_1, 8, 128, stream=stream0)
        buf176 = reinterpret_tensor(buf158, (16, 1, 960), (960, 960, 1), 0); del buf158  # reuse
        # Topologically Sorted Source Nodes: [mul_186, sum_68], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf175, arg34_1, buf176, 15360, 128, stream=stream0)
        buf182 = empty_strided_cuda((1, 16, 1, 960), (15360, 960, 15360, 1), torch.float32)
        # Topologically Sorted Source Nodes: [full_default_22, full_default_21, where_14, add_70, eq_7, logical_not_14, any_8, logical_not_15, full_default_23, , sub, exp, div_7, where_15], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_14.run(buf176, arg1_1, buf182, 16, 960, stream=stream0)
        buf183 = empty_strided_cuda((16, 1, 128, 8), (1024, 16384, 1, 128), torch.float32)
        # Topologically Sorted Source Nodes: [mul_187, sum_70], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf182, arg35_1, buf183, 16384, 120, stream=stream0)
        buf184 = reinterpret_tensor(buf175, (16, 1, 128), (128, 128, 1), 0); del buf175  # reuse
        # Topologically Sorted Source Nodes: [mul_187, sum_70], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf183, buf184, 2048, 8, stream=stream0)
        buf186 = buf139; del buf139  # reuse
        # Topologically Sorted Source Nodes: [add_55, add_62, add_64, mul_188, sum_71, add_71], Original ATen: [aten.add, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_sum_32.run(buf186, buf184, arg36_1, buf142, buf161, buf165, 2048, 2048, stream=stream0)
        buf187 = buf143; del buf143  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_257, pow_32, mean_31], Original ATen: [prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_mean_pow_25.run(buf186, buf187, 1, 2048, stream=stream0)
        buf188 = buf164; del buf164  # reuse
        # Topologically Sorted Source Nodes: [mul_191, sum_72], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_26.run(arg37_1, buf186, buf187, arg38_1, buf188, 12288, 2048, stream=stream0)
        buf189 = reinterpret_tensor(buf184, (1, 2048), (2048, 1), 0); del buf184  # reuse
        # Topologically Sorted Source Nodes: [mul_194, sum_73], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf188, arg39_1, buf189, 2048, 6144, stream=stream0)
        buf190 = buf187; del buf187  # reuse
        # Topologically Sorted Source Nodes: [add_73, convert_element_type_267, pow_33, mean_32], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_27.run(buf186, buf189, buf190, 1, 2048, stream=stream0)
        buf191 = buf165; del buf165  # reuse
        # Topologically Sorted Source Nodes: [mul_197, sum_74], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_28.run(arg40_1, buf186, buf189, buf190, arg41_1, buf191, 2048, 2048, stream=stream0)
        buf198 = reinterpret_tensor(buf161, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf161  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_272, pow_34, mean_33, mul_204, cat_16, mul_205, add_77, convert_element_type_282, mul_208], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf191, arg42_1, buf4, buf198, 16, 128, stream=stream0)
        buf194 = buf171; del buf171  # reuse
        # Topologically Sorted Source Nodes: [mul_200, sum_75, mul_203, sum_76, index_put_17], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_29.run(arg40_1, buf186, buf189, buf190, arg43_1, arg45_1, arg1_1, buf194, arg47_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_277, pow_35, mean_34, mul_206, cat_17, mul_207, add_78, index_put_16], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_12.run(buf194, arg44_1, arg1_1, buf4, arg46_1, 8, 128, stream=stream0)
        buf199 = reinterpret_tensor(buf182, (16, 1, 960), (960, 960, 1), 0); del buf182  # reuse
        # Topologically Sorted Source Nodes: [mul_210, sum_77], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf198, arg46_1, buf199, 15360, 128, stream=stream0)
        buf205 = empty_strided_cuda((1, 16, 1, 960), (15360, 960, 15360, 1), torch.float32)
        # Topologically Sorted Source Nodes: [full_default_25, full_default_24, where_16, add_79, eq_8, logical_not_16, any_9, logical_not_17, full_default_26, , sub, exp, div_8, where_17], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_14.run(buf199, arg1_1, buf205, 16, 960, stream=stream0)
        buf206 = empty_strided_cuda((16, 1, 128, 8), (1024, 16384, 1, 128), torch.float32)
        # Topologically Sorted Source Nodes: [mul_211, sum_79], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf205, arg47_1, buf206, 16384, 120, stream=stream0)
        buf207 = reinterpret_tensor(buf198, (16, 1, 128), (128, 128, 1), 0); del buf198  # reuse
        # Topologically Sorted Source Nodes: [mul_211, sum_79], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf206, buf207, 2048, 8, stream=stream0)
        buf208 = buf191; del buf191  # reuse
        # Topologically Sorted Source Nodes: [mul_212, sum_80], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_17.run(buf207, arg48_1, buf208, 2048, 2048, stream=stream0)
        buf210 = reinterpret_tensor(buf207, (1, 2048), (2048, 1), 0); del buf207  # reuse
        # Topologically Sorted Source Nodes: [add_73, add_80, convert_element_type_290, pow_36, mean_35, convert_element_type_292], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_30.run(buf186, buf189, buf208, arg49_1, buf210, 1, 2048, stream=stream0)
        buf211 = buf188; del buf188  # reuse
        # Topologically Sorted Source Nodes: [mul_215, sum_81], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_19.run(buf210, arg50_1, buf211, 12288, 2048, stream=stream0)
        buf212 = buf210; del buf210  # reuse
        # Topologically Sorted Source Nodes: [mul_218, sum_82], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf211, arg51_1, buf212, 2048, 6144, stream=stream0)
        buf214 = buf167; del buf167  # reuse
        # Topologically Sorted Source Nodes: [add_73, add_80, add_82, convert_element_type_300, pow_37, mean_36, add_83, rsqrt_36, mul_219, convert_element_type_301, mul_220], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_mul_pow_rsqrt_31.run(buf186, buf189, buf208, buf212, arg52_1, buf214, 1, 2048, stream=stream0)
        buf215 = buf142; del buf142  # reuse
        # Topologically Sorted Source Nodes: [mul_221, sum_83], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_22.run(buf214, arg53_1, buf215, 2048, 2048, stream=stream0)
        buf222 = reinterpret_tensor(buf168, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf168  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_305, pow_38, mean_37, mul_228, cat_18, mul_229, add_86, convert_element_type_315, mul_232], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf215, arg54_1, buf4, buf222, 16, 128, stream=stream0)
        buf218 = buf194; del buf194  # reuse
        # Topologically Sorted Source Nodes: [mul_224, sum_84, mul_227, sum_85, index_put_19], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_23.run(buf214, arg55_1, arg57_1, arg1_1, buf218, arg59_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_310, pow_39, mean_38, mul_230, cat_19, mul_231, add_87, index_put_18], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_12.run(buf218, arg56_1, arg1_1, buf4, arg58_1, 8, 128, stream=stream0)
        buf223 = reinterpret_tensor(buf205, (16, 1, 960), (960, 960, 1), 0); del buf205  # reuse
        # Topologically Sorted Source Nodes: [mul_234, sum_86], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf222, arg58_1, buf223, 15360, 128, stream=stream0)
        buf229 = empty_strided_cuda((1, 16, 1, 960), (15360, 960, 15360, 1), torch.float32)
        # Topologically Sorted Source Nodes: [full_default_28, full_default_27, where_18, add_88, eq_9, logical_not_18, any_10, logical_not_19, full_default_29, , sub, exp, div_9, where_19], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_14.run(buf223, arg1_1, buf229, 16, 960, stream=stream0)
        buf230 = empty_strided_cuda((16, 1, 128, 8), (1024, 16384, 1, 128), torch.float32)
        # Topologically Sorted Source Nodes: [mul_235, sum_88], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf229, arg59_1, buf230, 16384, 120, stream=stream0)
        buf231 = reinterpret_tensor(buf222, (16, 1, 128), (128, 128, 1), 0); del buf222  # reuse
        # Topologically Sorted Source Nodes: [mul_235, sum_88], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf230, buf231, 2048, 8, stream=stream0)
        buf233 = buf186; del buf186  # reuse
        # Topologically Sorted Source Nodes: [add_73, add_80, add_82, mul_236, sum_89, add_89], Original ATen: [aten.add, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_sum_32.run(buf233, buf231, arg60_1, buf189, buf208, buf212, 2048, 2048, stream=stream0)
        buf234 = buf190; del buf190  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_323, pow_40, mean_39], Original ATen: [prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_mean_pow_25.run(buf233, buf234, 1, 2048, stream=stream0)
        buf235 = buf211; del buf211  # reuse
        # Topologically Sorted Source Nodes: [mul_239, sum_90], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_26.run(arg61_1, buf233, buf234, arg62_1, buf235, 12288, 2048, stream=stream0)
        buf236 = reinterpret_tensor(buf231, (1, 2048), (2048, 1), 0); del buf231  # reuse
        # Topologically Sorted Source Nodes: [mul_242, sum_91], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf235, arg63_1, buf236, 2048, 6144, stream=stream0)
        buf237 = buf234; del buf234  # reuse
        # Topologically Sorted Source Nodes: [add_91, convert_element_type_333, pow_41, mean_40], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_27.run(buf233, buf236, buf237, 1, 2048, stream=stream0)
        buf238 = buf212; del buf212  # reuse
        # Topologically Sorted Source Nodes: [mul_245, sum_92], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_28.run(arg64_1, buf233, buf236, buf237, arg65_1, buf238, 2048, 2048, stream=stream0)
        buf245 = reinterpret_tensor(buf208, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf208  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_338, pow_42, mean_41, mul_252, cat_20, mul_253, add_95, convert_element_type_348, mul_256], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf238, arg66_1, buf4, buf245, 16, 128, stream=stream0)
        buf241 = buf218; del buf218  # reuse
        # Topologically Sorted Source Nodes: [mul_248, sum_93, mul_251, sum_94, index_put_21], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_29.run(arg64_1, buf233, buf236, buf237, arg67_1, arg69_1, arg1_1, buf241, arg71_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_343, pow_43, mean_42, mul_254, cat_21, mul_255, add_96, index_put_20], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_12.run(buf241, arg68_1, arg1_1, buf4, arg70_1, 8, 128, stream=stream0)
        buf246 = reinterpret_tensor(buf229, (16, 1, 960), (960, 960, 1), 0); del buf229  # reuse
        # Topologically Sorted Source Nodes: [mul_258, sum_95], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf245, arg70_1, buf246, 15360, 128, stream=stream0)
        buf252 = empty_strided_cuda((1, 16, 1, 960), (15360, 960, 15360, 1), torch.float32)
        # Topologically Sorted Source Nodes: [full_default_31, full_default_30, where_20, add_97, eq_10, logical_not_20, any_11, logical_not_21, full_default_32, , sub, exp, div_10, where_21], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_14.run(buf246, arg1_1, buf252, 16, 960, stream=stream0)
        buf253 = empty_strided_cuda((16, 1, 128, 8), (1024, 16384, 1, 128), torch.float32)
        # Topologically Sorted Source Nodes: [mul_259, sum_97], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf252, arg71_1, buf253, 16384, 120, stream=stream0)
        buf254 = reinterpret_tensor(buf245, (16, 1, 128), (128, 128, 1), 0); del buf245  # reuse
        # Topologically Sorted Source Nodes: [mul_259, sum_97], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf253, buf254, 2048, 8, stream=stream0)
        buf255 = buf238; del buf238  # reuse
        # Topologically Sorted Source Nodes: [mul_260, sum_98], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_17.run(buf254, arg72_1, buf255, 2048, 2048, stream=stream0)
        buf257 = reinterpret_tensor(buf254, (1, 2048), (2048, 1), 0); del buf254  # reuse
        # Topologically Sorted Source Nodes: [add_91, add_98, convert_element_type_356, pow_44, mean_43, convert_element_type_358], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_30.run(buf233, buf236, buf255, arg73_1, buf257, 1, 2048, stream=stream0)
        buf258 = buf235; del buf235  # reuse
        # Topologically Sorted Source Nodes: [mul_263, sum_99], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_19.run(buf257, arg74_1, buf258, 12288, 2048, stream=stream0)
        buf259 = buf257; del buf257  # reuse
        # Topologically Sorted Source Nodes: [mul_266, sum_100], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf258, arg75_1, buf259, 2048, 6144, stream=stream0)
        buf261 = buf214; del buf214  # reuse
        # Topologically Sorted Source Nodes: [add_91, add_98, add_100, convert_element_type_366, pow_45, mean_44, add_101, rsqrt_44, mul_267, convert_element_type_367, mul_268], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_mul_pow_rsqrt_31.run(buf233, buf236, buf255, buf259, arg76_1, buf261, 1, 2048, stream=stream0)
        buf262 = buf189; del buf189  # reuse
        # Topologically Sorted Source Nodes: [mul_269, sum_101], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_22.run(buf261, arg77_1, buf262, 2048, 2048, stream=stream0)
        buf269 = reinterpret_tensor(buf215, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf215  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_371, pow_46, mean_45, mul_276, cat_22, mul_277, add_104, convert_element_type_381, mul_280], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf262, arg78_1, buf4, buf269, 16, 128, stream=stream0)
        buf265 = buf241; del buf241  # reuse
        # Topologically Sorted Source Nodes: [mul_272, sum_102, mul_275, sum_103, index_put_23], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_23.run(buf261, arg79_1, arg81_1, arg1_1, buf265, arg83_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_376, pow_47, mean_46, mul_278, cat_23, mul_279, add_105, index_put_22], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_12.run(buf265, arg80_1, arg1_1, buf4, arg82_1, 8, 128, stream=stream0)
        buf270 = reinterpret_tensor(buf252, (16, 1, 960), (960, 960, 1), 0); del buf252  # reuse
        # Topologically Sorted Source Nodes: [mul_282, sum_104], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf269, arg82_1, buf270, 15360, 128, stream=stream0)
        buf276 = empty_strided_cuda((1, 16, 1, 960), (15360, 960, 15360, 1), torch.float32)
        # Topologically Sorted Source Nodes: [full_default_34, full_default_33, where_22, add_106, eq_11, logical_not_22, any_12, logical_not_23, full_default_35, , sub, exp, div_11, where_23], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_14.run(buf270, arg1_1, buf276, 16, 960, stream=stream0)
        buf277 = empty_strided_cuda((16, 1, 128, 8), (1024, 16384, 1, 128), torch.float32)
        # Topologically Sorted Source Nodes: [mul_283, sum_106], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf276, arg83_1, buf277, 16384, 120, stream=stream0)
        buf278 = reinterpret_tensor(buf269, (16, 1, 128), (128, 128, 1), 0); del buf269  # reuse
        # Topologically Sorted Source Nodes: [mul_283, sum_106], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf277, buf278, 2048, 8, stream=stream0)
        buf280 = buf233; del buf233  # reuse
        # Topologically Sorted Source Nodes: [add_91, add_98, add_100, mul_284, sum_107, add_107], Original ATen: [aten.add, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_sum_32.run(buf280, buf278, arg84_1, buf236, buf255, buf259, 2048, 2048, stream=stream0)
        buf281 = buf237; del buf237  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_389, pow_48, mean_47], Original ATen: [prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_mean_pow_25.run(buf280, buf281, 1, 2048, stream=stream0)
        buf282 = buf258; del buf258  # reuse
        # Topologically Sorted Source Nodes: [mul_287, sum_108], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_26.run(arg85_1, buf280, buf281, arg86_1, buf282, 12288, 2048, stream=stream0)
        buf283 = reinterpret_tensor(buf278, (1, 2048), (2048, 1), 0); del buf278  # reuse
        # Topologically Sorted Source Nodes: [mul_290, sum_109], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf282, arg87_1, buf283, 2048, 6144, stream=stream0)
        buf284 = buf281; del buf281  # reuse
        # Topologically Sorted Source Nodes: [add_109, convert_element_type_399, pow_49, mean_48], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_27.run(buf280, buf283, buf284, 1, 2048, stream=stream0)
        buf285 = buf259; del buf259  # reuse
        # Topologically Sorted Source Nodes: [mul_293, sum_110], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_28.run(arg88_1, buf280, buf283, buf284, arg89_1, buf285, 2048, 2048, stream=stream0)
        buf292 = reinterpret_tensor(buf255, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf255  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_404, pow_50, mean_49, mul_300, cat_24, mul_301, add_113, convert_element_type_414, mul_304], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf285, arg90_1, buf4, buf292, 16, 128, stream=stream0)
        buf288 = buf265; del buf265  # reuse
        # Topologically Sorted Source Nodes: [mul_296, sum_111, mul_299, sum_112, index_put_25], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_29.run(arg88_1, buf280, buf283, buf284, arg91_1, arg93_1, arg1_1, buf288, arg95_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_409, pow_51, mean_50, mul_302, cat_25, mul_303, add_114, index_put_24], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_12.run(buf288, arg92_1, arg1_1, buf4, arg94_1, 8, 128, stream=stream0)
        buf293 = reinterpret_tensor(buf276, (16, 1, 960), (960, 960, 1), 0); del buf276  # reuse
        # Topologically Sorted Source Nodes: [mul_306, sum_113], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf292, arg94_1, buf293, 15360, 128, stream=stream0)
        buf299 = empty_strided_cuda((1, 16, 1, 960), (15360, 960, 15360, 1), torch.float32)
        # Topologically Sorted Source Nodes: [full_default_37, full_default_36, where_24, add_115, eq_12, logical_not_24, any_13, logical_not_25, full_default_38, , sub, exp, div_12, where_25], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_14.run(buf293, arg1_1, buf299, 16, 960, stream=stream0)
        buf300 = empty_strided_cuda((16, 1, 128, 8), (1024, 16384, 1, 128), torch.float32)
        # Topologically Sorted Source Nodes: [mul_307, sum_115], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf299, arg95_1, buf300, 16384, 120, stream=stream0)
        buf301 = reinterpret_tensor(buf292, (16, 1, 128), (128, 128, 1), 0); del buf292  # reuse
        # Topologically Sorted Source Nodes: [mul_307, sum_115], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf300, buf301, 2048, 8, stream=stream0)
        buf302 = buf285; del buf285  # reuse
        # Topologically Sorted Source Nodes: [mul_308, sum_116], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_17.run(buf301, arg96_1, buf302, 2048, 2048, stream=stream0)
        buf304 = reinterpret_tensor(buf301, (1, 2048), (2048, 1), 0); del buf301  # reuse
        # Topologically Sorted Source Nodes: [add_109, add_116, convert_element_type_422, pow_52, mean_51, convert_element_type_424], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_30.run(buf280, buf283, buf302, arg97_1, buf304, 1, 2048, stream=stream0)
        buf305 = buf282; del buf282  # reuse
        # Topologically Sorted Source Nodes: [mul_311, sum_117], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_19.run(buf304, arg98_1, buf305, 12288, 2048, stream=stream0)
        buf306 = buf304; del buf304  # reuse
        # Topologically Sorted Source Nodes: [mul_314, sum_118], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf305, arg99_1, buf306, 2048, 6144, stream=stream0)
        buf308 = buf261; del buf261  # reuse
        # Topologically Sorted Source Nodes: [add_109, add_116, add_118, convert_element_type_432, pow_53, mean_52, add_119, rsqrt_52, mul_315, convert_element_type_433, mul_316], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_mul_pow_rsqrt_31.run(buf280, buf283, buf302, buf306, arg100_1, buf308, 1, 2048, stream=stream0)
        buf309 = buf236; del buf236  # reuse
        # Topologically Sorted Source Nodes: [mul_317, sum_119], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_22.run(buf308, arg101_1, buf309, 2048, 2048, stream=stream0)
        buf316 = reinterpret_tensor(buf262, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf262  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_437, pow_54, mean_53, mul_324, cat_26, mul_325, add_122, convert_element_type_447, mul_328], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf309, arg102_1, buf4, buf316, 16, 128, stream=stream0)
        buf312 = buf288; del buf288  # reuse
        # Topologically Sorted Source Nodes: [mul_320, sum_120, mul_323, sum_121, index_put_27], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_23.run(buf308, arg103_1, arg105_1, arg1_1, buf312, arg107_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_442, pow_55, mean_54, mul_326, cat_27, mul_327, add_123, index_put_26], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_12.run(buf312, arg104_1, arg1_1, buf4, arg106_1, 8, 128, stream=stream0)
        buf317 = reinterpret_tensor(buf299, (16, 1, 960), (960, 960, 1), 0); del buf299  # reuse
        # Topologically Sorted Source Nodes: [mul_330, sum_122], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf316, arg106_1, buf317, 15360, 128, stream=stream0)
        buf323 = empty_strided_cuda((1, 16, 1, 960), (15360, 960, 15360, 1), torch.float32)
        # Topologically Sorted Source Nodes: [full_default_40, full_default_39, where_26, add_124, eq_13, logical_not_26, any_14, logical_not_27, full_default_41, , sub, exp, div_13, where_27], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_14.run(buf317, arg1_1, buf323, 16, 960, stream=stream0)
        buf324 = empty_strided_cuda((16, 1, 128, 8), (1024, 16384, 1, 128), torch.float32)
        # Topologically Sorted Source Nodes: [mul_331, sum_124], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf323, arg107_1, buf324, 16384, 120, stream=stream0)
        buf325 = reinterpret_tensor(buf316, (16, 1, 128), (128, 128, 1), 0); del buf316  # reuse
        # Topologically Sorted Source Nodes: [mul_331, sum_124], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf324, buf325, 2048, 8, stream=stream0)
        buf327 = buf280; del buf280  # reuse
        # Topologically Sorted Source Nodes: [add_109, add_116, add_118, mul_332, sum_125, add_125], Original ATen: [aten.add, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_sum_32.run(buf327, buf325, arg108_1, buf283, buf302, buf306, 2048, 2048, stream=stream0)
        buf328 = buf284; del buf284  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_455, pow_56, mean_55], Original ATen: [prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_mean_pow_25.run(buf327, buf328, 1, 2048, stream=stream0)
        buf329 = buf305; del buf305  # reuse
        # Topologically Sorted Source Nodes: [mul_335, sum_126], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_26.run(arg109_1, buf327, buf328, arg110_1, buf329, 12288, 2048, stream=stream0)
        buf330 = reinterpret_tensor(buf325, (1, 2048), (2048, 1), 0); del buf325  # reuse
        # Topologically Sorted Source Nodes: [mul_338, sum_127], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf329, arg111_1, buf330, 2048, 6144, stream=stream0)
        buf331 = buf328; del buf328  # reuse
        # Topologically Sorted Source Nodes: [add_127, convert_element_type_465, pow_57, mean_56], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_27.run(buf327, buf330, buf331, 1, 2048, stream=stream0)
        buf332 = buf306; del buf306  # reuse
        # Topologically Sorted Source Nodes: [mul_341, sum_128], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_28.run(arg112_1, buf327, buf330, buf331, arg113_1, buf332, 2048, 2048, stream=stream0)
        buf339 = reinterpret_tensor(buf302, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf302  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_470, pow_58, mean_57, mul_348, cat_28, mul_349, add_131, convert_element_type_480, mul_352], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf332, arg114_1, buf4, buf339, 16, 128, stream=stream0)
        buf335 = buf312; del buf312  # reuse
        # Topologically Sorted Source Nodes: [mul_344, sum_129, mul_347, sum_130, index_put_29], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_29.run(arg112_1, buf327, buf330, buf331, arg115_1, arg117_1, arg1_1, buf335, arg119_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_475, pow_59, mean_58, mul_350, cat_29, mul_351, add_132, index_put_28], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_12.run(buf335, arg116_1, arg1_1, buf4, arg118_1, 8, 128, stream=stream0)
        buf340 = reinterpret_tensor(buf323, (16, 1, 960), (960, 960, 1), 0); del buf323  # reuse
        # Topologically Sorted Source Nodes: [mul_354, sum_131], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf339, arg118_1, buf340, 15360, 128, stream=stream0)
        buf346 = empty_strided_cuda((1, 16, 1, 960), (15360, 960, 15360, 1), torch.float32)
        # Topologically Sorted Source Nodes: [full_default_43, full_default_42, where_28, add_133, eq_14, logical_not_28, any_15, logical_not_29, full_default_44, , sub, exp, div_14, where_29], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_14.run(buf340, arg1_1, buf346, 16, 960, stream=stream0)
        buf347 = empty_strided_cuda((16, 1, 128, 8), (1024, 16384, 1, 128), torch.float32)
        # Topologically Sorted Source Nodes: [mul_355, sum_133], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf346, arg119_1, buf347, 16384, 120, stream=stream0)
        buf348 = reinterpret_tensor(buf339, (16, 1, 128), (128, 128, 1), 0); del buf339  # reuse
        # Topologically Sorted Source Nodes: [mul_355, sum_133], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf347, buf348, 2048, 8, stream=stream0)
        buf349 = buf332; del buf332  # reuse
        # Topologically Sorted Source Nodes: [mul_356, sum_134], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_17.run(buf348, arg120_1, buf349, 2048, 2048, stream=stream0)
        buf351 = reinterpret_tensor(buf348, (1, 2048), (2048, 1), 0); del buf348  # reuse
        # Topologically Sorted Source Nodes: [add_127, add_134, convert_element_type_488, pow_60, mean_59, convert_element_type_490], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_30.run(buf327, buf330, buf349, arg121_1, buf351, 1, 2048, stream=stream0)
        buf352 = buf329; del buf329  # reuse
        # Topologically Sorted Source Nodes: [mul_359, sum_135], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_19.run(buf351, arg122_1, buf352, 12288, 2048, stream=stream0)
        buf353 = buf351; del buf351  # reuse
        # Topologically Sorted Source Nodes: [mul_362, sum_136], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf352, arg123_1, buf353, 2048, 6144, stream=stream0)
        buf355 = buf308; del buf308  # reuse
        # Topologically Sorted Source Nodes: [add_127, add_134, add_136, convert_element_type_498, pow_61, mean_60, add_137, rsqrt_60, mul_363, convert_element_type_499, mul_364], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_mul_pow_rsqrt_31.run(buf327, buf330, buf349, buf353, arg124_1, buf355, 1, 2048, stream=stream0)
        buf356 = buf283; del buf283  # reuse
        # Topologically Sorted Source Nodes: [mul_365, sum_137], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_22.run(buf355, arg125_1, buf356, 2048, 2048, stream=stream0)
        buf363 = reinterpret_tensor(buf309, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf309  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_503, pow_62, mean_61, mul_372, cat_30, mul_373, add_140, convert_element_type_513, mul_376], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf356, arg126_1, buf4, buf363, 16, 128, stream=stream0)
        buf359 = buf335; del buf335  # reuse
        # Topologically Sorted Source Nodes: [mul_368, sum_138, mul_371, sum_139, index_put_31], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_23.run(buf355, arg127_1, arg129_1, arg1_1, buf359, arg131_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_508, pow_63, mean_62, mul_374, cat_31, mul_375, add_141, index_put_30], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_12.run(buf359, arg128_1, arg1_1, buf4, arg130_1, 8, 128, stream=stream0)
        buf364 = reinterpret_tensor(buf346, (16, 1, 960), (960, 960, 1), 0); del buf346  # reuse
        # Topologically Sorted Source Nodes: [mul_378, sum_140], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf363, arg130_1, buf364, 15360, 128, stream=stream0)
        buf370 = empty_strided_cuda((1, 16, 1, 960), (15360, 960, 15360, 1), torch.float32)
        # Topologically Sorted Source Nodes: [full_default_46, full_default_45, where_30, add_142, eq_15, logical_not_30, any_16, logical_not_31, full_default_47, , sub, exp, div_15, where_31], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_14.run(buf364, arg1_1, buf370, 16, 960, stream=stream0)
        buf371 = empty_strided_cuda((16, 1, 128, 8), (1024, 16384, 1, 128), torch.float32)
        # Topologically Sorted Source Nodes: [mul_379, sum_142], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf370, arg131_1, buf371, 16384, 120, stream=stream0)
        buf372 = reinterpret_tensor(buf363, (16, 1, 128), (128, 128, 1), 0); del buf363  # reuse
        # Topologically Sorted Source Nodes: [mul_379, sum_142], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf371, buf372, 2048, 8, stream=stream0)
        buf374 = buf327; del buf327  # reuse
        # Topologically Sorted Source Nodes: [add_127, add_134, add_136, mul_380, sum_143, add_143], Original ATen: [aten.add, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_sum_32.run(buf374, buf372, arg132_1, buf330, buf349, buf353, 2048, 2048, stream=stream0)
        buf375 = buf331; del buf331  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_521, pow_64, mean_63], Original ATen: [prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_mean_pow_25.run(buf374, buf375, 1, 2048, stream=stream0)
        buf376 = buf352; del buf352  # reuse
        # Topologically Sorted Source Nodes: [mul_383, sum_144], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_26.run(arg133_1, buf374, buf375, arg134_1, buf376, 12288, 2048, stream=stream0)
        buf377 = reinterpret_tensor(buf372, (1, 2048), (2048, 1), 0); del buf372  # reuse
        # Topologically Sorted Source Nodes: [mul_386, sum_145], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf376, arg135_1, buf377, 2048, 6144, stream=stream0)
        buf378 = buf375; del buf375  # reuse
        # Topologically Sorted Source Nodes: [add_145, convert_element_type_531, pow_65, mean_64], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_27.run(buf374, buf377, buf378, 1, 2048, stream=stream0)
        buf379 = buf353; del buf353  # reuse
        # Topologically Sorted Source Nodes: [mul_389, sum_146], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_28.run(arg136_1, buf374, buf377, buf378, arg137_1, buf379, 2048, 2048, stream=stream0)
        buf386 = reinterpret_tensor(buf349, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf349  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_536, pow_66, mean_65, mul_396, cat_32, mul_397, add_149, convert_element_type_546, mul_400], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf379, arg138_1, buf4, buf386, 16, 128, stream=stream0)
        buf382 = buf359; del buf359  # reuse
        # Topologically Sorted Source Nodes: [mul_392, sum_147, mul_395, sum_148, index_put_33], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_29.run(arg136_1, buf374, buf377, buf378, arg139_1, arg141_1, arg1_1, buf382, arg143_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_541, pow_67, mean_66, mul_398, cat_33, mul_399, add_150, index_put_32], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_12.run(buf382, arg140_1, arg1_1, buf4, arg142_1, 8, 128, stream=stream0)
        buf387 = reinterpret_tensor(buf370, (16, 1, 960), (960, 960, 1), 0); del buf370  # reuse
        # Topologically Sorted Source Nodes: [mul_402, sum_149], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf386, arg142_1, buf387, 15360, 128, stream=stream0)
        buf393 = empty_strided_cuda((1, 16, 1, 960), (15360, 960, 15360, 1), torch.float32)
        # Topologically Sorted Source Nodes: [full_default_49, full_default_48, where_32, add_151, eq_16, logical_not_32, any_17, logical_not_33, full_default_50, , sub, exp, div_16, where_33], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_14.run(buf387, arg1_1, buf393, 16, 960, stream=stream0)
        buf394 = empty_strided_cuda((16, 1, 128, 8), (1024, 16384, 1, 128), torch.float32)
        # Topologically Sorted Source Nodes: [mul_403, sum_151], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf393, arg143_1, buf394, 16384, 120, stream=stream0)
        buf395 = reinterpret_tensor(buf386, (16, 1, 128), (128, 128, 1), 0); del buf386  # reuse
        # Topologically Sorted Source Nodes: [mul_403, sum_151], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf394, buf395, 2048, 8, stream=stream0)
        buf396 = buf379; del buf379  # reuse
        # Topologically Sorted Source Nodes: [mul_404, sum_152], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_17.run(buf395, arg144_1, buf396, 2048, 2048, stream=stream0)
        buf398 = reinterpret_tensor(buf395, (1, 2048), (2048, 1), 0); del buf395  # reuse
        # Topologically Sorted Source Nodes: [add_145, add_152, convert_element_type_554, pow_68, mean_67, convert_element_type_556], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_30.run(buf374, buf377, buf396, arg145_1, buf398, 1, 2048, stream=stream0)
        buf399 = buf376; del buf376  # reuse
        # Topologically Sorted Source Nodes: [mul_407, sum_153], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_19.run(buf398, arg146_1, buf399, 12288, 2048, stream=stream0)
        buf400 = buf398; del buf398  # reuse
        # Topologically Sorted Source Nodes: [mul_410, sum_154], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf399, arg147_1, buf400, 2048, 6144, stream=stream0)
        buf402 = buf355; del buf355  # reuse
        # Topologically Sorted Source Nodes: [add_145, add_152, add_154, convert_element_type_564, pow_69, mean_68, add_155, rsqrt_68, mul_411, convert_element_type_565, mul_412], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_mul_pow_rsqrt_31.run(buf374, buf377, buf396, buf400, arg148_1, buf402, 1, 2048, stream=stream0)
        buf403 = buf330; del buf330  # reuse
        # Topologically Sorted Source Nodes: [mul_413, sum_155], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_22.run(buf402, arg149_1, buf403, 2048, 2048, stream=stream0)
        buf410 = reinterpret_tensor(buf356, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf356  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_569, pow_70, mean_69, mul_420, cat_34, mul_421, add_158, convert_element_type_579, mul_424], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf403, arg150_1, buf4, buf410, 16, 128, stream=stream0)
        buf406 = buf382; del buf382  # reuse
        # Topologically Sorted Source Nodes: [mul_416, sum_156, mul_419, sum_157, index_put_35], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_23.run(buf402, arg151_1, arg153_1, arg1_1, buf406, arg155_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_574, pow_71, mean_70, mul_422, cat_35, mul_423, add_159, index_put_34], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_12.run(buf406, arg152_1, arg1_1, buf4, arg154_1, 8, 128, stream=stream0)
        buf411 = reinterpret_tensor(buf393, (16, 1, 960), (960, 960, 1), 0); del buf393  # reuse
        # Topologically Sorted Source Nodes: [mul_426, sum_158], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf410, arg154_1, buf411, 15360, 128, stream=stream0)
        buf417 = empty_strided_cuda((1, 16, 1, 960), (15360, 960, 15360, 1), torch.float32)
        # Topologically Sorted Source Nodes: [full_default_52, full_default_51, where_34, add_160, eq_17, logical_not_34, any_18, logical_not_35, full_default_53, , sub, exp, div_17, where_35], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_14.run(buf411, arg1_1, buf417, 16, 960, stream=stream0)
        buf418 = empty_strided_cuda((16, 1, 128, 8), (1024, 16384, 1, 128), torch.float32)
        # Topologically Sorted Source Nodes: [mul_427, sum_160], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf417, arg155_1, buf418, 16384, 120, stream=stream0)
        buf419 = reinterpret_tensor(buf410, (16, 1, 128), (128, 128, 1), 0); del buf410  # reuse
        # Topologically Sorted Source Nodes: [mul_427, sum_160], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf418, buf419, 2048, 8, stream=stream0)
        buf421 = buf374; del buf374  # reuse
        # Topologically Sorted Source Nodes: [add_145, add_152, add_154, mul_428, sum_161, add_161], Original ATen: [aten.add, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_sum_32.run(buf421, buf419, arg156_1, buf377, buf396, buf400, 2048, 2048, stream=stream0)
        buf422 = buf378; del buf378  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_587, pow_72, mean_71], Original ATen: [prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_mean_pow_25.run(buf421, buf422, 1, 2048, stream=stream0)
        buf423 = buf399; del buf399  # reuse
        # Topologically Sorted Source Nodes: [mul_431, sum_162], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_26.run(arg157_1, buf421, buf422, arg158_1, buf423, 12288, 2048, stream=stream0)
        buf424 = reinterpret_tensor(buf419, (1, 2048), (2048, 1), 0); del buf419  # reuse
        # Topologically Sorted Source Nodes: [mul_434, sum_163], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf423, arg159_1, buf424, 2048, 6144, stream=stream0)
        buf425 = buf422; del buf422  # reuse
        # Topologically Sorted Source Nodes: [add_163, convert_element_type_597, pow_73, mean_72], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_27.run(buf421, buf424, buf425, 1, 2048, stream=stream0)
        buf426 = buf400; del buf400  # reuse
        # Topologically Sorted Source Nodes: [mul_437, sum_164], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_28.run(arg160_1, buf421, buf424, buf425, arg161_1, buf426, 2048, 2048, stream=stream0)
        buf433 = reinterpret_tensor(buf396, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf396  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_602, pow_74, mean_73, mul_444, cat_36, mul_445, add_167, convert_element_type_612, mul_448], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf426, arg162_1, buf4, buf433, 16, 128, stream=stream0)
        buf429 = buf406; del buf406  # reuse
        # Topologically Sorted Source Nodes: [mul_440, sum_165, mul_443, sum_166, index_put_37], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_29.run(arg160_1, buf421, buf424, buf425, arg163_1, arg165_1, arg1_1, buf429, arg167_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_607, pow_75, mean_74, mul_446, cat_37, mul_447, add_168, index_put_36], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_12.run(buf429, arg164_1, arg1_1, buf4, arg166_1, 8, 128, stream=stream0)
        buf434 = reinterpret_tensor(buf417, (16, 1, 960), (960, 960, 1), 0); del buf417  # reuse
        # Topologically Sorted Source Nodes: [mul_450, sum_167], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf433, arg166_1, buf434, 15360, 128, stream=stream0)
        buf440 = empty_strided_cuda((1, 16, 1, 960), (15360, 960, 15360, 1), torch.float32)
        # Topologically Sorted Source Nodes: [full_default_55, full_default_54, where_36, add_169, eq_18, logical_not_36, any_19, logical_not_37, full_default_56, , sub, exp, div_18, where_37], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_14.run(buf434, arg1_1, buf440, 16, 960, stream=stream0)
        buf441 = empty_strided_cuda((16, 1, 128, 8), (1024, 16384, 1, 128), torch.float32)
        # Topologically Sorted Source Nodes: [mul_451, sum_169], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf440, arg167_1, buf441, 16384, 120, stream=stream0)
        buf442 = reinterpret_tensor(buf433, (16, 1, 128), (128, 128, 1), 0); del buf433  # reuse
        # Topologically Sorted Source Nodes: [mul_451, sum_169], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf441, buf442, 2048, 8, stream=stream0)
        buf443 = buf426; del buf426  # reuse
        # Topologically Sorted Source Nodes: [mul_452, sum_170], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_17.run(buf442, arg168_1, buf443, 2048, 2048, stream=stream0)
        buf445 = reinterpret_tensor(buf442, (1, 2048), (2048, 1), 0); del buf442  # reuse
        # Topologically Sorted Source Nodes: [add_163, add_170, convert_element_type_620, pow_76, mean_75, convert_element_type_622], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_30.run(buf421, buf424, buf443, arg169_1, buf445, 1, 2048, stream=stream0)
        buf446 = buf423; del buf423  # reuse
        # Topologically Sorted Source Nodes: [mul_455, sum_171], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_19.run(buf445, arg170_1, buf446, 12288, 2048, stream=stream0)
        buf447 = buf445; del buf445  # reuse
        # Topologically Sorted Source Nodes: [mul_458, sum_172], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf446, arg171_1, buf447, 2048, 6144, stream=stream0)
        buf449 = buf402; del buf402  # reuse
        # Topologically Sorted Source Nodes: [add_163, add_170, add_172, convert_element_type_630, pow_77, mean_76, add_173, rsqrt_76, mul_459, convert_element_type_631, mul_460], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_mul_pow_rsqrt_31.run(buf421, buf424, buf443, buf447, arg172_1, buf449, 1, 2048, stream=stream0)
        buf450 = buf377; del buf377  # reuse
        # Topologically Sorted Source Nodes: [mul_461, sum_173], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_22.run(buf449, arg173_1, buf450, 2048, 2048, stream=stream0)
        buf457 = reinterpret_tensor(buf403, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf403  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_635, pow_78, mean_77, mul_468, cat_38, mul_469, add_176, convert_element_type_645, mul_472], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf450, arg174_1, buf4, buf457, 16, 128, stream=stream0)
        buf453 = buf429; del buf429  # reuse
        # Topologically Sorted Source Nodes: [mul_464, sum_174, mul_467, sum_175, index_put_39], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_23.run(buf449, arg175_1, arg177_1, arg1_1, buf453, arg179_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_640, pow_79, mean_78, mul_470, cat_39, mul_471, add_177, index_put_38], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_12.run(buf453, arg176_1, arg1_1, buf4, arg178_1, 8, 128, stream=stream0)
        buf458 = reinterpret_tensor(buf440, (16, 1, 960), (960, 960, 1), 0); del buf440  # reuse
        # Topologically Sorted Source Nodes: [mul_474, sum_176], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf457, arg178_1, buf458, 15360, 128, stream=stream0)
        buf464 = empty_strided_cuda((1, 16, 1, 960), (15360, 960, 15360, 1), torch.float32)
        # Topologically Sorted Source Nodes: [full_default_58, full_default_57, where_38, add_178, eq_19, logical_not_38, any_20, logical_not_39, full_default_59, , sub, exp, div_19, where_39], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_14.run(buf458, arg1_1, buf464, 16, 960, stream=stream0)
        buf465 = empty_strided_cuda((16, 1, 128, 8), (1024, 16384, 1, 128), torch.float32)
        # Topologically Sorted Source Nodes: [mul_475, sum_178], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf464, arg179_1, buf465, 16384, 120, stream=stream0)
        buf466 = reinterpret_tensor(buf457, (16, 1, 128), (128, 128, 1), 0); del buf457  # reuse
        # Topologically Sorted Source Nodes: [mul_475, sum_178], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf465, buf466, 2048, 8, stream=stream0)
        buf468 = buf421; del buf421  # reuse
        # Topologically Sorted Source Nodes: [add_163, add_170, add_172, mul_476, sum_179, add_179], Original ATen: [aten.add, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_sum_32.run(buf468, buf466, arg180_1, buf424, buf443, buf447, 2048, 2048, stream=stream0)
        buf469 = buf425; del buf425  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_653, pow_80, mean_79], Original ATen: [prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_mean_pow_25.run(buf468, buf469, 1, 2048, stream=stream0)
        buf470 = buf446; del buf446  # reuse
        # Topologically Sorted Source Nodes: [mul_479, sum_180], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_26.run(arg181_1, buf468, buf469, arg182_1, buf470, 12288, 2048, stream=stream0)
        buf471 = reinterpret_tensor(buf466, (1, 2048), (2048, 1), 0); del buf466  # reuse
        # Topologically Sorted Source Nodes: [mul_482, sum_181], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf470, arg183_1, buf471, 2048, 6144, stream=stream0)
        buf472 = buf469; del buf469  # reuse
        # Topologically Sorted Source Nodes: [add_181, convert_element_type_663, pow_81, mean_80], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_27.run(buf468, buf471, buf472, 1, 2048, stream=stream0)
        buf473 = buf447; del buf447  # reuse
        # Topologically Sorted Source Nodes: [mul_485, sum_182], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_28.run(arg184_1, buf468, buf471, buf472, arg185_1, buf473, 2048, 2048, stream=stream0)
        buf480 = reinterpret_tensor(buf443, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf443  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_668, pow_82, mean_81, mul_492, cat_40, mul_493, add_185, convert_element_type_678, mul_496], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf473, arg186_1, buf4, buf480, 16, 128, stream=stream0)
        buf476 = buf453; del buf453  # reuse
        # Topologically Sorted Source Nodes: [mul_488, sum_183, mul_491, sum_184, index_put_41], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_29.run(arg184_1, buf468, buf471, buf472, arg187_1, arg189_1, arg1_1, buf476, arg191_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_673, pow_83, mean_82, mul_494, cat_41, mul_495, add_186, index_put_40], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_12.run(buf476, arg188_1, arg1_1, buf4, arg190_1, 8, 128, stream=stream0)
        buf481 = reinterpret_tensor(buf464, (16, 1, 960), (960, 960, 1), 0); del buf464  # reuse
        # Topologically Sorted Source Nodes: [mul_498, sum_185], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf480, arg190_1, buf481, 15360, 128, stream=stream0)
        buf487 = empty_strided_cuda((1, 16, 1, 960), (15360, 960, 15360, 1), torch.float32)
        # Topologically Sorted Source Nodes: [full_default_61, full_default_60, where_40, add_187, eq_20, logical_not_40, any_21, logical_not_41, full_default_62, , sub, exp, div_20, where_41], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_14.run(buf481, arg1_1, buf487, 16, 960, stream=stream0)
        buf488 = empty_strided_cuda((16, 1, 128, 8), (1024, 16384, 1, 128), torch.float32)
        # Topologically Sorted Source Nodes: [mul_499, sum_187], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf487, arg191_1, buf488, 16384, 120, stream=stream0)
        buf489 = reinterpret_tensor(buf480, (16, 1, 128), (128, 128, 1), 0); del buf480  # reuse
        # Topologically Sorted Source Nodes: [mul_499, sum_187], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf488, buf489, 2048, 8, stream=stream0)
        buf490 = buf473; del buf473  # reuse
        # Topologically Sorted Source Nodes: [mul_500, sum_188], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_17.run(buf489, arg192_1, buf490, 2048, 2048, stream=stream0)
        buf492 = reinterpret_tensor(buf489, (1, 2048), (2048, 1), 0); del buf489  # reuse
        # Topologically Sorted Source Nodes: [add_181, add_188, convert_element_type_686, pow_84, mean_83, convert_element_type_688], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_30.run(buf468, buf471, buf490, arg193_1, buf492, 1, 2048, stream=stream0)
        buf493 = buf470; del buf470  # reuse
        # Topologically Sorted Source Nodes: [mul_503, sum_189], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_19.run(buf492, arg194_1, buf493, 12288, 2048, stream=stream0)
        buf494 = buf492; del buf492  # reuse
        # Topologically Sorted Source Nodes: [mul_506, sum_190], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf493, arg195_1, buf494, 2048, 6144, stream=stream0)
        buf496 = buf449; del buf449  # reuse
        # Topologically Sorted Source Nodes: [add_181, add_188, add_190, convert_element_type_696, pow_85, mean_84, add_191, rsqrt_84, mul_507, convert_element_type_697, mul_508], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_mul_pow_rsqrt_31.run(buf468, buf471, buf490, buf494, arg196_1, buf496, 1, 2048, stream=stream0)
        buf497 = buf424; del buf424  # reuse
        # Topologically Sorted Source Nodes: [mul_509, sum_191], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_22.run(buf496, arg197_1, buf497, 2048, 2048, stream=stream0)
        buf504 = reinterpret_tensor(buf450, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf450  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_701, pow_86, mean_85, mul_516, cat_42, mul_517, add_194, convert_element_type_711, mul_520], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf497, arg198_1, buf4, buf504, 16, 128, stream=stream0)
        buf500 = buf476; del buf476  # reuse
        # Topologically Sorted Source Nodes: [mul_512, sum_192, mul_515, sum_193, index_put_43], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_23.run(buf496, arg199_1, arg201_1, arg1_1, buf500, arg203_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_706, pow_87, mean_86, mul_518, cat_43, mul_519, add_195, index_put_42], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_12.run(buf500, arg200_1, arg1_1, buf4, arg202_1, 8, 128, stream=stream0)
        buf505 = reinterpret_tensor(buf487, (16, 1, 960), (960, 960, 1), 0); del buf487  # reuse
        # Topologically Sorted Source Nodes: [mul_522, sum_194], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf504, arg202_1, buf505, 15360, 128, stream=stream0)
        buf511 = empty_strided_cuda((1, 16, 1, 960), (15360, 960, 15360, 1), torch.float32)
        # Topologically Sorted Source Nodes: [full_default_64, full_default_63, where_42, add_196, eq_21, logical_not_42, any_22, logical_not_43, full_default_65, , sub, exp, div_21, where_43], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_14.run(buf505, arg1_1, buf511, 16, 960, stream=stream0)
        buf512 = empty_strided_cuda((16, 1, 128, 8), (1024, 16384, 1, 128), torch.float32)
        # Topologically Sorted Source Nodes: [mul_523, sum_196], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf511, arg203_1, buf512, 16384, 120, stream=stream0)
        buf513 = reinterpret_tensor(buf504, (16, 1, 128), (128, 128, 1), 0); del buf504  # reuse
        # Topologically Sorted Source Nodes: [mul_523, sum_196], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf512, buf513, 2048, 8, stream=stream0)
        buf515 = buf468; del buf468  # reuse
        # Topologically Sorted Source Nodes: [add_181, add_188, add_190, mul_524, sum_197, add_197], Original ATen: [aten.add, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_sum_32.run(buf515, buf513, arg204_1, buf471, buf490, buf494, 2048, 2048, stream=stream0)
        buf516 = buf472; del buf472  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_719, pow_88, mean_87], Original ATen: [prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_mean_pow_25.run(buf515, buf516, 1, 2048, stream=stream0)
        buf517 = buf493; del buf493  # reuse
        # Topologically Sorted Source Nodes: [mul_527, sum_198], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_26.run(arg205_1, buf515, buf516, arg206_1, buf517, 12288, 2048, stream=stream0)
        buf518 = reinterpret_tensor(buf513, (1, 2048), (2048, 1), 0); del buf513  # reuse
        # Topologically Sorted Source Nodes: [mul_530, sum_199], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf517, arg207_1, buf518, 2048, 6144, stream=stream0)
        buf519 = buf516; del buf516  # reuse
        # Topologically Sorted Source Nodes: [add_199, convert_element_type_729, pow_89, mean_88], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_27.run(buf515, buf518, buf519, 1, 2048, stream=stream0)
        buf520 = buf494; del buf494  # reuse
        # Topologically Sorted Source Nodes: [mul_533, sum_200], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_28.run(arg208_1, buf515, buf518, buf519, arg209_1, buf520, 2048, 2048, stream=stream0)
        buf527 = reinterpret_tensor(buf490, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf490  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_734, pow_90, mean_89, mul_540, cat_44, mul_541, add_203, convert_element_type_744, mul_544], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf520, arg210_1, buf4, buf527, 16, 128, stream=stream0)
        buf523 = buf500; del buf500  # reuse
        # Topologically Sorted Source Nodes: [mul_536, sum_201, mul_539, sum_202, index_put_45], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_29.run(arg208_1, buf515, buf518, buf519, arg211_1, arg213_1, arg1_1, buf523, arg215_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_739, pow_91, mean_90, mul_542, cat_45, mul_543, add_204, index_put_44], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_12.run(buf523, arg212_1, arg1_1, buf4, arg214_1, 8, 128, stream=stream0)
        buf528 = reinterpret_tensor(buf511, (16, 1, 960), (960, 960, 1), 0); del buf511  # reuse
        # Topologically Sorted Source Nodes: [mul_546, sum_203], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf527, arg214_1, buf528, 15360, 128, stream=stream0)
        buf534 = empty_strided_cuda((1, 16, 1, 960), (15360, 960, 15360, 1), torch.float32)
        # Topologically Sorted Source Nodes: [full_default_67, full_default_66, where_44, add_205, eq_22, logical_not_44, any_23, logical_not_45, full_default_68, , sub, exp, div_22, where_45], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_14.run(buf528, arg1_1, buf534, 16, 960, stream=stream0)
        buf535 = empty_strided_cuda((16, 1, 128, 8), (1024, 16384, 1, 128), torch.float32)
        # Topologically Sorted Source Nodes: [mul_547, sum_205], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf534, arg215_1, buf535, 16384, 120, stream=stream0)
        buf536 = reinterpret_tensor(buf527, (16, 1, 128), (128, 128, 1), 0); del buf527  # reuse
        # Topologically Sorted Source Nodes: [mul_547, sum_205], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf535, buf536, 2048, 8, stream=stream0)
        buf537 = buf520; del buf520  # reuse
        # Topologically Sorted Source Nodes: [mul_548, sum_206], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_17.run(buf536, arg216_1, buf537, 2048, 2048, stream=stream0)
        buf539 = reinterpret_tensor(buf536, (1, 2048), (2048, 1), 0); del buf536  # reuse
        # Topologically Sorted Source Nodes: [add_199, add_206, convert_element_type_752, pow_92, mean_91, convert_element_type_754], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_30.run(buf515, buf518, buf537, arg217_1, buf539, 1, 2048, stream=stream0)
        buf540 = buf517; del buf517  # reuse
        # Topologically Sorted Source Nodes: [mul_551, sum_207], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_19.run(buf539, arg218_1, buf540, 12288, 2048, stream=stream0)
        buf541 = buf539; del buf539  # reuse
        # Topologically Sorted Source Nodes: [mul_554, sum_208], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf540, arg219_1, buf541, 2048, 6144, stream=stream0)
        buf543 = buf496; del buf496  # reuse
        # Topologically Sorted Source Nodes: [add_199, add_206, add_208, convert_element_type_762, pow_93, mean_92, add_209, rsqrt_92, mul_555, convert_element_type_763, mul_556], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_mul_pow_rsqrt_31.run(buf515, buf518, buf537, buf541, arg220_1, buf543, 1, 2048, stream=stream0)
        buf544 = buf471; del buf471  # reuse
        # Topologically Sorted Source Nodes: [mul_557, sum_209], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_22.run(buf543, arg221_1, buf544, 2048, 2048, stream=stream0)
        buf551 = reinterpret_tensor(buf497, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf497  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_767, pow_94, mean_93, mul_564, cat_46, mul_565, add_212, convert_element_type_777, mul_568], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf544, arg222_1, buf4, buf551, 16, 128, stream=stream0)
        buf547 = buf523; del buf523  # reuse
        # Topologically Sorted Source Nodes: [mul_560, sum_210, mul_563, sum_211, index_put_47], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_23.run(buf543, arg223_1, arg225_1, arg1_1, buf547, arg227_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_772, pow_95, mean_94, mul_566, cat_47, mul_567, add_213, index_put_46], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_12.run(buf547, arg224_1, arg1_1, buf4, arg226_1, 8, 128, stream=stream0)
        buf552 = reinterpret_tensor(buf534, (16, 1, 960), (960, 960, 1), 0); del buf534  # reuse
        # Topologically Sorted Source Nodes: [mul_570, sum_212], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf551, arg226_1, buf552, 15360, 128, stream=stream0)
        buf558 = empty_strided_cuda((1, 16, 1, 960), (15360, 960, 15360, 1), torch.float32)
        # Topologically Sorted Source Nodes: [full_default_70, full_default_69, where_46, add_214, eq_23, logical_not_46, any_24, logical_not_47, full_default_71, , sub, exp, div_23, where_47], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_14.run(buf552, arg1_1, buf558, 16, 960, stream=stream0)
        buf559 = empty_strided_cuda((16, 1, 128, 8), (1024, 16384, 1, 128), torch.float32)
        # Topologically Sorted Source Nodes: [mul_571, sum_214], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf558, arg227_1, buf559, 16384, 120, stream=stream0)
        buf560 = reinterpret_tensor(buf551, (16, 1, 128), (128, 128, 1), 0); del buf551  # reuse
        # Topologically Sorted Source Nodes: [mul_571, sum_214], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf559, buf560, 2048, 8, stream=stream0)
        buf562 = buf515; del buf515  # reuse
        # Topologically Sorted Source Nodes: [add_199, add_206, add_208, mul_572, sum_215, add_215], Original ATen: [aten.add, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_sum_32.run(buf562, buf560, arg228_1, buf518, buf537, buf541, 2048, 2048, stream=stream0)
        buf563 = buf519; del buf519  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_785, pow_96, mean_95], Original ATen: [prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_mean_pow_25.run(buf562, buf563, 1, 2048, stream=stream0)
        buf564 = buf540; del buf540  # reuse
        # Topologically Sorted Source Nodes: [mul_575, sum_216], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_26.run(arg229_1, buf562, buf563, arg230_1, buf564, 12288, 2048, stream=stream0)
        buf565 = reinterpret_tensor(buf560, (1, 2048), (2048, 1), 0); del buf560  # reuse
        # Topologically Sorted Source Nodes: [mul_578, sum_217], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf564, arg231_1, buf565, 2048, 6144, stream=stream0)
        buf566 = buf563; del buf563  # reuse
        # Topologically Sorted Source Nodes: [add_217, convert_element_type_795, pow_97, mean_96], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_27.run(buf562, buf565, buf566, 1, 2048, stream=stream0)
        buf567 = buf541; del buf541  # reuse
        # Topologically Sorted Source Nodes: [mul_581, sum_218], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_28.run(arg232_1, buf562, buf565, buf566, arg233_1, buf567, 2048, 2048, stream=stream0)
        buf574 = reinterpret_tensor(buf537, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf537  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_800, pow_98, mean_97, mul_588, cat_48, mul_589, add_221, convert_element_type_810, mul_592], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf567, arg234_1, buf4, buf574, 16, 128, stream=stream0)
        buf570 = buf547; del buf547  # reuse
        # Topologically Sorted Source Nodes: [mul_584, sum_219, mul_587, sum_220, index_put_49], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_29.run(arg232_1, buf562, buf565, buf566, arg235_1, arg237_1, arg1_1, buf570, arg239_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_805, pow_99, mean_98, mul_590, cat_49, mul_591, add_222, index_put_48], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_12.run(buf570, arg236_1, arg1_1, buf4, arg238_1, 8, 128, stream=stream0)
        buf575 = reinterpret_tensor(buf558, (16, 1, 960), (960, 960, 1), 0); del buf558  # reuse
        # Topologically Sorted Source Nodes: [mul_594, sum_221], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf574, arg238_1, buf575, 15360, 128, stream=stream0)
        buf581 = empty_strided_cuda((1, 16, 1, 960), (15360, 960, 15360, 1), torch.float32)
        # Topologically Sorted Source Nodes: [full_default_73, full_default_72, where_48, add_223, eq_24, logical_not_48, any_25, logical_not_49, full_default_74, , sub, exp, div_24, where_49], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_14.run(buf575, arg1_1, buf581, 16, 960, stream=stream0)
        buf582 = empty_strided_cuda((16, 1, 128, 8), (1024, 16384, 1, 128), torch.float32)
        # Topologically Sorted Source Nodes: [mul_595, sum_223], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf581, arg239_1, buf582, 16384, 120, stream=stream0)
        buf583 = reinterpret_tensor(buf574, (16, 1, 128), (128, 128, 1), 0); del buf574  # reuse
        # Topologically Sorted Source Nodes: [mul_595, sum_223], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf582, buf583, 2048, 8, stream=stream0)
        buf584 = buf567; del buf567  # reuse
        # Topologically Sorted Source Nodes: [mul_596, sum_224], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_17.run(buf583, arg240_1, buf584, 2048, 2048, stream=stream0)
        buf586 = reinterpret_tensor(buf583, (1, 2048), (2048, 1), 0); del buf583  # reuse
        # Topologically Sorted Source Nodes: [add_217, add_224, convert_element_type_818, pow_100, mean_99, convert_element_type_820], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_30.run(buf562, buf565, buf584, arg241_1, buf586, 1, 2048, stream=stream0)
        buf587 = buf564; del buf564  # reuse
        # Topologically Sorted Source Nodes: [mul_599, sum_225], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_19.run(buf586, arg242_1, buf587, 12288, 2048, stream=stream0)
        buf588 = buf586; del buf586  # reuse
        # Topologically Sorted Source Nodes: [mul_602, sum_226], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf587, arg243_1, buf588, 2048, 6144, stream=stream0)
        buf590 = buf543; del buf543  # reuse
        # Topologically Sorted Source Nodes: [add_217, add_224, add_226, convert_element_type_828, pow_101, mean_100, add_227, rsqrt_100, mul_603, convert_element_type_829, mul_604], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_mul_pow_rsqrt_31.run(buf562, buf565, buf584, buf588, arg244_1, buf590, 1, 2048, stream=stream0)
        buf591 = buf518; del buf518  # reuse
        # Topologically Sorted Source Nodes: [mul_605, sum_227], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_22.run(buf590, arg245_1, buf591, 2048, 2048, stream=stream0)
        buf598 = reinterpret_tensor(buf544, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf544  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_833, pow_102, mean_101, mul_612, cat_50, mul_613, add_230, convert_element_type_843, mul_616], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf591, arg246_1, buf4, buf598, 16, 128, stream=stream0)
        buf594 = buf570; del buf570  # reuse
        # Topologically Sorted Source Nodes: [mul_608, sum_228, mul_611, sum_229, index_put_51], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_23.run(buf590, arg247_1, arg249_1, arg1_1, buf594, arg251_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_838, pow_103, mean_102, mul_614, cat_51, mul_615, add_231, index_put_50], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_12.run(buf594, arg248_1, arg1_1, buf4, arg250_1, 8, 128, stream=stream0)
        buf599 = reinterpret_tensor(buf581, (16, 1, 960), (960, 960, 1), 0); del buf581  # reuse
        # Topologically Sorted Source Nodes: [mul_618, sum_230], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf598, arg250_1, buf599, 15360, 128, stream=stream0)
        buf605 = empty_strided_cuda((1, 16, 1, 960), (15360, 960, 15360, 1), torch.float32)
        # Topologically Sorted Source Nodes: [full_default_76, full_default_75, where_50, add_232, eq_25, logical_not_50, any_26, logical_not_51, full_default_77, , sub, exp, div_25, where_51], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_14.run(buf599, arg1_1, buf605, 16, 960, stream=stream0)
        buf606 = empty_strided_cuda((16, 1, 128, 8), (1024, 16384, 1, 128), torch.float32)
        # Topologically Sorted Source Nodes: [mul_619, sum_232], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf605, arg251_1, buf606, 16384, 120, stream=stream0)
        buf607 = reinterpret_tensor(buf598, (16, 1, 128), (128, 128, 1), 0); del buf598  # reuse
        # Topologically Sorted Source Nodes: [mul_619, sum_232], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf606, buf607, 2048, 8, stream=stream0)
        buf609 = buf562; del buf562  # reuse
        # Topologically Sorted Source Nodes: [add_217, add_224, add_226, mul_620, sum_233, add_233], Original ATen: [aten.add, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_sum_32.run(buf609, buf607, arg252_1, buf565, buf584, buf588, 2048, 2048, stream=stream0)
        buf610 = buf566; del buf566  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_851, pow_104, mean_103], Original ATen: [prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_mean_pow_25.run(buf609, buf610, 1, 2048, stream=stream0)
        buf611 = buf587; del buf587  # reuse
        # Topologically Sorted Source Nodes: [mul_623, sum_234], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_26.run(arg253_1, buf609, buf610, arg254_1, buf611, 12288, 2048, stream=stream0)
        buf612 = reinterpret_tensor(buf607, (1, 2048), (2048, 1), 0); del buf607  # reuse
        # Topologically Sorted Source Nodes: [mul_626, sum_235], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf611, arg255_1, buf612, 2048, 6144, stream=stream0)
        buf613 = buf610; del buf610  # reuse
        # Topologically Sorted Source Nodes: [add_235, convert_element_type_861, pow_105, mean_104], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_27.run(buf609, buf612, buf613, 1, 2048, stream=stream0)
        buf614 = buf588; del buf588  # reuse
        # Topologically Sorted Source Nodes: [mul_629, sum_236], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_28.run(arg256_1, buf609, buf612, buf613, arg257_1, buf614, 2048, 2048, stream=stream0)
        buf621 = reinterpret_tensor(buf584, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf584  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_866, pow_106, mean_105, mul_636, cat_52, mul_637, add_239, convert_element_type_876, mul_640], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf614, arg258_1, buf4, buf621, 16, 128, stream=stream0)
        buf617 = buf594; del buf594  # reuse
        # Topologically Sorted Source Nodes: [mul_632, sum_237, mul_635, sum_238, index_put_53], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_29.run(arg256_1, buf609, buf612, buf613, arg259_1, arg261_1, arg1_1, buf617, arg263_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_871, pow_107, mean_106, mul_638, cat_53, mul_639, add_240, index_put_52], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_12.run(buf617, arg260_1, arg1_1, buf4, arg262_1, 8, 128, stream=stream0)
        buf622 = reinterpret_tensor(buf605, (16, 1, 960), (960, 960, 1), 0); del buf605  # reuse
        # Topologically Sorted Source Nodes: [mul_642, sum_239], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf621, arg262_1, buf622, 15360, 128, stream=stream0)
        buf628 = empty_strided_cuda((1, 16, 1, 960), (15360, 960, 15360, 1), torch.float32)
        # Topologically Sorted Source Nodes: [full_default_79, full_default_78, where_52, add_241, eq_26, logical_not_52, any_27, logical_not_53, full_default_80, , sub, exp, div_26, where_53], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_14.run(buf622, arg1_1, buf628, 16, 960, stream=stream0)
        buf629 = empty_strided_cuda((16, 1, 128, 8), (1024, 16384, 1, 128), torch.float32)
        # Topologically Sorted Source Nodes: [mul_643, sum_241], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf628, arg263_1, buf629, 16384, 120, stream=stream0)
        buf630 = reinterpret_tensor(buf621, (16, 1, 128), (128, 128, 1), 0); del buf621  # reuse
        # Topologically Sorted Source Nodes: [mul_643, sum_241], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf629, buf630, 2048, 8, stream=stream0)
        buf631 = buf614; del buf614  # reuse
        # Topologically Sorted Source Nodes: [mul_644, sum_242], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_17.run(buf630, arg264_1, buf631, 2048, 2048, stream=stream0)
        buf633 = reinterpret_tensor(buf630, (1, 2048), (2048, 1), 0); del buf630  # reuse
        # Topologically Sorted Source Nodes: [add_235, add_242, convert_element_type_884, pow_108, mean_107, convert_element_type_886], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_30.run(buf609, buf612, buf631, arg265_1, buf633, 1, 2048, stream=stream0)
        buf634 = buf611; del buf611  # reuse
        # Topologically Sorted Source Nodes: [mul_647, sum_243], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_19.run(buf633, arg266_1, buf634, 12288, 2048, stream=stream0)
        buf635 = buf633; del buf633  # reuse
        # Topologically Sorted Source Nodes: [mul_650, sum_244], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf634, arg267_1, buf635, 2048, 6144, stream=stream0)
        buf637 = buf590; del buf590  # reuse
        # Topologically Sorted Source Nodes: [add_235, add_242, add_244, convert_element_type_894, pow_109, mean_108, add_245, rsqrt_108, mul_651, convert_element_type_895, mul_652], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_mul_pow_rsqrt_31.run(buf609, buf612, buf631, buf635, arg268_1, buf637, 1, 2048, stream=stream0)
        buf638 = buf565; del buf565  # reuse
        # Topologically Sorted Source Nodes: [mul_653, sum_245], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_22.run(buf637, arg269_1, buf638, 2048, 2048, stream=stream0)
        buf645 = reinterpret_tensor(buf591, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf591  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_899, pow_110, mean_109, mul_660, cat_54, mul_661, add_248, convert_element_type_909, mul_664], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf638, arg270_1, buf4, buf645, 16, 128, stream=stream0)
        buf641 = buf617; del buf617  # reuse
        # Topologically Sorted Source Nodes: [mul_656, sum_246, mul_659, sum_247, index_put_55], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_23.run(buf637, arg271_1, arg273_1, arg1_1, buf641, arg275_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_904, pow_111, mean_110, mul_662, cat_55, mul_663, add_249, index_put_54], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_12.run(buf641, arg272_1, arg1_1, buf4, arg274_1, 8, 128, stream=stream0)
        del buf4
        buf646 = reinterpret_tensor(buf628, (16, 1, 960), (960, 960, 1), 0); del buf628  # reuse
        # Topologically Sorted Source Nodes: [mul_666, sum_248], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf645, arg274_1, buf646, 15360, 128, stream=stream0)
        buf652 = empty_strided_cuda((1, 16, 1, 960), (15360, 960, 15360, 1), torch.float32)
        # Topologically Sorted Source Nodes: [full_default_82, full_default_81, where_54, add_250, eq_27, logical_not_54, any_28, logical_not_55, full_default_83, , sub, exp, div_27, where_55], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_14.run(buf646, arg1_1, buf652, 16, 960, stream=stream0)
        buf653 = empty_strided_cuda((16, 1, 128, 8), (1024, 16384, 1, 128), torch.float32)
        # Topologically Sorted Source Nodes: [mul_667, sum_250], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf652, arg275_1, buf653, 16384, 120, stream=stream0)
        buf654 = reinterpret_tensor(buf645, (16, 1, 128), (128, 128, 1), 0); del buf645  # reuse
        # Topologically Sorted Source Nodes: [mul_667, sum_250], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf653, buf654, 2048, 8, stream=stream0)
        buf656 = buf609; del buf609  # reuse
        # Topologically Sorted Source Nodes: [add_235, add_242, add_244, mul_668, sum_251, add_251], Original ATen: [aten.add, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_sum_32.run(buf656, buf654, arg276_1, buf612, buf631, buf635, 2048, 2048, stream=stream0)
        buf657 = buf613; del buf613  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_917, pow_112, mean_111], Original ATen: [prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_mean_pow_25.run(buf656, buf657, 1, 2048, stream=stream0)
        buf658 = buf634; del buf634  # reuse
        # Topologically Sorted Source Nodes: [mul_671, sum_252], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_26.run(arg277_1, buf656, buf657, arg278_1, buf658, 12288, 2048, stream=stream0)
        buf659 = reinterpret_tensor(buf654, (1, 2048), (2048, 1), 0); del buf654  # reuse
        # Topologically Sorted Source Nodes: [mul_674, sum_253], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf658, arg279_1, buf659, 2048, 6144, stream=stream0)
        buf660 = buf657; del buf657  # reuse
        # Topologically Sorted Source Nodes: [add_253, convert_element_type_927, pow_113, mean_112], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_27.run(buf656, buf659, buf660, 1, 2048, stream=stream0)
        buf661 = empty_strided_cuda((1, 151936), (151936, 1), torch.float32)
        # Topologically Sorted Source Nodes: [logits_0], Original ATen: [aten.mm]
        stream0 = get_raw_stream(0)
        triton_red_fused_mm_33.run(arg280_1, buf656, buf659, buf660, arg282_1, buf661, 151936, 2048, stream=stream0)
        buf662 = empty_strided_cuda((1, ), (1, ), torch.int64)
        buf3244 = empty_strided_cuda((1, 4), (4, 1), torch.int64)
        buf3240 = reinterpret_tensor(buf3244, (1, 1), (4, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [logits_0, argmax, cat_224], Original ATen: [aten.mm, aten.argmax, aten.cat]
        stream0 = get_raw_stream(0)
        triton_red_fused_argmax_cat_mm_34.run(buf661, buf662, buf3240, 1, 151936, stream=stream0)
        buf663 = buf660; del buf660  # reuse
        # Topologically Sorted Source Nodes: [embedding_1, convert_element_type_87, pow_114, mean_113], Original ATen: [aten.embedding, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_embedding_mean_pow_0.run(buf662, arg282_1, buf663, 1, 2048, stream=stream0)
        buf669 = buf641; del buf641  # reuse
        # Topologically Sorted Source Nodes: [mul_685, sum_256, mul_688, sum_257, index_put_57], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_35.run(arg284_1, buf662, arg282_1, buf663, arg287_1, arg289_1, arg1_1, buf669, arg291_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_417, pow_116, mean_115, mul_691, cat_57, mul_692, add_262, index_put_56], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_36.run(buf669, arg288_1, arg1_1, buf667, arg290_1, 8, 128, stream=stream0)
        buf664 = buf659; del buf659  # reuse
        # Topologically Sorted Source Nodes: [mul_682, sum_255], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_5.run(arg284_1, buf662, arg282_1, buf663, arg285_1, buf664, 2048, 2048, stream=stream0)
        buf677 = reinterpret_tensor(buf635, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf635  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_252, pow_115, mean_114, mul_689, cat_56, mul_690, add_261, convert_element_type_582, mul_693], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf664, arg286_1, buf667, buf677, 16, 128, stream=stream0)
        buf678 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [mul_695, sum_258], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf677, arg290_1, buf678, 15360, 128, stream=stream0)
        buf687 = buf652; del buf652  # reuse
        # Topologically Sorted Source Nodes: [full_default_87, full_default_85, where_56, add_263, eq_28, logical_not_56, any_29, logical_not_57, full_default_89, , sub, exp, div_28, where_57], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_37.run(buf678, arg1_1, buf687, 16, 960, stream=stream0)
        buf688 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [mul_696, sum_260], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf687, arg291_1, buf688, 16384, 120, stream=stream0)
        buf689 = reinterpret_tensor(buf677, (16, 1, 128), (128, 128, 1), 0); del buf677  # reuse
        # Topologically Sorted Source Nodes: [mul_696, sum_260], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf688, buf689, 2048, 8, stream=stream0)
        buf690 = buf664; del buf664  # reuse
        # Topologically Sorted Source Nodes: [mul_697, sum_261], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_17.run(buf689, arg292_1, buf690, 2048, 2048, stream=stream0)
        buf692 = reinterpret_tensor(buf689, (1, 2048), (2048, 1), 0); del buf689  # reuse
        # Topologically Sorted Source Nodes: [embedding_1, add_264, convert_element_type_813, pow_117, mean_116, convert_element_type_879], Original ATen: [aten.embedding, aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_embedding_mean_pow_18.run(buf662, arg282_1, buf690, arg293_1, buf692, 1, 2048, stream=stream0)
        buf693 = buf658; del buf658  # reuse
        # Topologically Sorted Source Nodes: [mul_700, sum_262], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_19.run(buf692, arg294_1, buf693, 12288, 2048, stream=stream0)
        buf694 = buf692; del buf692  # reuse
        # Topologically Sorted Source Nodes: [mul_703, sum_263], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf693, arg295_1, buf694, 2048, 6144, stream=stream0)
        buf696 = buf656; del buf656  # reuse
        # Topologically Sorted Source Nodes: [embedding_1, add_264, add_266, convert_element_type_936, pow_118, mean_117, add_267, rsqrt_117, mul_704, convert_element_type_937, mul_705], Original ATen: [aten.embedding, aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_embedding_mean_mul_pow_rsqrt_21.run(buf662, arg282_1, buf690, buf694, arg296_1, buf696, 1, 2048, stream=stream0)
        buf700 = buf669; del buf669  # reuse
        # Topologically Sorted Source Nodes: [mul_709, sum_265, mul_712, sum_266, index_put_59], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_38.run(buf696, arg299_1, arg301_1, arg1_1, buf700, arg303_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_946, pow_120, mean_119, mul_715, cat_59, mul_716, add_271, index_put_58], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_36.run(buf700, arg300_1, arg1_1, buf667, arg302_1, 8, 128, stream=stream0)
        buf697 = buf631; del buf631  # reuse
        # Topologically Sorted Source Nodes: [mul_706, sum_264], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_22.run(buf696, arg297_1, buf697, 2048, 2048, stream=stream0)
        buf708 = reinterpret_tensor(buf612, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf612  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_941, pow_119, mean_118, mul_713, cat_58, mul_714, add_270, convert_element_type_951, mul_717], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf697, arg298_1, buf667, buf708, 16, 128, stream=stream0)
        buf709 = buf35; del buf35  # reuse
        # Topologically Sorted Source Nodes: [mul_719, sum_267], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf708, arg302_1, buf709, 15360, 128, stream=stream0)
        buf718 = buf687; del buf687  # reuse
        # Topologically Sorted Source Nodes: [full_default_93, full_default_91, where_58, add_272, eq_29, logical_not_58, any_30, logical_not_59, full_default_95, , sub, exp, div_29, where_59], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_37.run(buf709, arg1_1, buf718, 16, 960, stream=stream0)
        buf719 = buf42; del buf42  # reuse
        # Topologically Sorted Source Nodes: [mul_720, sum_269], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf718, arg303_1, buf719, 16384, 120, stream=stream0)
        buf720 = reinterpret_tensor(buf708, (16, 1, 128), (128, 128, 1), 0); del buf708  # reuse
        # Topologically Sorted Source Nodes: [mul_720, sum_269], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf719, buf720, 2048, 8, stream=stream0)
        buf722 = buf696; del buf696  # reuse
        # Topologically Sorted Source Nodes: [embedding_1, add_264, add_266, mul_721, sum_270, add_273], Original ATen: [aten.embedding, aten.add, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_embedding_mul_sum_24.run(buf720, arg304_1, buf662, arg282_1, buf690, buf694, buf722, 2048, 2048, stream=stream0)
        buf723 = buf663; del buf663  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_958, pow_121, mean_120], Original ATen: [prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_mean_pow_25.run(buf722, buf723, 1, 2048, stream=stream0)
        buf724 = buf693; del buf693  # reuse
        # Topologically Sorted Source Nodes: [mul_724, sum_271], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_26.run(arg305_1, buf722, buf723, arg306_1, buf724, 12288, 2048, stream=stream0)
        buf725 = reinterpret_tensor(buf720, (1, 2048), (2048, 1), 0); del buf720  # reuse
        # Topologically Sorted Source Nodes: [mul_727, sum_272], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf724, arg307_1, buf725, 2048, 6144, stream=stream0)
        buf726 = buf723; del buf723  # reuse
        # Topologically Sorted Source Nodes: [add_275, convert_element_type_968, pow_122, mean_121], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_27.run(buf722, buf725, buf726, 1, 2048, stream=stream0)
        buf730 = buf700; del buf700  # reuse
        # Topologically Sorted Source Nodes: [mul_733, sum_274, mul_736, sum_275, index_put_61], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_39.run(arg308_1, buf722, buf725, buf726, arg311_1, arg313_1, arg1_1, buf730, arg315_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_978, pow_124, mean_123, mul_739, cat_61, mul_740, add_280, index_put_60], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_36.run(buf730, arg312_1, arg1_1, buf667, arg314_1, 8, 128, stream=stream0)
        buf727 = buf694; del buf694  # reuse
        # Topologically Sorted Source Nodes: [mul_730, sum_273], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_28.run(arg308_1, buf722, buf725, buf726, arg309_1, buf727, 2048, 2048, stream=stream0)
        buf738 = reinterpret_tensor(buf690, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf690  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_973, pow_123, mean_122, mul_737, cat_60, mul_738, add_279, convert_element_type_983, mul_741], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf727, arg310_1, buf667, buf738, 16, 128, stream=stream0)
        buf739 = buf58; del buf58  # reuse
        # Topologically Sorted Source Nodes: [mul_743, sum_276], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf738, arg314_1, buf739, 15360, 128, stream=stream0)
        buf748 = buf718; del buf718  # reuse
        # Topologically Sorted Source Nodes: [full_default_99, full_default_97, where_60, add_281, eq_30, logical_not_60, any_31, logical_not_61, full_default_101, , sub, exp, div_30, where_61], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_37.run(buf739, arg1_1, buf748, 16, 960, stream=stream0)
        buf749 = buf65; del buf65  # reuse
        # Topologically Sorted Source Nodes: [mul_744, sum_278], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf748, arg315_1, buf749, 16384, 120, stream=stream0)
        buf750 = reinterpret_tensor(buf738, (16, 1, 128), (128, 128, 1), 0); del buf738  # reuse
        # Topologically Sorted Source Nodes: [mul_744, sum_278], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf749, buf750, 2048, 8, stream=stream0)
        buf751 = buf727; del buf727  # reuse
        # Topologically Sorted Source Nodes: [mul_745, sum_279], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_17.run(buf750, arg316_1, buf751, 2048, 2048, stream=stream0)
        buf753 = reinterpret_tensor(buf750, (1, 2048), (2048, 1), 0); del buf750  # reuse
        # Topologically Sorted Source Nodes: [add_275, add_282, convert_element_type_990, pow_125, mean_124, convert_element_type_992], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_30.run(buf722, buf725, buf751, arg317_1, buf753, 1, 2048, stream=stream0)
        buf754 = buf724; del buf724  # reuse
        # Topologically Sorted Source Nodes: [mul_748, sum_280], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_19.run(buf753, arg318_1, buf754, 12288, 2048, stream=stream0)
        buf755 = buf753; del buf753  # reuse
        # Topologically Sorted Source Nodes: [mul_751, sum_281], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf754, arg319_1, buf755, 2048, 6144, stream=stream0)
        buf757 = buf637; del buf637  # reuse
        # Topologically Sorted Source Nodes: [add_275, add_282, add_284, convert_element_type_1000, pow_126, mean_125, add_285, rsqrt_125, mul_752, convert_element_type_1001, mul_753], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_mul_pow_rsqrt_31.run(buf722, buf725, buf751, buf755, arg320_1, buf757, 1, 2048, stream=stream0)
        buf761 = buf730; del buf730  # reuse
        # Topologically Sorted Source Nodes: [mul_757, sum_283, mul_760, sum_284, index_put_63], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_38.run(buf757, arg323_1, arg325_1, arg1_1, buf761, arg327_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_1010, pow_128, mean_127, mul_763, cat_63, mul_764, add_289, index_put_62], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_36.run(buf761, arg324_1, arg1_1, buf667, arg326_1, 8, 128, stream=stream0)
        buf758 = buf697; del buf697  # reuse
        # Topologically Sorted Source Nodes: [mul_754, sum_282], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_22.run(buf757, arg321_1, buf758, 2048, 2048, stream=stream0)
        buf769 = reinterpret_tensor(buf638, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf638  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_1005, pow_127, mean_126, mul_761, cat_62, mul_762, add_288, convert_element_type_1015, mul_765], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf758, arg322_1, buf667, buf769, 16, 128, stream=stream0)
        buf770 = buf82; del buf82  # reuse
        # Topologically Sorted Source Nodes: [mul_767, sum_285], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf769, arg326_1, buf770, 15360, 128, stream=stream0)
        buf779 = buf748; del buf748  # reuse
        # Topologically Sorted Source Nodes: [full_default_105, full_default_103, where_62, add_290, eq_31, logical_not_62, any_32, logical_not_63, full_default_107, , sub, exp, div_31, where_63], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_37.run(buf770, arg1_1, buf779, 16, 960, stream=stream0)
        buf780 = buf89; del buf89  # reuse
        # Topologically Sorted Source Nodes: [mul_768, sum_287], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf779, arg327_1, buf780, 16384, 120, stream=stream0)
        buf781 = reinterpret_tensor(buf769, (16, 1, 128), (128, 128, 1), 0); del buf769  # reuse
        # Topologically Sorted Source Nodes: [mul_768, sum_287], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf780, buf781, 2048, 8, stream=stream0)
        buf783 = buf722; del buf722  # reuse
        # Topologically Sorted Source Nodes: [add_275, add_282, add_284, mul_769, sum_288, add_291], Original ATen: [aten.add, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_sum_32.run(buf783, buf781, arg328_1, buf725, buf751, buf755, 2048, 2048, stream=stream0)
        buf784 = buf726; del buf726  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_1022, pow_129, mean_128], Original ATen: [prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_mean_pow_25.run(buf783, buf784, 1, 2048, stream=stream0)
        buf785 = buf754; del buf754  # reuse
        # Topologically Sorted Source Nodes: [mul_772, sum_289], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_26.run(arg329_1, buf783, buf784, arg330_1, buf785, 12288, 2048, stream=stream0)
        buf786 = reinterpret_tensor(buf781, (1, 2048), (2048, 1), 0); del buf781  # reuse
        # Topologically Sorted Source Nodes: [mul_775, sum_290], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf785, arg331_1, buf786, 2048, 6144, stream=stream0)
        buf787 = buf784; del buf784  # reuse
        # Topologically Sorted Source Nodes: [add_293, convert_element_type_1032, pow_130, mean_129], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_27.run(buf783, buf786, buf787, 1, 2048, stream=stream0)
        buf791 = buf761; del buf761  # reuse
        # Topologically Sorted Source Nodes: [mul_781, sum_292, mul_784, sum_293, index_put_65], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_39.run(arg332_1, buf783, buf786, buf787, arg335_1, arg337_1, arg1_1, buf791, arg339_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_1042, pow_132, mean_131, mul_787, cat_65, mul_788, add_298, index_put_64], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_36.run(buf791, arg336_1, arg1_1, buf667, arg338_1, 8, 128, stream=stream0)
        buf788 = buf755; del buf755  # reuse
        # Topologically Sorted Source Nodes: [mul_778, sum_291], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_28.run(arg332_1, buf783, buf786, buf787, arg333_1, buf788, 2048, 2048, stream=stream0)
        buf799 = reinterpret_tensor(buf751, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf751  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_1037, pow_131, mean_130, mul_785, cat_64, mul_786, add_297, convert_element_type_1047, mul_789], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf788, arg334_1, buf667, buf799, 16, 128, stream=stream0)
        buf800 = buf105; del buf105  # reuse
        # Topologically Sorted Source Nodes: [mul_791, sum_294], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf799, arg338_1, buf800, 15360, 128, stream=stream0)
        buf809 = buf779; del buf779  # reuse
        # Topologically Sorted Source Nodes: [full_default_111, full_default_109, where_64, add_299, eq_32, logical_not_64, any_33, logical_not_65, full_default_113, , sub, exp, div_32, where_65], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_37.run(buf800, arg1_1, buf809, 16, 960, stream=stream0)
        buf810 = buf112; del buf112  # reuse
        # Topologically Sorted Source Nodes: [mul_792, sum_296], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf809, arg339_1, buf810, 16384, 120, stream=stream0)
        buf811 = reinterpret_tensor(buf799, (16, 1, 128), (128, 128, 1), 0); del buf799  # reuse
        # Topologically Sorted Source Nodes: [mul_792, sum_296], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf810, buf811, 2048, 8, stream=stream0)
        buf812 = buf788; del buf788  # reuse
        # Topologically Sorted Source Nodes: [mul_793, sum_297], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_17.run(buf811, arg340_1, buf812, 2048, 2048, stream=stream0)
        buf814 = reinterpret_tensor(buf811, (1, 2048), (2048, 1), 0); del buf811  # reuse
        # Topologically Sorted Source Nodes: [add_293, add_300, convert_element_type_1054, pow_133, mean_132, convert_element_type_1056], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_30.run(buf783, buf786, buf812, arg341_1, buf814, 1, 2048, stream=stream0)
        buf815 = buf785; del buf785  # reuse
        # Topologically Sorted Source Nodes: [mul_796, sum_298], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_19.run(buf814, arg2_1, buf815, 12288, 2048, stream=stream0)
        buf816 = buf814; del buf814  # reuse
        # Topologically Sorted Source Nodes: [mul_799, sum_299], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf815, arg3_1, buf816, 2048, 6144, stream=stream0)
        buf818 = buf757; del buf757  # reuse
        # Topologically Sorted Source Nodes: [add_293, add_300, add_302, convert_element_type_1064, pow_134, mean_133, add_303, rsqrt_133, mul_800, convert_element_type_1065, mul_801], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_mul_pow_rsqrt_31.run(buf783, buf786, buf812, buf816, arg4_1, buf818, 1, 2048, stream=stream0)
        buf822 = buf791; del buf791  # reuse
        # Topologically Sorted Source Nodes: [mul_805, sum_301, mul_808, sum_302, index_put_67], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_38.run(buf818, arg7_1, arg9_1, arg1_1, buf822, arg11_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_1074, pow_136, mean_135, mul_811, cat_67, mul_812, add_307, index_put_66], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_36.run(buf822, arg8_1, arg1_1, buf667, arg10_1, 8, 128, stream=stream0)
        buf819 = buf725; del buf725  # reuse
        # Topologically Sorted Source Nodes: [mul_802, sum_300], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_22.run(buf818, arg5_1, buf819, 2048, 2048, stream=stream0)
        buf830 = reinterpret_tensor(buf758, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf758  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_1069, pow_135, mean_134, mul_809, cat_66, mul_810, add_306, convert_element_type_1079, mul_813], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf819, arg6_1, buf667, buf830, 16, 128, stream=stream0)
        buf831 = buf129; del buf129  # reuse
        # Topologically Sorted Source Nodes: [mul_815, sum_303], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf830, arg10_1, buf831, 15360, 128, stream=stream0)
        buf840 = buf809; del buf809  # reuse
        # Topologically Sorted Source Nodes: [full_default_117, full_default_115, where_66, add_308, eq_33, logical_not_66, any_34, logical_not_67, full_default_119, , sub, exp, div_33, where_67], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_37.run(buf831, arg1_1, buf840, 16, 960, stream=stream0)
        buf841 = buf136; del buf136  # reuse
        # Topologically Sorted Source Nodes: [mul_816, sum_305], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf840, arg11_1, buf841, 16384, 120, stream=stream0)
        buf842 = reinterpret_tensor(buf830, (16, 1, 128), (128, 128, 1), 0); del buf830  # reuse
        # Topologically Sorted Source Nodes: [mul_816, sum_305], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf841, buf842, 2048, 8, stream=stream0)
        buf844 = buf783; del buf783  # reuse
        # Topologically Sorted Source Nodes: [add_293, add_300, add_302, mul_817, sum_306, add_309], Original ATen: [aten.add, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_sum_32.run(buf844, buf842, arg12_1, buf786, buf812, buf816, 2048, 2048, stream=stream0)
        buf845 = buf787; del buf787  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_1086, pow_137, mean_136], Original ATen: [prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_mean_pow_25.run(buf844, buf845, 1, 2048, stream=stream0)
        buf846 = buf815; del buf815  # reuse
        # Topologically Sorted Source Nodes: [mul_820, sum_307], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_26.run(arg13_1, buf844, buf845, arg14_1, buf846, 12288, 2048, stream=stream0)
        buf847 = reinterpret_tensor(buf842, (1, 2048), (2048, 1), 0); del buf842  # reuse
        # Topologically Sorted Source Nodes: [mul_823, sum_308], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf846, arg15_1, buf847, 2048, 6144, stream=stream0)
        buf848 = buf845; del buf845  # reuse
        # Topologically Sorted Source Nodes: [add_311, convert_element_type_1096, pow_138, mean_137], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_27.run(buf844, buf847, buf848, 1, 2048, stream=stream0)
        buf852 = buf822; del buf822  # reuse
        # Topologically Sorted Source Nodes: [mul_829, sum_310, mul_832, sum_311, index_put_69], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_39.run(arg16_1, buf844, buf847, buf848, arg19_1, arg21_1, arg1_1, buf852, arg23_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_1106, pow_140, mean_139, mul_835, cat_69, mul_836, add_316, index_put_68], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_36.run(buf852, arg20_1, arg1_1, buf667, arg22_1, 8, 128, stream=stream0)
        buf849 = buf816; del buf816  # reuse
        # Topologically Sorted Source Nodes: [mul_826, sum_309], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_28.run(arg16_1, buf844, buf847, buf848, arg17_1, buf849, 2048, 2048, stream=stream0)
        buf860 = reinterpret_tensor(buf812, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf812  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_1101, pow_139, mean_138, mul_833, cat_68, mul_834, add_315, convert_element_type_1111, mul_837], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf849, arg18_1, buf667, buf860, 16, 128, stream=stream0)
        buf861 = buf152; del buf152  # reuse
        # Topologically Sorted Source Nodes: [mul_839, sum_312], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf860, arg22_1, buf861, 15360, 128, stream=stream0)
        buf870 = buf840; del buf840  # reuse
        # Topologically Sorted Source Nodes: [full_default_123, full_default_121, where_68, add_317, eq_34, logical_not_68, any_35, logical_not_69, full_default_125, , sub, exp, div_34, where_69], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_37.run(buf861, arg1_1, buf870, 16, 960, stream=stream0)
        buf871 = buf159; del buf159  # reuse
        # Topologically Sorted Source Nodes: [mul_840, sum_314], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf870, arg23_1, buf871, 16384, 120, stream=stream0)
        buf872 = reinterpret_tensor(buf860, (16, 1, 128), (128, 128, 1), 0); del buf860  # reuse
        # Topologically Sorted Source Nodes: [mul_840, sum_314], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf871, buf872, 2048, 8, stream=stream0)
        buf873 = buf849; del buf849  # reuse
        # Topologically Sorted Source Nodes: [mul_841, sum_315], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_17.run(buf872, arg24_1, buf873, 2048, 2048, stream=stream0)
        buf875 = reinterpret_tensor(buf872, (1, 2048), (2048, 1), 0); del buf872  # reuse
        # Topologically Sorted Source Nodes: [add_311, add_318, convert_element_type_1118, pow_141, mean_140, convert_element_type_1120], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_30.run(buf844, buf847, buf873, arg25_1, buf875, 1, 2048, stream=stream0)
        buf876 = buf846; del buf846  # reuse
        # Topologically Sorted Source Nodes: [mul_844, sum_316], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_19.run(buf875, arg26_1, buf876, 12288, 2048, stream=stream0)
        buf877 = buf875; del buf875  # reuse
        # Topologically Sorted Source Nodes: [mul_847, sum_317], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf876, arg27_1, buf877, 2048, 6144, stream=stream0)
        buf879 = buf818; del buf818  # reuse
        # Topologically Sorted Source Nodes: [add_311, add_318, add_320, convert_element_type_1128, pow_142, mean_141, add_321, rsqrt_141, mul_848, convert_element_type_1129, mul_849], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_mul_pow_rsqrt_31.run(buf844, buf847, buf873, buf877, arg28_1, buf879, 1, 2048, stream=stream0)
        buf883 = buf852; del buf852  # reuse
        # Topologically Sorted Source Nodes: [mul_853, sum_319, mul_856, sum_320, index_put_71], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_38.run(buf879, arg31_1, arg33_1, arg1_1, buf883, arg35_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_1138, pow_144, mean_143, mul_859, cat_71, mul_860, add_325, index_put_70], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_36.run(buf883, arg32_1, arg1_1, buf667, arg34_1, 8, 128, stream=stream0)
        buf880 = buf786; del buf786  # reuse
        # Topologically Sorted Source Nodes: [mul_850, sum_318], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_22.run(buf879, arg29_1, buf880, 2048, 2048, stream=stream0)
        buf891 = reinterpret_tensor(buf819, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf819  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_1133, pow_143, mean_142, mul_857, cat_70, mul_858, add_324, convert_element_type_1143, mul_861], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf880, arg30_1, buf667, buf891, 16, 128, stream=stream0)
        buf892 = buf176; del buf176  # reuse
        # Topologically Sorted Source Nodes: [mul_863, sum_321], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf891, arg34_1, buf892, 15360, 128, stream=stream0)
        buf901 = buf870; del buf870  # reuse
        # Topologically Sorted Source Nodes: [full_default_129, full_default_127, where_70, add_326, eq_35, logical_not_70, any_36, logical_not_71, full_default_131, , sub, exp, div_35, where_71], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_37.run(buf892, arg1_1, buf901, 16, 960, stream=stream0)
        buf902 = buf183; del buf183  # reuse
        # Topologically Sorted Source Nodes: [mul_864, sum_323], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf901, arg35_1, buf902, 16384, 120, stream=stream0)
        buf903 = reinterpret_tensor(buf891, (16, 1, 128), (128, 128, 1), 0); del buf891  # reuse
        # Topologically Sorted Source Nodes: [mul_864, sum_323], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf902, buf903, 2048, 8, stream=stream0)
        buf905 = buf844; del buf844  # reuse
        # Topologically Sorted Source Nodes: [add_311, add_318, add_320, mul_865, sum_324, add_327], Original ATen: [aten.add, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_sum_32.run(buf905, buf903, arg36_1, buf847, buf873, buf877, 2048, 2048, stream=stream0)
        buf906 = buf848; del buf848  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_1150, pow_145, mean_144], Original ATen: [prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_mean_pow_25.run(buf905, buf906, 1, 2048, stream=stream0)
        buf907 = buf876; del buf876  # reuse
        # Topologically Sorted Source Nodes: [mul_868, sum_325], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_26.run(arg37_1, buf905, buf906, arg38_1, buf907, 12288, 2048, stream=stream0)
        buf908 = reinterpret_tensor(buf903, (1, 2048), (2048, 1), 0); del buf903  # reuse
        # Topologically Sorted Source Nodes: [mul_871, sum_326], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf907, arg39_1, buf908, 2048, 6144, stream=stream0)
        buf909 = buf906; del buf906  # reuse
        # Topologically Sorted Source Nodes: [add_329, convert_element_type_1160, pow_146, mean_145], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_27.run(buf905, buf908, buf909, 1, 2048, stream=stream0)
        buf913 = buf883; del buf883  # reuse
        # Topologically Sorted Source Nodes: [mul_877, sum_328, mul_880, sum_329, index_put_73], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_39.run(arg40_1, buf905, buf908, buf909, arg43_1, arg45_1, arg1_1, buf913, arg47_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_1170, pow_148, mean_147, mul_883, cat_73, mul_884, add_334, index_put_72], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_36.run(buf913, arg44_1, arg1_1, buf667, arg46_1, 8, 128, stream=stream0)
        buf910 = buf877; del buf877  # reuse
        # Topologically Sorted Source Nodes: [mul_874, sum_327], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_28.run(arg40_1, buf905, buf908, buf909, arg41_1, buf910, 2048, 2048, stream=stream0)
        buf921 = reinterpret_tensor(buf873, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf873  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_1165, pow_147, mean_146, mul_881, cat_72, mul_882, add_333, convert_element_type_1175, mul_885], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf910, arg42_1, buf667, buf921, 16, 128, stream=stream0)
        buf922 = buf199; del buf199  # reuse
        # Topologically Sorted Source Nodes: [mul_887, sum_330], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf921, arg46_1, buf922, 15360, 128, stream=stream0)
        buf931 = buf901; del buf901  # reuse
        # Topologically Sorted Source Nodes: [full_default_135, full_default_133, where_72, add_335, eq_36, logical_not_72, any_37, logical_not_73, full_default_137, , sub, exp, div_36, where_73], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_37.run(buf922, arg1_1, buf931, 16, 960, stream=stream0)
        buf932 = buf206; del buf206  # reuse
        # Topologically Sorted Source Nodes: [mul_888, sum_332], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf931, arg47_1, buf932, 16384, 120, stream=stream0)
        buf933 = reinterpret_tensor(buf921, (16, 1, 128), (128, 128, 1), 0); del buf921  # reuse
        # Topologically Sorted Source Nodes: [mul_888, sum_332], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf932, buf933, 2048, 8, stream=stream0)
        buf934 = buf910; del buf910  # reuse
        # Topologically Sorted Source Nodes: [mul_889, sum_333], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_17.run(buf933, arg48_1, buf934, 2048, 2048, stream=stream0)
        buf936 = reinterpret_tensor(buf933, (1, 2048), (2048, 1), 0); del buf933  # reuse
        # Topologically Sorted Source Nodes: [add_329, add_336, convert_element_type_1182, pow_149, mean_148, convert_element_type_1184], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_30.run(buf905, buf908, buf934, arg49_1, buf936, 1, 2048, stream=stream0)
        buf937 = buf907; del buf907  # reuse
        # Topologically Sorted Source Nodes: [mul_892, sum_334], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_19.run(buf936, arg50_1, buf937, 12288, 2048, stream=stream0)
        buf938 = buf936; del buf936  # reuse
        # Topologically Sorted Source Nodes: [mul_895, sum_335], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf937, arg51_1, buf938, 2048, 6144, stream=stream0)
        buf940 = buf879; del buf879  # reuse
        # Topologically Sorted Source Nodes: [add_329, add_336, add_338, convert_element_type_1192, pow_150, mean_149, add_339, rsqrt_149, mul_896, convert_element_type_1193, mul_897], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_mul_pow_rsqrt_31.run(buf905, buf908, buf934, buf938, arg52_1, buf940, 1, 2048, stream=stream0)
        buf944 = buf913; del buf913  # reuse
        # Topologically Sorted Source Nodes: [mul_901, sum_337, mul_904, sum_338, index_put_75], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_38.run(buf940, arg55_1, arg57_1, arg1_1, buf944, arg59_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_1202, pow_152, mean_151, mul_907, cat_75, mul_908, add_343, index_put_74], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_36.run(buf944, arg56_1, arg1_1, buf667, arg58_1, 8, 128, stream=stream0)
        buf941 = buf847; del buf847  # reuse
        # Topologically Sorted Source Nodes: [mul_898, sum_336], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_22.run(buf940, arg53_1, buf941, 2048, 2048, stream=stream0)
        buf952 = reinterpret_tensor(buf880, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf880  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_1197, pow_151, mean_150, mul_905, cat_74, mul_906, add_342, convert_element_type_1207, mul_909], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf941, arg54_1, buf667, buf952, 16, 128, stream=stream0)
        buf953 = buf223; del buf223  # reuse
        # Topologically Sorted Source Nodes: [mul_911, sum_339], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf952, arg58_1, buf953, 15360, 128, stream=stream0)
        buf962 = buf931; del buf931  # reuse
        # Topologically Sorted Source Nodes: [full_default_141, full_default_139, where_74, add_344, eq_37, logical_not_74, any_38, logical_not_75, full_default_143, , sub, exp, div_37, where_75], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_37.run(buf953, arg1_1, buf962, 16, 960, stream=stream0)
        buf963 = buf230; del buf230  # reuse
        # Topologically Sorted Source Nodes: [mul_912, sum_341], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf962, arg59_1, buf963, 16384, 120, stream=stream0)
        buf964 = reinterpret_tensor(buf952, (16, 1, 128), (128, 128, 1), 0); del buf952  # reuse
        # Topologically Sorted Source Nodes: [mul_912, sum_341], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf963, buf964, 2048, 8, stream=stream0)
        buf966 = buf905; del buf905  # reuse
        # Topologically Sorted Source Nodes: [add_329, add_336, add_338, mul_913, sum_342, add_345], Original ATen: [aten.add, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_sum_32.run(buf966, buf964, arg60_1, buf908, buf934, buf938, 2048, 2048, stream=stream0)
        buf967 = buf909; del buf909  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_1214, pow_153, mean_152], Original ATen: [prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_mean_pow_25.run(buf966, buf967, 1, 2048, stream=stream0)
        buf968 = buf937; del buf937  # reuse
        # Topologically Sorted Source Nodes: [mul_916, sum_343], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_26.run(arg61_1, buf966, buf967, arg62_1, buf968, 12288, 2048, stream=stream0)
        buf969 = reinterpret_tensor(buf964, (1, 2048), (2048, 1), 0); del buf964  # reuse
        # Topologically Sorted Source Nodes: [mul_919, sum_344], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf968, arg63_1, buf969, 2048, 6144, stream=stream0)
        buf970 = buf967; del buf967  # reuse
        # Topologically Sorted Source Nodes: [add_347, convert_element_type_1224, pow_154, mean_153], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_27.run(buf966, buf969, buf970, 1, 2048, stream=stream0)
        buf974 = buf944; del buf944  # reuse
        # Topologically Sorted Source Nodes: [mul_925, sum_346, mul_928, sum_347, index_put_77], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_39.run(arg64_1, buf966, buf969, buf970, arg67_1, arg69_1, arg1_1, buf974, arg71_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_1234, pow_156, mean_155, mul_931, cat_77, mul_932, add_352, index_put_76], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_36.run(buf974, arg68_1, arg1_1, buf667, arg70_1, 8, 128, stream=stream0)
        buf971 = buf938; del buf938  # reuse
        # Topologically Sorted Source Nodes: [mul_922, sum_345], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_28.run(arg64_1, buf966, buf969, buf970, arg65_1, buf971, 2048, 2048, stream=stream0)
        buf982 = reinterpret_tensor(buf934, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf934  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_1229, pow_155, mean_154, mul_929, cat_76, mul_930, add_351, convert_element_type_1239, mul_933], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf971, arg66_1, buf667, buf982, 16, 128, stream=stream0)
        buf983 = buf246; del buf246  # reuse
        # Topologically Sorted Source Nodes: [mul_935, sum_348], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf982, arg70_1, buf983, 15360, 128, stream=stream0)
        buf992 = buf962; del buf962  # reuse
        # Topologically Sorted Source Nodes: [full_default_147, full_default_145, where_76, add_353, eq_38, logical_not_76, any_39, logical_not_77, full_default_149, , sub, exp, div_38, where_77], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_37.run(buf983, arg1_1, buf992, 16, 960, stream=stream0)
        buf993 = buf253; del buf253  # reuse
        # Topologically Sorted Source Nodes: [mul_936, sum_350], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf992, arg71_1, buf993, 16384, 120, stream=stream0)
        buf994 = reinterpret_tensor(buf982, (16, 1, 128), (128, 128, 1), 0); del buf982  # reuse
        # Topologically Sorted Source Nodes: [mul_936, sum_350], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf993, buf994, 2048, 8, stream=stream0)
        buf995 = buf971; del buf971  # reuse
        # Topologically Sorted Source Nodes: [mul_937, sum_351], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_17.run(buf994, arg72_1, buf995, 2048, 2048, stream=stream0)
        buf997 = reinterpret_tensor(buf994, (1, 2048), (2048, 1), 0); del buf994  # reuse
        # Topologically Sorted Source Nodes: [add_347, add_354, convert_element_type_1246, pow_157, mean_156, convert_element_type_1248], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_30.run(buf966, buf969, buf995, arg73_1, buf997, 1, 2048, stream=stream0)
        buf998 = buf968; del buf968  # reuse
        # Topologically Sorted Source Nodes: [mul_940, sum_352], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_19.run(buf997, arg74_1, buf998, 12288, 2048, stream=stream0)
        buf999 = buf997; del buf997  # reuse
        # Topologically Sorted Source Nodes: [mul_943, sum_353], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf998, arg75_1, buf999, 2048, 6144, stream=stream0)
        buf1001 = buf940; del buf940  # reuse
        # Topologically Sorted Source Nodes: [add_347, add_354, add_356, convert_element_type_1256, pow_158, mean_157, add_357, rsqrt_157, mul_944, convert_element_type_1257, mul_945], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_mul_pow_rsqrt_31.run(buf966, buf969, buf995, buf999, arg76_1, buf1001, 1, 2048, stream=stream0)
        buf1005 = buf974; del buf974  # reuse
        # Topologically Sorted Source Nodes: [mul_949, sum_355, mul_952, sum_356, index_put_79], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_38.run(buf1001, arg79_1, arg81_1, arg1_1, buf1005, arg83_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_1266, pow_160, mean_159, mul_955, cat_79, mul_956, add_361, index_put_78], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_36.run(buf1005, arg80_1, arg1_1, buf667, arg82_1, 8, 128, stream=stream0)
        buf1002 = buf908; del buf908  # reuse
        # Topologically Sorted Source Nodes: [mul_946, sum_354], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_22.run(buf1001, arg77_1, buf1002, 2048, 2048, stream=stream0)
        buf1013 = reinterpret_tensor(buf941, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf941  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_1261, pow_159, mean_158, mul_953, cat_78, mul_954, add_360, convert_element_type_1271, mul_957], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf1002, arg78_1, buf667, buf1013, 16, 128, stream=stream0)
        buf1014 = buf270; del buf270  # reuse
        # Topologically Sorted Source Nodes: [mul_959, sum_357], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf1013, arg82_1, buf1014, 15360, 128, stream=stream0)
        buf1023 = buf992; del buf992  # reuse
        # Topologically Sorted Source Nodes: [full_default_153, full_default_151, where_78, add_362, eq_39, logical_not_78, any_40, logical_not_79, full_default_155, , sub, exp, div_39, where_79], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_37.run(buf1014, arg1_1, buf1023, 16, 960, stream=stream0)
        buf1024 = buf277; del buf277  # reuse
        # Topologically Sorted Source Nodes: [mul_960, sum_359], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf1023, arg83_1, buf1024, 16384, 120, stream=stream0)
        buf1025 = reinterpret_tensor(buf1013, (16, 1, 128), (128, 128, 1), 0); del buf1013  # reuse
        # Topologically Sorted Source Nodes: [mul_960, sum_359], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf1024, buf1025, 2048, 8, stream=stream0)
        buf1027 = buf966; del buf966  # reuse
        # Topologically Sorted Source Nodes: [add_347, add_354, add_356, mul_961, sum_360, add_363], Original ATen: [aten.add, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_sum_32.run(buf1027, buf1025, arg84_1, buf969, buf995, buf999, 2048, 2048, stream=stream0)
        buf1028 = buf970; del buf970  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_1278, pow_161, mean_160], Original ATen: [prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_mean_pow_25.run(buf1027, buf1028, 1, 2048, stream=stream0)
        buf1029 = buf998; del buf998  # reuse
        # Topologically Sorted Source Nodes: [mul_964, sum_361], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_26.run(arg85_1, buf1027, buf1028, arg86_1, buf1029, 12288, 2048, stream=stream0)
        buf1030 = buf999; del buf999  # reuse
        # Topologically Sorted Source Nodes: [mul_967, sum_362], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf1029, arg87_1, buf1030, 2048, 6144, stream=stream0)
        buf1031 = buf1028; del buf1028  # reuse
        # Topologically Sorted Source Nodes: [add_365, convert_element_type_1288, pow_162, mean_161], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_27.run(buf1027, buf1030, buf1031, 1, 2048, stream=stream0)
        buf1035 = buf1005; del buf1005  # reuse
        # Topologically Sorted Source Nodes: [mul_973, sum_364, mul_976, sum_365, index_put_81], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_39.run(arg88_1, buf1027, buf1030, buf1031, arg91_1, arg93_1, arg1_1, buf1035, arg95_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_1298, pow_164, mean_163, mul_979, cat_81, mul_980, add_370, index_put_80], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_36.run(buf1035, arg92_1, arg1_1, buf667, arg94_1, 8, 128, stream=stream0)
        buf1032 = buf995; del buf995  # reuse
        # Topologically Sorted Source Nodes: [mul_970, sum_363], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_28.run(arg88_1, buf1027, buf1030, buf1031, arg89_1, buf1032, 2048, 2048, stream=stream0)
        buf1043 = reinterpret_tensor(buf969, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf969  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_1293, pow_163, mean_162, mul_977, cat_80, mul_978, add_369, convert_element_type_1303, mul_981], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf1032, arg90_1, buf667, buf1043, 16, 128, stream=stream0)
        buf1044 = buf293; del buf293  # reuse
        # Topologically Sorted Source Nodes: [mul_983, sum_366], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf1043, arg94_1, buf1044, 15360, 128, stream=stream0)
        buf1053 = buf1023; del buf1023  # reuse
        # Topologically Sorted Source Nodes: [full_default_159, full_default_157, where_80, add_371, eq_40, logical_not_80, any_41, logical_not_81, full_default_161, , sub, exp, div_40, where_81], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_37.run(buf1044, arg1_1, buf1053, 16, 960, stream=stream0)
        buf1054 = buf300; del buf300  # reuse
        # Topologically Sorted Source Nodes: [mul_984, sum_368], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf1053, arg95_1, buf1054, 16384, 120, stream=stream0)
        buf1055 = reinterpret_tensor(buf1043, (16, 1, 128), (128, 128, 1), 0); del buf1043  # reuse
        # Topologically Sorted Source Nodes: [mul_984, sum_368], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf1054, buf1055, 2048, 8, stream=stream0)
        buf1056 = buf1032; del buf1032  # reuse
        # Topologically Sorted Source Nodes: [mul_985, sum_369], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_17.run(buf1055, arg96_1, buf1056, 2048, 2048, stream=stream0)
        buf1058 = reinterpret_tensor(buf1055, (1, 2048), (2048, 1), 0); del buf1055  # reuse
        # Topologically Sorted Source Nodes: [add_365, add_372, convert_element_type_1310, pow_165, mean_164, convert_element_type_1312], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_30.run(buf1027, buf1030, buf1056, arg97_1, buf1058, 1, 2048, stream=stream0)
        buf1059 = buf1029; del buf1029  # reuse
        # Topologically Sorted Source Nodes: [mul_988, sum_370], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_19.run(buf1058, arg98_1, buf1059, 12288, 2048, stream=stream0)
        buf1060 = buf1058; del buf1058  # reuse
        # Topologically Sorted Source Nodes: [mul_991, sum_371], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf1059, arg99_1, buf1060, 2048, 6144, stream=stream0)
        buf1062 = buf1001; del buf1001  # reuse
        # Topologically Sorted Source Nodes: [add_365, add_372, add_374, convert_element_type_1320, pow_166, mean_165, add_375, rsqrt_165, mul_992, convert_element_type_1321, mul_993], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_mul_pow_rsqrt_31.run(buf1027, buf1030, buf1056, buf1060, arg100_1, buf1062, 1, 2048, stream=stream0)
        buf1066 = buf1035; del buf1035  # reuse
        # Topologically Sorted Source Nodes: [mul_997, sum_373, mul_1000, sum_374, index_put_83], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_38.run(buf1062, arg103_1, arg105_1, arg1_1, buf1066, arg107_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_1330, pow_168, mean_167, mul_1003, cat_83, mul_1004, add_379, index_put_82], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_36.run(buf1066, arg104_1, arg1_1, buf667, arg106_1, 8, 128, stream=stream0)
        buf1063 = reinterpret_tensor(buf1025, (1, 2048), (2048, 1), 0); del buf1025  # reuse
        # Topologically Sorted Source Nodes: [mul_994, sum_372], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_22.run(buf1062, arg101_1, buf1063, 2048, 2048, stream=stream0)
        buf1074 = reinterpret_tensor(buf1002, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf1002  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_1325, pow_167, mean_166, mul_1001, cat_82, mul_1002, add_378, convert_element_type_1335, mul_1005], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf1063, arg102_1, buf667, buf1074, 16, 128, stream=stream0)
        buf1075 = buf317; del buf317  # reuse
        # Topologically Sorted Source Nodes: [mul_1007, sum_375], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf1074, arg106_1, buf1075, 15360, 128, stream=stream0)
        buf1084 = buf1053; del buf1053  # reuse
        # Topologically Sorted Source Nodes: [full_default_165, full_default_163, where_82, add_380, eq_41, logical_not_82, any_42, logical_not_83, full_default_167, , sub, exp, div_41, where_83], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_37.run(buf1075, arg1_1, buf1084, 16, 960, stream=stream0)
        buf1085 = buf324; del buf324  # reuse
        # Topologically Sorted Source Nodes: [mul_1008, sum_377], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf1084, arg107_1, buf1085, 16384, 120, stream=stream0)
        buf1086 = reinterpret_tensor(buf1074, (16, 1, 128), (128, 128, 1), 0); del buf1074  # reuse
        # Topologically Sorted Source Nodes: [mul_1008, sum_377], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf1085, buf1086, 2048, 8, stream=stream0)
        buf1088 = buf1027; del buf1027  # reuse
        # Topologically Sorted Source Nodes: [add_365, add_372, add_374, mul_1009, sum_378, add_381], Original ATen: [aten.add, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_sum_32.run(buf1088, buf1086, arg108_1, buf1030, buf1056, buf1060, 2048, 2048, stream=stream0)
        buf1089 = buf1031; del buf1031  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_1342, pow_169, mean_168], Original ATen: [prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_mean_pow_25.run(buf1088, buf1089, 1, 2048, stream=stream0)
        buf1090 = buf1059; del buf1059  # reuse
        # Topologically Sorted Source Nodes: [mul_1012, sum_379], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_26.run(arg109_1, buf1088, buf1089, arg110_1, buf1090, 12288, 2048, stream=stream0)
        buf1091 = reinterpret_tensor(buf1086, (1, 2048), (2048, 1), 0); del buf1086  # reuse
        # Topologically Sorted Source Nodes: [mul_1015, sum_380], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf1090, arg111_1, buf1091, 2048, 6144, stream=stream0)
        buf1092 = buf1089; del buf1089  # reuse
        # Topologically Sorted Source Nodes: [add_383, convert_element_type_1352, pow_170, mean_169], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_27.run(buf1088, buf1091, buf1092, 1, 2048, stream=stream0)
        buf1096 = buf1066; del buf1066  # reuse
        # Topologically Sorted Source Nodes: [mul_1021, sum_382, mul_1024, sum_383, index_put_85], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_39.run(arg112_1, buf1088, buf1091, buf1092, arg115_1, arg117_1, arg1_1, buf1096, arg119_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_1362, pow_172, mean_171, mul_1027, cat_85, mul_1028, add_388, index_put_84], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_36.run(buf1096, arg116_1, arg1_1, buf667, arg118_1, 8, 128, stream=stream0)
        buf1093 = buf1060; del buf1060  # reuse
        # Topologically Sorted Source Nodes: [mul_1018, sum_381], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_28.run(arg112_1, buf1088, buf1091, buf1092, arg113_1, buf1093, 2048, 2048, stream=stream0)
        buf1104 = reinterpret_tensor(buf1056, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf1056  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_1357, pow_171, mean_170, mul_1025, cat_84, mul_1026, add_387, convert_element_type_1367, mul_1029], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf1093, arg114_1, buf667, buf1104, 16, 128, stream=stream0)
        buf1105 = buf340; del buf340  # reuse
        # Topologically Sorted Source Nodes: [mul_1031, sum_384], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf1104, arg118_1, buf1105, 15360, 128, stream=stream0)
        buf1114 = buf1084; del buf1084  # reuse
        # Topologically Sorted Source Nodes: [full_default_171, full_default_169, where_84, add_389, eq_42, logical_not_84, any_43, logical_not_85, full_default_173, , sub, exp, div_42, where_85], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_37.run(buf1105, arg1_1, buf1114, 16, 960, stream=stream0)
        buf1115 = buf347; del buf347  # reuse
        # Topologically Sorted Source Nodes: [mul_1032, sum_386], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf1114, arg119_1, buf1115, 16384, 120, stream=stream0)
        buf1116 = reinterpret_tensor(buf1104, (16, 1, 128), (128, 128, 1), 0); del buf1104  # reuse
        # Topologically Sorted Source Nodes: [mul_1032, sum_386], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf1115, buf1116, 2048, 8, stream=stream0)
        buf1117 = buf1093; del buf1093  # reuse
        # Topologically Sorted Source Nodes: [mul_1033, sum_387], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_17.run(buf1116, arg120_1, buf1117, 2048, 2048, stream=stream0)
        buf1119 = reinterpret_tensor(buf1116, (1, 2048), (2048, 1), 0); del buf1116  # reuse
        # Topologically Sorted Source Nodes: [add_383, add_390, convert_element_type_1374, pow_173, mean_172, convert_element_type_1376], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_30.run(buf1088, buf1091, buf1117, arg121_1, buf1119, 1, 2048, stream=stream0)
        buf1120 = buf1090; del buf1090  # reuse
        # Topologically Sorted Source Nodes: [mul_1036, sum_388], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_19.run(buf1119, arg122_1, buf1120, 12288, 2048, stream=stream0)
        buf1121 = buf1119; del buf1119  # reuse
        # Topologically Sorted Source Nodes: [mul_1039, sum_389], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf1120, arg123_1, buf1121, 2048, 6144, stream=stream0)
        buf1123 = buf1062; del buf1062  # reuse
        # Topologically Sorted Source Nodes: [add_383, add_390, add_392, convert_element_type_1384, pow_174, mean_173, add_393, rsqrt_173, mul_1040, convert_element_type_1385, mul_1041], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_mul_pow_rsqrt_31.run(buf1088, buf1091, buf1117, buf1121, arg124_1, buf1123, 1, 2048, stream=stream0)
        buf1127 = buf1096; del buf1096  # reuse
        # Topologically Sorted Source Nodes: [mul_1045, sum_391, mul_1048, sum_392, index_put_87], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_38.run(buf1123, arg127_1, arg129_1, arg1_1, buf1127, arg131_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_1394, pow_176, mean_175, mul_1051, cat_87, mul_1052, add_397, index_put_86], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_36.run(buf1127, arg128_1, arg1_1, buf667, arg130_1, 8, 128, stream=stream0)
        buf1124 = buf1030; del buf1030  # reuse
        # Topologically Sorted Source Nodes: [mul_1042, sum_390], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_22.run(buf1123, arg125_1, buf1124, 2048, 2048, stream=stream0)
        buf1135 = reinterpret_tensor(buf1063, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf1063  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_1389, pow_175, mean_174, mul_1049, cat_86, mul_1050, add_396, convert_element_type_1399, mul_1053], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf1124, arg126_1, buf667, buf1135, 16, 128, stream=stream0)
        buf1136 = buf364; del buf364  # reuse
        # Topologically Sorted Source Nodes: [mul_1055, sum_393], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf1135, arg130_1, buf1136, 15360, 128, stream=stream0)
        buf1145 = buf1114; del buf1114  # reuse
        # Topologically Sorted Source Nodes: [full_default_177, full_default_175, where_86, add_398, eq_43, logical_not_86, any_44, logical_not_87, full_default_179, , sub, exp, div_43, where_87], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_37.run(buf1136, arg1_1, buf1145, 16, 960, stream=stream0)
        buf1146 = buf371; del buf371  # reuse
        # Topologically Sorted Source Nodes: [mul_1056, sum_395], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf1145, arg131_1, buf1146, 16384, 120, stream=stream0)
        buf1147 = reinterpret_tensor(buf1135, (16, 1, 128), (128, 128, 1), 0); del buf1135  # reuse
        # Topologically Sorted Source Nodes: [mul_1056, sum_395], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf1146, buf1147, 2048, 8, stream=stream0)
        buf1149 = buf1088; del buf1088  # reuse
        # Topologically Sorted Source Nodes: [add_383, add_390, add_392, mul_1057, sum_396, add_399], Original ATen: [aten.add, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_sum_32.run(buf1149, buf1147, arg132_1, buf1091, buf1117, buf1121, 2048, 2048, stream=stream0)
        buf1150 = buf1092; del buf1092  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_1406, pow_177, mean_176], Original ATen: [prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_mean_pow_25.run(buf1149, buf1150, 1, 2048, stream=stream0)
        buf1151 = buf1120; del buf1120  # reuse
        # Topologically Sorted Source Nodes: [mul_1060, sum_397], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_26.run(arg133_1, buf1149, buf1150, arg134_1, buf1151, 12288, 2048, stream=stream0)
        buf1152 = reinterpret_tensor(buf1147, (1, 2048), (2048, 1), 0); del buf1147  # reuse
        # Topologically Sorted Source Nodes: [mul_1063, sum_398], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf1151, arg135_1, buf1152, 2048, 6144, stream=stream0)
        buf1153 = buf1150; del buf1150  # reuse
        # Topologically Sorted Source Nodes: [add_401, convert_element_type_1416, pow_178, mean_177], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_27.run(buf1149, buf1152, buf1153, 1, 2048, stream=stream0)
        buf1157 = buf1127; del buf1127  # reuse
        # Topologically Sorted Source Nodes: [mul_1069, sum_400, mul_1072, sum_401, index_put_89], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_39.run(arg136_1, buf1149, buf1152, buf1153, arg139_1, arg141_1, arg1_1, buf1157, arg143_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_1426, pow_180, mean_179, mul_1075, cat_89, mul_1076, add_406, index_put_88], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_36.run(buf1157, arg140_1, arg1_1, buf667, arg142_1, 8, 128, stream=stream0)
        buf1154 = buf1121; del buf1121  # reuse
        # Topologically Sorted Source Nodes: [mul_1066, sum_399], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_28.run(arg136_1, buf1149, buf1152, buf1153, arg137_1, buf1154, 2048, 2048, stream=stream0)
        buf1165 = reinterpret_tensor(buf1117, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf1117  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_1421, pow_179, mean_178, mul_1073, cat_88, mul_1074, add_405, convert_element_type_1431, mul_1077], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf1154, arg138_1, buf667, buf1165, 16, 128, stream=stream0)
        buf1166 = buf387; del buf387  # reuse
        # Topologically Sorted Source Nodes: [mul_1079, sum_402], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf1165, arg142_1, buf1166, 15360, 128, stream=stream0)
        buf1175 = buf1145; del buf1145  # reuse
        # Topologically Sorted Source Nodes: [full_default_183, full_default_181, where_88, add_407, eq_44, logical_not_88, any_45, logical_not_89, full_default_185, , sub, exp, div_44, where_89], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_37.run(buf1166, arg1_1, buf1175, 16, 960, stream=stream0)
        buf1176 = buf394; del buf394  # reuse
        # Topologically Sorted Source Nodes: [mul_1080, sum_404], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf1175, arg143_1, buf1176, 16384, 120, stream=stream0)
        buf1177 = reinterpret_tensor(buf1165, (16, 1, 128), (128, 128, 1), 0); del buf1165  # reuse
        # Topologically Sorted Source Nodes: [mul_1080, sum_404], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf1176, buf1177, 2048, 8, stream=stream0)
        buf1178 = buf1154; del buf1154  # reuse
        # Topologically Sorted Source Nodes: [mul_1081, sum_405], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_17.run(buf1177, arg144_1, buf1178, 2048, 2048, stream=stream0)
        buf1180 = reinterpret_tensor(buf1177, (1, 2048), (2048, 1), 0); del buf1177  # reuse
        # Topologically Sorted Source Nodes: [add_401, add_408, convert_element_type_1438, pow_181, mean_180, convert_element_type_1440], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_30.run(buf1149, buf1152, buf1178, arg145_1, buf1180, 1, 2048, stream=stream0)
        buf1181 = buf1151; del buf1151  # reuse
        # Topologically Sorted Source Nodes: [mul_1084, sum_406], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_19.run(buf1180, arg146_1, buf1181, 12288, 2048, stream=stream0)
        buf1182 = buf1180; del buf1180  # reuse
        # Topologically Sorted Source Nodes: [mul_1087, sum_407], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf1181, arg147_1, buf1182, 2048, 6144, stream=stream0)
        buf1184 = buf1123; del buf1123  # reuse
        # Topologically Sorted Source Nodes: [add_401, add_408, add_410, convert_element_type_1448, pow_182, mean_181, add_411, rsqrt_181, mul_1088, convert_element_type_1449, mul_1089], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_mul_pow_rsqrt_31.run(buf1149, buf1152, buf1178, buf1182, arg148_1, buf1184, 1, 2048, stream=stream0)
        buf1188 = buf1157; del buf1157  # reuse
        # Topologically Sorted Source Nodes: [mul_1093, sum_409, mul_1096, sum_410, index_put_91], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_38.run(buf1184, arg151_1, arg153_1, arg1_1, buf1188, arg155_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_1458, pow_184, mean_183, mul_1099, cat_91, mul_1100, add_415, index_put_90], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_36.run(buf1188, arg152_1, arg1_1, buf667, arg154_1, 8, 128, stream=stream0)
        buf1185 = buf1091; del buf1091  # reuse
        # Topologically Sorted Source Nodes: [mul_1090, sum_408], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_22.run(buf1184, arg149_1, buf1185, 2048, 2048, stream=stream0)
        buf1196 = reinterpret_tensor(buf1124, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf1124  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_1453, pow_183, mean_182, mul_1097, cat_90, mul_1098, add_414, convert_element_type_1463, mul_1101], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf1185, arg150_1, buf667, buf1196, 16, 128, stream=stream0)
        buf1197 = buf411; del buf411  # reuse
        # Topologically Sorted Source Nodes: [mul_1103, sum_411], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf1196, arg154_1, buf1197, 15360, 128, stream=stream0)
        buf1206 = buf1175; del buf1175  # reuse
        # Topologically Sorted Source Nodes: [full_default_189, full_default_187, where_90, add_416, eq_45, logical_not_90, any_46, logical_not_91, full_default_191, , sub, exp, div_45, where_91], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_37.run(buf1197, arg1_1, buf1206, 16, 960, stream=stream0)
        buf1207 = buf418; del buf418  # reuse
        # Topologically Sorted Source Nodes: [mul_1104, sum_413], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf1206, arg155_1, buf1207, 16384, 120, stream=stream0)
        buf1208 = reinterpret_tensor(buf1196, (16, 1, 128), (128, 128, 1), 0); del buf1196  # reuse
        # Topologically Sorted Source Nodes: [mul_1104, sum_413], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf1207, buf1208, 2048, 8, stream=stream0)
        buf1210 = buf1149; del buf1149  # reuse
        # Topologically Sorted Source Nodes: [add_401, add_408, add_410, mul_1105, sum_414, add_417], Original ATen: [aten.add, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_sum_32.run(buf1210, buf1208, arg156_1, buf1152, buf1178, buf1182, 2048, 2048, stream=stream0)
        buf1211 = buf1153; del buf1153  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_1470, pow_185, mean_184], Original ATen: [prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_mean_pow_25.run(buf1210, buf1211, 1, 2048, stream=stream0)
        buf1212 = buf1181; del buf1181  # reuse
        # Topologically Sorted Source Nodes: [mul_1108, sum_415], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_26.run(arg157_1, buf1210, buf1211, arg158_1, buf1212, 12288, 2048, stream=stream0)
        buf1213 = reinterpret_tensor(buf1208, (1, 2048), (2048, 1), 0); del buf1208  # reuse
        # Topologically Sorted Source Nodes: [mul_1111, sum_416], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf1212, arg159_1, buf1213, 2048, 6144, stream=stream0)
        buf1214 = buf1211; del buf1211  # reuse
        # Topologically Sorted Source Nodes: [add_419, convert_element_type_1480, pow_186, mean_185], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_27.run(buf1210, buf1213, buf1214, 1, 2048, stream=stream0)
        buf1218 = buf1188; del buf1188  # reuse
        # Topologically Sorted Source Nodes: [mul_1117, sum_418, mul_1120, sum_419, index_put_93], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_39.run(arg160_1, buf1210, buf1213, buf1214, arg163_1, arg165_1, arg1_1, buf1218, arg167_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_1490, pow_188, mean_187, mul_1123, cat_93, mul_1124, add_424, index_put_92], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_36.run(buf1218, arg164_1, arg1_1, buf667, arg166_1, 8, 128, stream=stream0)
        buf1215 = buf1182; del buf1182  # reuse
        # Topologically Sorted Source Nodes: [mul_1114, sum_417], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_28.run(arg160_1, buf1210, buf1213, buf1214, arg161_1, buf1215, 2048, 2048, stream=stream0)
        buf1226 = reinterpret_tensor(buf1178, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf1178  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_1485, pow_187, mean_186, mul_1121, cat_92, mul_1122, add_423, convert_element_type_1495, mul_1125], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf1215, arg162_1, buf667, buf1226, 16, 128, stream=stream0)
        buf1227 = buf434; del buf434  # reuse
        # Topologically Sorted Source Nodes: [mul_1127, sum_420], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf1226, arg166_1, buf1227, 15360, 128, stream=stream0)
        buf1236 = buf1206; del buf1206  # reuse
        # Topologically Sorted Source Nodes: [full_default_195, full_default_193, where_92, add_425, eq_46, logical_not_92, any_47, logical_not_93, full_default_197, , sub, exp, div_46, where_93], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_37.run(buf1227, arg1_1, buf1236, 16, 960, stream=stream0)
        buf1237 = buf441; del buf441  # reuse
        # Topologically Sorted Source Nodes: [mul_1128, sum_422], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf1236, arg167_1, buf1237, 16384, 120, stream=stream0)
        buf1238 = reinterpret_tensor(buf1226, (16, 1, 128), (128, 128, 1), 0); del buf1226  # reuse
        # Topologically Sorted Source Nodes: [mul_1128, sum_422], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf1237, buf1238, 2048, 8, stream=stream0)
        buf1239 = buf1215; del buf1215  # reuse
        # Topologically Sorted Source Nodes: [mul_1129, sum_423], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_17.run(buf1238, arg168_1, buf1239, 2048, 2048, stream=stream0)
        buf1241 = reinterpret_tensor(buf1238, (1, 2048), (2048, 1), 0); del buf1238  # reuse
        # Topologically Sorted Source Nodes: [add_419, add_426, convert_element_type_1502, pow_189, mean_188, convert_element_type_1504], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_30.run(buf1210, buf1213, buf1239, arg169_1, buf1241, 1, 2048, stream=stream0)
        buf1242 = buf1212; del buf1212  # reuse
        # Topologically Sorted Source Nodes: [mul_1132, sum_424], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_19.run(buf1241, arg170_1, buf1242, 12288, 2048, stream=stream0)
        buf1243 = buf1241; del buf1241  # reuse
        # Topologically Sorted Source Nodes: [mul_1135, sum_425], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf1242, arg171_1, buf1243, 2048, 6144, stream=stream0)
        buf1245 = buf1184; del buf1184  # reuse
        # Topologically Sorted Source Nodes: [add_419, add_426, add_428, convert_element_type_1512, pow_190, mean_189, add_429, rsqrt_189, mul_1136, convert_element_type_1513, mul_1137], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_mul_pow_rsqrt_31.run(buf1210, buf1213, buf1239, buf1243, arg172_1, buf1245, 1, 2048, stream=stream0)
        buf1249 = buf1218; del buf1218  # reuse
        # Topologically Sorted Source Nodes: [mul_1141, sum_427, mul_1144, sum_428, index_put_95], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_38.run(buf1245, arg175_1, arg177_1, arg1_1, buf1249, arg179_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_1522, pow_192, mean_191, mul_1147, cat_95, mul_1148, add_433, index_put_94], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_36.run(buf1249, arg176_1, arg1_1, buf667, arg178_1, 8, 128, stream=stream0)
        buf1246 = buf1152; del buf1152  # reuse
        # Topologically Sorted Source Nodes: [mul_1138, sum_426], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_22.run(buf1245, arg173_1, buf1246, 2048, 2048, stream=stream0)
        buf1257 = reinterpret_tensor(buf1185, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf1185  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_1517, pow_191, mean_190, mul_1145, cat_94, mul_1146, add_432, convert_element_type_1527, mul_1149], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf1246, arg174_1, buf667, buf1257, 16, 128, stream=stream0)
        buf1258 = buf458; del buf458  # reuse
        # Topologically Sorted Source Nodes: [mul_1151, sum_429], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf1257, arg178_1, buf1258, 15360, 128, stream=stream0)
        buf1267 = buf1236; del buf1236  # reuse
        # Topologically Sorted Source Nodes: [full_default_201, full_default_199, where_94, add_434, eq_47, logical_not_94, any_48, logical_not_95, full_default_203, , sub, exp, div_47, where_95], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_37.run(buf1258, arg1_1, buf1267, 16, 960, stream=stream0)
        buf1268 = buf465; del buf465  # reuse
        # Topologically Sorted Source Nodes: [mul_1152, sum_431], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf1267, arg179_1, buf1268, 16384, 120, stream=stream0)
        buf1269 = reinterpret_tensor(buf1257, (16, 1, 128), (128, 128, 1), 0); del buf1257  # reuse
        # Topologically Sorted Source Nodes: [mul_1152, sum_431], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf1268, buf1269, 2048, 8, stream=stream0)
        buf1271 = buf1210; del buf1210  # reuse
        # Topologically Sorted Source Nodes: [add_419, add_426, add_428, mul_1153, sum_432, add_435], Original ATen: [aten.add, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_sum_32.run(buf1271, buf1269, arg180_1, buf1213, buf1239, buf1243, 2048, 2048, stream=stream0)
        buf1272 = buf1214; del buf1214  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_1534, pow_193, mean_192], Original ATen: [prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_mean_pow_25.run(buf1271, buf1272, 1, 2048, stream=stream0)
        buf1273 = buf1242; del buf1242  # reuse
        # Topologically Sorted Source Nodes: [mul_1156, sum_433], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_26.run(arg181_1, buf1271, buf1272, arg182_1, buf1273, 12288, 2048, stream=stream0)
        buf1274 = reinterpret_tensor(buf1269, (1, 2048), (2048, 1), 0); del buf1269  # reuse
        # Topologically Sorted Source Nodes: [mul_1159, sum_434], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf1273, arg183_1, buf1274, 2048, 6144, stream=stream0)
        buf1275 = buf1272; del buf1272  # reuse
        # Topologically Sorted Source Nodes: [add_437, convert_element_type_1544, pow_194, mean_193], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_27.run(buf1271, buf1274, buf1275, 1, 2048, stream=stream0)
        buf1279 = buf1249; del buf1249  # reuse
        # Topologically Sorted Source Nodes: [mul_1165, sum_436, mul_1168, sum_437, index_put_97], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_39.run(arg184_1, buf1271, buf1274, buf1275, arg187_1, arg189_1, arg1_1, buf1279, arg191_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_1554, pow_196, mean_195, mul_1171, cat_97, mul_1172, add_442, index_put_96], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_36.run(buf1279, arg188_1, arg1_1, buf667, arg190_1, 8, 128, stream=stream0)
        buf1276 = buf1243; del buf1243  # reuse
        # Topologically Sorted Source Nodes: [mul_1162, sum_435], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_28.run(arg184_1, buf1271, buf1274, buf1275, arg185_1, buf1276, 2048, 2048, stream=stream0)
        buf1287 = reinterpret_tensor(buf1239, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf1239  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_1549, pow_195, mean_194, mul_1169, cat_96, mul_1170, add_441, convert_element_type_1559, mul_1173], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf1276, arg186_1, buf667, buf1287, 16, 128, stream=stream0)
        buf1288 = buf481; del buf481  # reuse
        # Topologically Sorted Source Nodes: [mul_1175, sum_438], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf1287, arg190_1, buf1288, 15360, 128, stream=stream0)
        buf1297 = buf1267; del buf1267  # reuse
        # Topologically Sorted Source Nodes: [full_default_207, full_default_205, where_96, add_443, eq_48, logical_not_96, any_49, logical_not_97, full_default_209, , sub, exp, div_48, where_97], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_37.run(buf1288, arg1_1, buf1297, 16, 960, stream=stream0)
        buf1298 = buf488; del buf488  # reuse
        # Topologically Sorted Source Nodes: [mul_1176, sum_440], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf1297, arg191_1, buf1298, 16384, 120, stream=stream0)
        buf1299 = reinterpret_tensor(buf1287, (16, 1, 128), (128, 128, 1), 0); del buf1287  # reuse
        # Topologically Sorted Source Nodes: [mul_1176, sum_440], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf1298, buf1299, 2048, 8, stream=stream0)
        buf1300 = buf1276; del buf1276  # reuse
        # Topologically Sorted Source Nodes: [mul_1177, sum_441], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_17.run(buf1299, arg192_1, buf1300, 2048, 2048, stream=stream0)
        buf1302 = reinterpret_tensor(buf1299, (1, 2048), (2048, 1), 0); del buf1299  # reuse
        # Topologically Sorted Source Nodes: [add_437, add_444, convert_element_type_1566, pow_197, mean_196, convert_element_type_1568], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_30.run(buf1271, buf1274, buf1300, arg193_1, buf1302, 1, 2048, stream=stream0)
        buf1303 = buf1273; del buf1273  # reuse
        # Topologically Sorted Source Nodes: [mul_1180, sum_442], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_19.run(buf1302, arg194_1, buf1303, 12288, 2048, stream=stream0)
        buf1304 = buf1302; del buf1302  # reuse
        # Topologically Sorted Source Nodes: [mul_1183, sum_443], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf1303, arg195_1, buf1304, 2048, 6144, stream=stream0)
        buf1306 = buf1245; del buf1245  # reuse
        # Topologically Sorted Source Nodes: [add_437, add_444, add_446, convert_element_type_1576, pow_198, mean_197, add_447, rsqrt_197, mul_1184, convert_element_type_1577, mul_1185], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_mul_pow_rsqrt_31.run(buf1271, buf1274, buf1300, buf1304, arg196_1, buf1306, 1, 2048, stream=stream0)
        buf1310 = buf1279; del buf1279  # reuse
        # Topologically Sorted Source Nodes: [mul_1189, sum_445, mul_1192, sum_446, index_put_99], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_38.run(buf1306, arg199_1, arg201_1, arg1_1, buf1310, arg203_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_1586, pow_200, mean_199, mul_1195, cat_99, mul_1196, add_451, index_put_98], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_36.run(buf1310, arg200_1, arg1_1, buf667, arg202_1, 8, 128, stream=stream0)
        buf1307 = buf1213; del buf1213  # reuse
        # Topologically Sorted Source Nodes: [mul_1186, sum_444], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_22.run(buf1306, arg197_1, buf1307, 2048, 2048, stream=stream0)
        buf1318 = reinterpret_tensor(buf1246, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf1246  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_1581, pow_199, mean_198, mul_1193, cat_98, mul_1194, add_450, convert_element_type_1591, mul_1197], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf1307, arg198_1, buf667, buf1318, 16, 128, stream=stream0)
        buf1319 = buf505; del buf505  # reuse
        # Topologically Sorted Source Nodes: [mul_1199, sum_447], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf1318, arg202_1, buf1319, 15360, 128, stream=stream0)
        buf1328 = buf1297; del buf1297  # reuse
        # Topologically Sorted Source Nodes: [full_default_213, full_default_211, where_98, add_452, eq_49, logical_not_98, any_50, logical_not_99, full_default_215, , sub, exp, div_49, where_99], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_37.run(buf1319, arg1_1, buf1328, 16, 960, stream=stream0)
        buf1329 = buf512; del buf512  # reuse
        # Topologically Sorted Source Nodes: [mul_1200, sum_449], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf1328, arg203_1, buf1329, 16384, 120, stream=stream0)
        buf1330 = reinterpret_tensor(buf1318, (16, 1, 128), (128, 128, 1), 0); del buf1318  # reuse
        # Topologically Sorted Source Nodes: [mul_1200, sum_449], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf1329, buf1330, 2048, 8, stream=stream0)
        buf1332 = buf1271; del buf1271  # reuse
        # Topologically Sorted Source Nodes: [add_437, add_444, add_446, mul_1201, sum_450, add_453], Original ATen: [aten.add, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_sum_32.run(buf1332, buf1330, arg204_1, buf1274, buf1300, buf1304, 2048, 2048, stream=stream0)
        buf1333 = buf1275; del buf1275  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_1598, pow_201, mean_200], Original ATen: [prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_mean_pow_25.run(buf1332, buf1333, 1, 2048, stream=stream0)
        buf1334 = buf1303; del buf1303  # reuse
        # Topologically Sorted Source Nodes: [mul_1204, sum_451], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_26.run(arg205_1, buf1332, buf1333, arg206_1, buf1334, 12288, 2048, stream=stream0)
        buf1335 = reinterpret_tensor(buf1330, (1, 2048), (2048, 1), 0); del buf1330  # reuse
        # Topologically Sorted Source Nodes: [mul_1207, sum_452], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf1334, arg207_1, buf1335, 2048, 6144, stream=stream0)
        buf1336 = buf1333; del buf1333  # reuse
        # Topologically Sorted Source Nodes: [add_455, convert_element_type_1608, pow_202, mean_201], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_27.run(buf1332, buf1335, buf1336, 1, 2048, stream=stream0)
        buf1340 = buf1310; del buf1310  # reuse
        # Topologically Sorted Source Nodes: [mul_1213, sum_454, mul_1216, sum_455, index_put_101], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_39.run(arg208_1, buf1332, buf1335, buf1336, arg211_1, arg213_1, arg1_1, buf1340, arg215_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_1618, pow_204, mean_203, mul_1219, cat_101, mul_1220, add_460, index_put_100], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_36.run(buf1340, arg212_1, arg1_1, buf667, arg214_1, 8, 128, stream=stream0)
        buf1337 = buf1304; del buf1304  # reuse
        # Topologically Sorted Source Nodes: [mul_1210, sum_453], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_28.run(arg208_1, buf1332, buf1335, buf1336, arg209_1, buf1337, 2048, 2048, stream=stream0)
        buf1348 = reinterpret_tensor(buf1300, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf1300  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_1613, pow_203, mean_202, mul_1217, cat_100, mul_1218, add_459, convert_element_type_1623, mul_1221], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf1337, arg210_1, buf667, buf1348, 16, 128, stream=stream0)
        buf1349 = buf528; del buf528  # reuse
        # Topologically Sorted Source Nodes: [mul_1223, sum_456], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf1348, arg214_1, buf1349, 15360, 128, stream=stream0)
        buf1358 = buf1328; del buf1328  # reuse
        # Topologically Sorted Source Nodes: [full_default_219, full_default_217, where_100, add_461, eq_50, logical_not_100, any_51, logical_not_101, full_default_221, , sub, exp, div_50, where_101], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_37.run(buf1349, arg1_1, buf1358, 16, 960, stream=stream0)
        buf1359 = buf535; del buf535  # reuse
        # Topologically Sorted Source Nodes: [mul_1224, sum_458], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf1358, arg215_1, buf1359, 16384, 120, stream=stream0)
        buf1360 = reinterpret_tensor(buf1348, (16, 1, 128), (128, 128, 1), 0); del buf1348  # reuse
        # Topologically Sorted Source Nodes: [mul_1224, sum_458], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf1359, buf1360, 2048, 8, stream=stream0)
        buf1361 = buf1337; del buf1337  # reuse
        # Topologically Sorted Source Nodes: [mul_1225, sum_459], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_17.run(buf1360, arg216_1, buf1361, 2048, 2048, stream=stream0)
        buf1363 = reinterpret_tensor(buf1360, (1, 2048), (2048, 1), 0); del buf1360  # reuse
        # Topologically Sorted Source Nodes: [add_455, add_462, convert_element_type_1630, pow_205, mean_204, convert_element_type_1632], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_30.run(buf1332, buf1335, buf1361, arg217_1, buf1363, 1, 2048, stream=stream0)
        buf1364 = buf1334; del buf1334  # reuse
        # Topologically Sorted Source Nodes: [mul_1228, sum_460], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_19.run(buf1363, arg218_1, buf1364, 12288, 2048, stream=stream0)
        buf1365 = buf1363; del buf1363  # reuse
        # Topologically Sorted Source Nodes: [mul_1231, sum_461], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf1364, arg219_1, buf1365, 2048, 6144, stream=stream0)
        buf1367 = buf1306; del buf1306  # reuse
        # Topologically Sorted Source Nodes: [add_455, add_462, add_464, convert_element_type_1640, pow_206, mean_205, add_465, rsqrt_205, mul_1232, convert_element_type_1641, mul_1233], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_mul_pow_rsqrt_31.run(buf1332, buf1335, buf1361, buf1365, arg220_1, buf1367, 1, 2048, stream=stream0)
        buf1371 = buf1340; del buf1340  # reuse
        # Topologically Sorted Source Nodes: [mul_1237, sum_463, mul_1240, sum_464, index_put_103], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_38.run(buf1367, arg223_1, arg225_1, arg1_1, buf1371, arg227_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_1650, pow_208, mean_207, mul_1243, cat_103, mul_1244, add_469, index_put_102], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_36.run(buf1371, arg224_1, arg1_1, buf667, arg226_1, 8, 128, stream=stream0)
        buf1368 = buf1274; del buf1274  # reuse
        # Topologically Sorted Source Nodes: [mul_1234, sum_462], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_22.run(buf1367, arg221_1, buf1368, 2048, 2048, stream=stream0)
        buf1379 = reinterpret_tensor(buf1307, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf1307  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_1645, pow_207, mean_206, mul_1241, cat_102, mul_1242, add_468, convert_element_type_1655, mul_1245], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf1368, arg222_1, buf667, buf1379, 16, 128, stream=stream0)
        buf1380 = buf552; del buf552  # reuse
        # Topologically Sorted Source Nodes: [mul_1247, sum_465], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf1379, arg226_1, buf1380, 15360, 128, stream=stream0)
        buf1389 = buf1358; del buf1358  # reuse
        # Topologically Sorted Source Nodes: [full_default_225, full_default_223, where_102, add_470, eq_51, logical_not_102, any_52, logical_not_103, full_default_227, , sub, exp, div_51, where_103], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_37.run(buf1380, arg1_1, buf1389, 16, 960, stream=stream0)
        buf1390 = buf559; del buf559  # reuse
        # Topologically Sorted Source Nodes: [mul_1248, sum_467], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf1389, arg227_1, buf1390, 16384, 120, stream=stream0)
        buf1391 = reinterpret_tensor(buf1379, (16, 1, 128), (128, 128, 1), 0); del buf1379  # reuse
        # Topologically Sorted Source Nodes: [mul_1248, sum_467], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf1390, buf1391, 2048, 8, stream=stream0)
        buf1393 = buf1332; del buf1332  # reuse
        # Topologically Sorted Source Nodes: [add_455, add_462, add_464, mul_1249, sum_468, add_471], Original ATen: [aten.add, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_sum_32.run(buf1393, buf1391, arg228_1, buf1335, buf1361, buf1365, 2048, 2048, stream=stream0)
        buf1394 = buf1336; del buf1336  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_1662, pow_209, mean_208], Original ATen: [prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_mean_pow_25.run(buf1393, buf1394, 1, 2048, stream=stream0)
        buf1395 = buf1364; del buf1364  # reuse
        # Topologically Sorted Source Nodes: [mul_1252, sum_469], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_26.run(arg229_1, buf1393, buf1394, arg230_1, buf1395, 12288, 2048, stream=stream0)
        buf1396 = reinterpret_tensor(buf1391, (1, 2048), (2048, 1), 0); del buf1391  # reuse
        # Topologically Sorted Source Nodes: [mul_1255, sum_470], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf1395, arg231_1, buf1396, 2048, 6144, stream=stream0)
        buf1397 = buf1394; del buf1394  # reuse
        # Topologically Sorted Source Nodes: [add_473, convert_element_type_1672, pow_210, mean_209], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_27.run(buf1393, buf1396, buf1397, 1, 2048, stream=stream0)
        buf1401 = buf1371; del buf1371  # reuse
        # Topologically Sorted Source Nodes: [mul_1261, sum_472, mul_1264, sum_473, index_put_105], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_39.run(arg232_1, buf1393, buf1396, buf1397, arg235_1, arg237_1, arg1_1, buf1401, arg239_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_1682, pow_212, mean_211, mul_1267, cat_105, mul_1268, add_478, index_put_104], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_36.run(buf1401, arg236_1, arg1_1, buf667, arg238_1, 8, 128, stream=stream0)
        buf1398 = buf1365; del buf1365  # reuse
        # Topologically Sorted Source Nodes: [mul_1258, sum_471], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_28.run(arg232_1, buf1393, buf1396, buf1397, arg233_1, buf1398, 2048, 2048, stream=stream0)
        buf1409 = reinterpret_tensor(buf1361, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf1361  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_1677, pow_211, mean_210, mul_1265, cat_104, mul_1266, add_477, convert_element_type_1687, mul_1269], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf1398, arg234_1, buf667, buf1409, 16, 128, stream=stream0)
        buf1410 = buf575; del buf575  # reuse
        # Topologically Sorted Source Nodes: [mul_1271, sum_474], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf1409, arg238_1, buf1410, 15360, 128, stream=stream0)
        buf1419 = buf1389; del buf1389  # reuse
        # Topologically Sorted Source Nodes: [full_default_231, full_default_229, where_104, add_479, eq_52, logical_not_104, any_53, logical_not_105, full_default_233, , sub, exp, div_52, where_105], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_37.run(buf1410, arg1_1, buf1419, 16, 960, stream=stream0)
        buf1420 = buf582; del buf582  # reuse
        # Topologically Sorted Source Nodes: [mul_1272, sum_476], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf1419, arg239_1, buf1420, 16384, 120, stream=stream0)
        buf1421 = reinterpret_tensor(buf1409, (16, 1, 128), (128, 128, 1), 0); del buf1409  # reuse
        # Topologically Sorted Source Nodes: [mul_1272, sum_476], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf1420, buf1421, 2048, 8, stream=stream0)
        buf1422 = buf1398; del buf1398  # reuse
        # Topologically Sorted Source Nodes: [mul_1273, sum_477], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_17.run(buf1421, arg240_1, buf1422, 2048, 2048, stream=stream0)
        buf1424 = reinterpret_tensor(buf1421, (1, 2048), (2048, 1), 0); del buf1421  # reuse
        # Topologically Sorted Source Nodes: [add_473, add_480, convert_element_type_1694, pow_213, mean_212, convert_element_type_1696], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_30.run(buf1393, buf1396, buf1422, arg241_1, buf1424, 1, 2048, stream=stream0)
        buf1425 = buf1395; del buf1395  # reuse
        # Topologically Sorted Source Nodes: [mul_1276, sum_478], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_19.run(buf1424, arg242_1, buf1425, 12288, 2048, stream=stream0)
        buf1426 = buf1424; del buf1424  # reuse
        # Topologically Sorted Source Nodes: [mul_1279, sum_479], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf1425, arg243_1, buf1426, 2048, 6144, stream=stream0)
        buf1428 = buf1367; del buf1367  # reuse
        # Topologically Sorted Source Nodes: [add_473, add_480, add_482, convert_element_type_1704, pow_214, mean_213, add_483, rsqrt_213, mul_1280, convert_element_type_1705, mul_1281], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_mul_pow_rsqrt_31.run(buf1393, buf1396, buf1422, buf1426, arg244_1, buf1428, 1, 2048, stream=stream0)
        buf1432 = buf1401; del buf1401  # reuse
        # Topologically Sorted Source Nodes: [mul_1285, sum_481, mul_1288, sum_482, index_put_107], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_38.run(buf1428, arg247_1, arg249_1, arg1_1, buf1432, arg251_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_1714, pow_216, mean_215, mul_1291, cat_107, mul_1292, add_487, index_put_106], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_36.run(buf1432, arg248_1, arg1_1, buf667, arg250_1, 8, 128, stream=stream0)
        buf1429 = buf1335; del buf1335  # reuse
        # Topologically Sorted Source Nodes: [mul_1282, sum_480], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_22.run(buf1428, arg245_1, buf1429, 2048, 2048, stream=stream0)
        buf1440 = reinterpret_tensor(buf1368, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf1368  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_1709, pow_215, mean_214, mul_1289, cat_106, mul_1290, add_486, convert_element_type_1719, mul_1293], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf1429, arg246_1, buf667, buf1440, 16, 128, stream=stream0)
        buf1441 = buf599; del buf599  # reuse
        # Topologically Sorted Source Nodes: [mul_1295, sum_483], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf1440, arg250_1, buf1441, 15360, 128, stream=stream0)
        buf1450 = buf1419; del buf1419  # reuse
        # Topologically Sorted Source Nodes: [full_default_237, full_default_235, where_106, add_488, eq_53, logical_not_106, any_54, logical_not_107, full_default_239, , sub, exp, div_53, where_107], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_37.run(buf1441, arg1_1, buf1450, 16, 960, stream=stream0)
        buf1451 = buf606; del buf606  # reuse
        # Topologically Sorted Source Nodes: [mul_1296, sum_485], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf1450, arg251_1, buf1451, 16384, 120, stream=stream0)
        buf1452 = reinterpret_tensor(buf1440, (16, 1, 128), (128, 128, 1), 0); del buf1440  # reuse
        # Topologically Sorted Source Nodes: [mul_1296, sum_485], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf1451, buf1452, 2048, 8, stream=stream0)
        buf1454 = buf1393; del buf1393  # reuse
        # Topologically Sorted Source Nodes: [add_473, add_480, add_482, mul_1297, sum_486, add_489], Original ATen: [aten.add, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_sum_32.run(buf1454, buf1452, arg252_1, buf1396, buf1422, buf1426, 2048, 2048, stream=stream0)
        buf1455 = buf1397; del buf1397  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_1726, pow_217, mean_216], Original ATen: [prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_mean_pow_25.run(buf1454, buf1455, 1, 2048, stream=stream0)
        buf1456 = buf1425; del buf1425  # reuse
        # Topologically Sorted Source Nodes: [mul_1300, sum_487], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_26.run(arg253_1, buf1454, buf1455, arg254_1, buf1456, 12288, 2048, stream=stream0)
        buf1457 = reinterpret_tensor(buf1452, (1, 2048), (2048, 1), 0); del buf1452  # reuse
        # Topologically Sorted Source Nodes: [mul_1303, sum_488], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf1456, arg255_1, buf1457, 2048, 6144, stream=stream0)
        buf1458 = buf1455; del buf1455  # reuse
        # Topologically Sorted Source Nodes: [add_491, convert_element_type_1736, pow_218, mean_217], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_27.run(buf1454, buf1457, buf1458, 1, 2048, stream=stream0)
        buf1462 = buf1432; del buf1432  # reuse
        # Topologically Sorted Source Nodes: [mul_1309, sum_490, mul_1312, sum_491, index_put_109], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_39.run(arg256_1, buf1454, buf1457, buf1458, arg259_1, arg261_1, arg1_1, buf1462, arg263_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_1746, pow_220, mean_219, mul_1315, cat_109, mul_1316, add_496, index_put_108], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_36.run(buf1462, arg260_1, arg1_1, buf667, arg262_1, 8, 128, stream=stream0)
        buf1459 = buf1426; del buf1426  # reuse
        # Topologically Sorted Source Nodes: [mul_1306, sum_489], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_28.run(arg256_1, buf1454, buf1457, buf1458, arg257_1, buf1459, 2048, 2048, stream=stream0)
        buf1470 = reinterpret_tensor(buf1422, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf1422  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_1741, pow_219, mean_218, mul_1313, cat_108, mul_1314, add_495, convert_element_type_1751, mul_1317], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf1459, arg258_1, buf667, buf1470, 16, 128, stream=stream0)
        buf1471 = buf622; del buf622  # reuse
        # Topologically Sorted Source Nodes: [mul_1319, sum_492], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf1470, arg262_1, buf1471, 15360, 128, stream=stream0)
        buf1480 = buf1450; del buf1450  # reuse
        # Topologically Sorted Source Nodes: [full_default_243, full_default_241, where_108, add_497, eq_54, logical_not_108, any_55, logical_not_109, full_default_245, , sub, exp, div_54, where_109], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_37.run(buf1471, arg1_1, buf1480, 16, 960, stream=stream0)
        buf1481 = buf629; del buf629  # reuse
        # Topologically Sorted Source Nodes: [mul_1320, sum_494], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf1480, arg263_1, buf1481, 16384, 120, stream=stream0)
        buf1482 = reinterpret_tensor(buf1470, (16, 1, 128), (128, 128, 1), 0); del buf1470  # reuse
        # Topologically Sorted Source Nodes: [mul_1320, sum_494], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf1481, buf1482, 2048, 8, stream=stream0)
        buf1483 = buf1459; del buf1459  # reuse
        # Topologically Sorted Source Nodes: [mul_1321, sum_495], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_17.run(buf1482, arg264_1, buf1483, 2048, 2048, stream=stream0)
        buf1485 = reinterpret_tensor(buf1482, (1, 2048), (2048, 1), 0); del buf1482  # reuse
        # Topologically Sorted Source Nodes: [add_491, add_498, convert_element_type_1758, pow_221, mean_220, convert_element_type_1760], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_30.run(buf1454, buf1457, buf1483, arg265_1, buf1485, 1, 2048, stream=stream0)
        buf1486 = buf1456; del buf1456  # reuse
        # Topologically Sorted Source Nodes: [mul_1324, sum_496], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_19.run(buf1485, arg266_1, buf1486, 12288, 2048, stream=stream0)
        buf1487 = buf1485; del buf1485  # reuse
        # Topologically Sorted Source Nodes: [mul_1327, sum_497], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf1486, arg267_1, buf1487, 2048, 6144, stream=stream0)
        buf1489 = buf1428; del buf1428  # reuse
        # Topologically Sorted Source Nodes: [add_491, add_498, add_500, convert_element_type_1768, pow_222, mean_221, add_501, rsqrt_221, mul_1328, convert_element_type_1769, mul_1329], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_mul_pow_rsqrt_31.run(buf1454, buf1457, buf1483, buf1487, arg268_1, buf1489, 1, 2048, stream=stream0)
        buf1493 = buf1462; del buf1462  # reuse
        # Topologically Sorted Source Nodes: [mul_1333, sum_499, mul_1336, sum_500, index_put_111], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_38.run(buf1489, arg271_1, arg273_1, arg1_1, buf1493, arg275_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_1778, pow_224, mean_223, mul_1339, cat_111, mul_1340, add_505, index_put_110], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_36.run(buf1493, arg272_1, arg1_1, buf667, arg274_1, 8, 128, stream=stream0)
        buf1490 = buf1396; del buf1396  # reuse
        # Topologically Sorted Source Nodes: [mul_1330, sum_498], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_22.run(buf1489, arg269_1, buf1490, 2048, 2048, stream=stream0)
        buf1501 = reinterpret_tensor(buf1429, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf1429  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_1773, pow_223, mean_222, mul_1337, cat_110, mul_1338, add_504, convert_element_type_1783, mul_1341], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf1490, arg270_1, buf667, buf1501, 16, 128, stream=stream0)
        del buf667
        buf1502 = buf646; del buf646  # reuse
        # Topologically Sorted Source Nodes: [mul_1343, sum_501], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf1501, arg274_1, buf1502, 15360, 128, stream=stream0)
        buf1511 = buf1480; del buf1480  # reuse
        # Topologically Sorted Source Nodes: [full_default_249, full_default_247, where_110, add_506, eq_55, logical_not_110, any_56, logical_not_111, full_default_251, , sub, exp, div_55, where_111], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_37.run(buf1502, arg1_1, buf1511, 16, 960, stream=stream0)
        buf1512 = buf653; del buf653  # reuse
        # Topologically Sorted Source Nodes: [mul_1344, sum_503], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf1511, arg275_1, buf1512, 16384, 120, stream=stream0)
        buf1513 = reinterpret_tensor(buf1501, (16, 1, 128), (128, 128, 1), 0); del buf1501  # reuse
        # Topologically Sorted Source Nodes: [mul_1344, sum_503], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf1512, buf1513, 2048, 8, stream=stream0)
        buf1515 = buf1454; del buf1454  # reuse
        # Topologically Sorted Source Nodes: [add_491, add_498, add_500, mul_1345, sum_504, add_507], Original ATen: [aten.add, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_sum_32.run(buf1515, buf1513, arg276_1, buf1457, buf1483, buf1487, 2048, 2048, stream=stream0)
        buf1516 = buf1458; del buf1458  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_1790, pow_225, mean_224], Original ATen: [prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_mean_pow_25.run(buf1515, buf1516, 1, 2048, stream=stream0)
        buf1517 = buf1486; del buf1486  # reuse
        # Topologically Sorted Source Nodes: [mul_1348, sum_505], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_26.run(arg277_1, buf1515, buf1516, arg278_1, buf1517, 12288, 2048, stream=stream0)
        buf1518 = reinterpret_tensor(buf1513, (1, 2048), (2048, 1), 0); del buf1513  # reuse
        # Topologically Sorted Source Nodes: [mul_1351, sum_506], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf1517, arg279_1, buf1518, 2048, 6144, stream=stream0)
        buf1519 = buf1516; del buf1516  # reuse
        # Topologically Sorted Source Nodes: [add_509, convert_element_type_1800, pow_226, mean_225], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_27.run(buf1515, buf1518, buf1519, 1, 2048, stream=stream0)
        buf1520 = buf661; del buf661  # reuse
        # Topologically Sorted Source Nodes: [logits_1], Original ATen: [aten.mm]
        stream0 = get_raw_stream(0)
        triton_red_fused_mm_33.run(arg280_1, buf1515, buf1518, buf1519, arg282_1, buf1520, 151936, 2048, stream=stream0)
        buf1521 = buf662; del buf662  # reuse
        buf3241 = reinterpret_tensor(buf3244, (1, 1), (4, 1), 1)  # alias
        # Topologically Sorted Source Nodes: [logits_1, argmax_1, cat_224], Original ATen: [aten.mm, aten.argmax, aten.cat]
        stream0 = get_raw_stream(0)
        triton_red_fused_argmax_cat_mm_40.run(buf1520, buf1521, buf3241, 1, 151936, stream=stream0)
        buf1522 = buf1519; del buf1519  # reuse
        # Topologically Sorted Source Nodes: [embedding_2, convert_element_type_1805, pow_227, mean_226], Original ATen: [aten.embedding, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_embedding_mean_pow_0.run(buf1521, arg282_1, buf1522, 1, 2048, stream=stream0)
        buf1528 = buf1493; del buf1493  # reuse
        # Topologically Sorted Source Nodes: [mul_1362, sum_509, mul_1365, sum_510, index_put_113], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_41.run(arg284_1, buf1521, arg282_1, buf1522, arg287_1, arg289_1, arg1_1, buf1528, arg291_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_1815, pow_229, mean_228, mul_1368, cat_113, mul_1369, add_518, index_put_112], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_42.run(buf1528, arg288_1, arg1_1, buf1526, arg290_1, 8, 128, stream=stream0)
        buf1523 = buf1518; del buf1518  # reuse
        # Topologically Sorted Source Nodes: [mul_1359, sum_508], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_5.run(arg284_1, buf1521, arg282_1, buf1522, arg285_1, buf1523, 2048, 2048, stream=stream0)
        buf1536 = reinterpret_tensor(buf1487, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf1487  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_1810, pow_228, mean_227, mul_1366, cat_112, mul_1367, add_517, convert_element_type_1820, mul_1370], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf1523, arg286_1, buf1526, buf1536, 16, 128, stream=stream0)
        buf1537 = buf678; del buf678  # reuse
        # Topologically Sorted Source Nodes: [mul_1372, sum_511], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf1536, arg290_1, buf1537, 15360, 128, stream=stream0)
        buf1546 = buf1511; del buf1511  # reuse
        # Topologically Sorted Source Nodes: [full_default_255, full_default_253, where_112, add_519, eq_56, logical_not_112, any_57, logical_not_113, full_default_257, , sub, exp, div_56, where_113], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_43.run(buf1537, arg1_1, buf1546, 16, 960, stream=stream0)
        buf1547 = buf688; del buf688  # reuse
        # Topologically Sorted Source Nodes: [mul_1373, sum_513], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf1546, arg291_1, buf1547, 16384, 120, stream=stream0)
        buf1548 = reinterpret_tensor(buf1536, (16, 1, 128), (128, 128, 1), 0); del buf1536  # reuse
        # Topologically Sorted Source Nodes: [mul_1373, sum_513], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf1547, buf1548, 2048, 8, stream=stream0)
        buf1549 = buf1523; del buf1523  # reuse
        # Topologically Sorted Source Nodes: [mul_1374, sum_514], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_17.run(buf1548, arg292_1, buf1549, 2048, 2048, stream=stream0)
        buf1551 = reinterpret_tensor(buf1548, (1, 2048), (2048, 1), 0); del buf1548  # reuse
        # Topologically Sorted Source Nodes: [embedding_2, add_520, convert_element_type_1827, pow_230, mean_229, convert_element_type_1829], Original ATen: [aten.embedding, aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_embedding_mean_pow_18.run(buf1521, arg282_1, buf1549, arg293_1, buf1551, 1, 2048, stream=stream0)
        buf1552 = buf1517; del buf1517  # reuse
        # Topologically Sorted Source Nodes: [mul_1377, sum_515], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_19.run(buf1551, arg294_1, buf1552, 12288, 2048, stream=stream0)
        buf1553 = buf1551; del buf1551  # reuse
        # Topologically Sorted Source Nodes: [mul_1380, sum_516], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf1552, arg295_1, buf1553, 2048, 6144, stream=stream0)
        buf1555 = buf1515; del buf1515  # reuse
        # Topologically Sorted Source Nodes: [embedding_2, add_520, add_522, convert_element_type_1837, pow_231, mean_230, add_523, rsqrt_230, mul_1381, convert_element_type_1838, mul_1382], Original ATen: [aten.embedding, aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_embedding_mean_mul_pow_rsqrt_21.run(buf1521, arg282_1, buf1549, buf1553, arg296_1, buf1555, 1, 2048, stream=stream0)
        buf1559 = buf1528; del buf1528  # reuse
        # Topologically Sorted Source Nodes: [mul_1386, sum_518, mul_1389, sum_519, index_put_115], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_44.run(buf1555, arg299_1, arg301_1, arg1_1, buf1559, arg303_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_1847, pow_233, mean_232, mul_1392, cat_115, mul_1393, add_527, index_put_114], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_42.run(buf1559, arg300_1, arg1_1, buf1526, arg302_1, 8, 128, stream=stream0)
        buf1556 = buf1483; del buf1483  # reuse
        # Topologically Sorted Source Nodes: [mul_1383, sum_517], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_22.run(buf1555, arg297_1, buf1556, 2048, 2048, stream=stream0)
        buf1567 = reinterpret_tensor(buf1457, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf1457  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_1842, pow_232, mean_231, mul_1390, cat_114, mul_1391, add_526, convert_element_type_1852, mul_1394], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf1556, arg298_1, buf1526, buf1567, 16, 128, stream=stream0)
        buf1568 = buf709; del buf709  # reuse
        # Topologically Sorted Source Nodes: [mul_1396, sum_520], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf1567, arg302_1, buf1568, 15360, 128, stream=stream0)
        buf1577 = buf1546; del buf1546  # reuse
        # Topologically Sorted Source Nodes: [full_default_261, full_default_259, where_114, add_528, eq_57, logical_not_114, any_58, logical_not_115, full_default_263, , sub, exp, div_57, where_115], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_43.run(buf1568, arg1_1, buf1577, 16, 960, stream=stream0)
        buf1578 = buf719; del buf719  # reuse
        # Topologically Sorted Source Nodes: [mul_1397, sum_522], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf1577, arg303_1, buf1578, 16384, 120, stream=stream0)
        buf1579 = reinterpret_tensor(buf1567, (16, 1, 128), (128, 128, 1), 0); del buf1567  # reuse
        # Topologically Sorted Source Nodes: [mul_1397, sum_522], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf1578, buf1579, 2048, 8, stream=stream0)
        buf1581 = buf1555; del buf1555  # reuse
        # Topologically Sorted Source Nodes: [embedding_2, add_520, add_522, mul_1398, sum_523, add_529], Original ATen: [aten.embedding, aten.add, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_embedding_mul_sum_24.run(buf1579, arg304_1, buf1521, arg282_1, buf1549, buf1553, buf1581, 2048, 2048, stream=stream0)
        buf1582 = buf1522; del buf1522  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_1859, pow_234, mean_233], Original ATen: [prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_mean_pow_25.run(buf1581, buf1582, 1, 2048, stream=stream0)
        buf1583 = buf1552; del buf1552  # reuse
        # Topologically Sorted Source Nodes: [mul_1401, sum_524], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_26.run(arg305_1, buf1581, buf1582, arg306_1, buf1583, 12288, 2048, stream=stream0)
        buf1584 = reinterpret_tensor(buf1579, (1, 2048), (2048, 1), 0); del buf1579  # reuse
        # Topologically Sorted Source Nodes: [mul_1404, sum_525], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf1583, arg307_1, buf1584, 2048, 6144, stream=stream0)
        buf1585 = buf1582; del buf1582  # reuse
        # Topologically Sorted Source Nodes: [add_531, convert_element_type_1869, pow_235, mean_234], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_27.run(buf1581, buf1584, buf1585, 1, 2048, stream=stream0)
        buf1589 = buf1559; del buf1559  # reuse
        # Topologically Sorted Source Nodes: [mul_1410, sum_527, mul_1413, sum_528, index_put_117], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_45.run(arg308_1, buf1581, buf1584, buf1585, arg311_1, arg313_1, arg1_1, buf1589, arg315_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_1879, pow_237, mean_236, mul_1416, cat_117, mul_1417, add_536, index_put_116], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_42.run(buf1589, arg312_1, arg1_1, buf1526, arg314_1, 8, 128, stream=stream0)
        buf1586 = buf1553; del buf1553  # reuse
        # Topologically Sorted Source Nodes: [mul_1407, sum_526], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_28.run(arg308_1, buf1581, buf1584, buf1585, arg309_1, buf1586, 2048, 2048, stream=stream0)
        buf1597 = reinterpret_tensor(buf1549, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf1549  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_1874, pow_236, mean_235, mul_1414, cat_116, mul_1415, add_535, convert_element_type_1884, mul_1418], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf1586, arg310_1, buf1526, buf1597, 16, 128, stream=stream0)
        buf1598 = buf739; del buf739  # reuse
        # Topologically Sorted Source Nodes: [mul_1420, sum_529], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf1597, arg314_1, buf1598, 15360, 128, stream=stream0)
        buf1607 = buf1577; del buf1577  # reuse
        # Topologically Sorted Source Nodes: [full_default_267, full_default_265, where_116, add_537, eq_58, logical_not_116, any_59, logical_not_117, full_default_269, , sub, exp, div_58, where_117], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_43.run(buf1598, arg1_1, buf1607, 16, 960, stream=stream0)
        buf1608 = buf749; del buf749  # reuse
        # Topologically Sorted Source Nodes: [mul_1421, sum_531], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf1607, arg315_1, buf1608, 16384, 120, stream=stream0)
        buf1609 = reinterpret_tensor(buf1597, (16, 1, 128), (128, 128, 1), 0); del buf1597  # reuse
        # Topologically Sorted Source Nodes: [mul_1421, sum_531], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf1608, buf1609, 2048, 8, stream=stream0)
        buf1610 = buf1586; del buf1586  # reuse
        # Topologically Sorted Source Nodes: [mul_1422, sum_532], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_17.run(buf1609, arg316_1, buf1610, 2048, 2048, stream=stream0)
        buf1612 = reinterpret_tensor(buf1609, (1, 2048), (2048, 1), 0); del buf1609  # reuse
        # Topologically Sorted Source Nodes: [add_531, add_538, convert_element_type_1891, pow_238, mean_237, convert_element_type_1893], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_30.run(buf1581, buf1584, buf1610, arg317_1, buf1612, 1, 2048, stream=stream0)
        buf1613 = buf1583; del buf1583  # reuse
        # Topologically Sorted Source Nodes: [mul_1425, sum_533], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_19.run(buf1612, arg318_1, buf1613, 12288, 2048, stream=stream0)
        buf1614 = buf1612; del buf1612  # reuse
        # Topologically Sorted Source Nodes: [mul_1428, sum_534], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf1613, arg319_1, buf1614, 2048, 6144, stream=stream0)
        buf1616 = buf1489; del buf1489  # reuse
        # Topologically Sorted Source Nodes: [add_531, add_538, add_540, convert_element_type_1901, pow_239, mean_238, add_541, rsqrt_238, mul_1429, convert_element_type_1902, mul_1430], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_mul_pow_rsqrt_31.run(buf1581, buf1584, buf1610, buf1614, arg320_1, buf1616, 1, 2048, stream=stream0)
        buf1620 = buf1589; del buf1589  # reuse
        # Topologically Sorted Source Nodes: [mul_1434, sum_536, mul_1437, sum_537, index_put_119], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_44.run(buf1616, arg323_1, arg325_1, arg1_1, buf1620, arg327_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_1911, pow_241, mean_240, mul_1440, cat_119, mul_1441, add_545, index_put_118], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_42.run(buf1620, arg324_1, arg1_1, buf1526, arg326_1, 8, 128, stream=stream0)
        buf1617 = buf1556; del buf1556  # reuse
        # Topologically Sorted Source Nodes: [mul_1431, sum_535], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_22.run(buf1616, arg321_1, buf1617, 2048, 2048, stream=stream0)
        buf1628 = reinterpret_tensor(buf1490, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf1490  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_1906, pow_240, mean_239, mul_1438, cat_118, mul_1439, add_544, convert_element_type_1916, mul_1442], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf1617, arg322_1, buf1526, buf1628, 16, 128, stream=stream0)
        buf1629 = buf770; del buf770  # reuse
        # Topologically Sorted Source Nodes: [mul_1444, sum_538], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf1628, arg326_1, buf1629, 15360, 128, stream=stream0)
        buf1638 = buf1607; del buf1607  # reuse
        # Topologically Sorted Source Nodes: [full_default_273, full_default_271, where_118, add_546, eq_59, logical_not_118, any_60, logical_not_119, full_default_275, , sub, exp, div_59, where_119], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_43.run(buf1629, arg1_1, buf1638, 16, 960, stream=stream0)
        buf1639 = buf780; del buf780  # reuse
        # Topologically Sorted Source Nodes: [mul_1445, sum_540], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf1638, arg327_1, buf1639, 16384, 120, stream=stream0)
        buf1640 = reinterpret_tensor(buf1628, (16, 1, 128), (128, 128, 1), 0); del buf1628  # reuse
        # Topologically Sorted Source Nodes: [mul_1445, sum_540], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf1639, buf1640, 2048, 8, stream=stream0)
        buf1642 = buf1581; del buf1581  # reuse
        # Topologically Sorted Source Nodes: [add_531, add_538, add_540, mul_1446, sum_541, add_547], Original ATen: [aten.add, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_sum_32.run(buf1642, buf1640, arg328_1, buf1584, buf1610, buf1614, 2048, 2048, stream=stream0)
        buf1643 = buf1585; del buf1585  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_1923, pow_242, mean_241], Original ATen: [prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_mean_pow_25.run(buf1642, buf1643, 1, 2048, stream=stream0)
        buf1644 = buf1613; del buf1613  # reuse
        # Topologically Sorted Source Nodes: [mul_1449, sum_542], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_26.run(arg329_1, buf1642, buf1643, arg330_1, buf1644, 12288, 2048, stream=stream0)
        buf1645 = reinterpret_tensor(buf1640, (1, 2048), (2048, 1), 0); del buf1640  # reuse
        # Topologically Sorted Source Nodes: [mul_1452, sum_543], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf1644, arg331_1, buf1645, 2048, 6144, stream=stream0)
        buf1646 = buf1643; del buf1643  # reuse
        # Topologically Sorted Source Nodes: [add_549, convert_element_type_1933, pow_243, mean_242], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_27.run(buf1642, buf1645, buf1646, 1, 2048, stream=stream0)
        buf1650 = buf1620; del buf1620  # reuse
        # Topologically Sorted Source Nodes: [mul_1458, sum_545, mul_1461, sum_546, index_put_121], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_45.run(arg332_1, buf1642, buf1645, buf1646, arg335_1, arg337_1, arg1_1, buf1650, arg339_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_1943, pow_245, mean_244, mul_1464, cat_121, mul_1465, add_554, index_put_120], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_42.run(buf1650, arg336_1, arg1_1, buf1526, arg338_1, 8, 128, stream=stream0)
        buf1647 = buf1614; del buf1614  # reuse
        # Topologically Sorted Source Nodes: [mul_1455, sum_544], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_28.run(arg332_1, buf1642, buf1645, buf1646, arg333_1, buf1647, 2048, 2048, stream=stream0)
        buf1658 = reinterpret_tensor(buf1610, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf1610  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_1938, pow_244, mean_243, mul_1462, cat_120, mul_1463, add_553, convert_element_type_1948, mul_1466], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf1647, arg334_1, buf1526, buf1658, 16, 128, stream=stream0)
        buf1659 = buf800; del buf800  # reuse
        # Topologically Sorted Source Nodes: [mul_1468, sum_547], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf1658, arg338_1, buf1659, 15360, 128, stream=stream0)
        buf1668 = buf1638; del buf1638  # reuse
        # Topologically Sorted Source Nodes: [full_default_279, full_default_277, where_120, add_555, eq_60, logical_not_120, any_61, logical_not_121, full_default_281, , sub, exp, div_60, where_121], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_43.run(buf1659, arg1_1, buf1668, 16, 960, stream=stream0)
        buf1669 = buf810; del buf810  # reuse
        # Topologically Sorted Source Nodes: [mul_1469, sum_549], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf1668, arg339_1, buf1669, 16384, 120, stream=stream0)
        buf1670 = reinterpret_tensor(buf1658, (16, 1, 128), (128, 128, 1), 0); del buf1658  # reuse
        # Topologically Sorted Source Nodes: [mul_1469, sum_549], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf1669, buf1670, 2048, 8, stream=stream0)
        buf1671 = buf1647; del buf1647  # reuse
        # Topologically Sorted Source Nodes: [mul_1470, sum_550], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_17.run(buf1670, arg340_1, buf1671, 2048, 2048, stream=stream0)
        buf1673 = reinterpret_tensor(buf1670, (1, 2048), (2048, 1), 0); del buf1670  # reuse
        # Topologically Sorted Source Nodes: [add_549, add_556, convert_element_type_1955, pow_246, mean_245, convert_element_type_1957], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_30.run(buf1642, buf1645, buf1671, arg341_1, buf1673, 1, 2048, stream=stream0)
        buf1674 = buf1644; del buf1644  # reuse
        # Topologically Sorted Source Nodes: [mul_1473, sum_551], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_19.run(buf1673, arg2_1, buf1674, 12288, 2048, stream=stream0)
        buf1675 = buf1673; del buf1673  # reuse
        # Topologically Sorted Source Nodes: [mul_1476, sum_552], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf1674, arg3_1, buf1675, 2048, 6144, stream=stream0)
        buf1677 = buf1616; del buf1616  # reuse
        # Topologically Sorted Source Nodes: [add_549, add_556, add_558, convert_element_type_1965, pow_247, mean_246, add_559, rsqrt_246, mul_1477, convert_element_type_1966, mul_1478], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_mul_pow_rsqrt_31.run(buf1642, buf1645, buf1671, buf1675, arg4_1, buf1677, 1, 2048, stream=stream0)
        buf1681 = buf1650; del buf1650  # reuse
        # Topologically Sorted Source Nodes: [mul_1482, sum_554, mul_1485, sum_555, index_put_123], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_44.run(buf1677, arg7_1, arg9_1, arg1_1, buf1681, arg11_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_1975, pow_249, mean_248, mul_1488, cat_123, mul_1489, add_563, index_put_122], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_42.run(buf1681, arg8_1, arg1_1, buf1526, arg10_1, 8, 128, stream=stream0)
        buf1678 = buf1584; del buf1584  # reuse
        # Topologically Sorted Source Nodes: [mul_1479, sum_553], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_22.run(buf1677, arg5_1, buf1678, 2048, 2048, stream=stream0)
        buf1689 = reinterpret_tensor(buf1617, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf1617  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_1970, pow_248, mean_247, mul_1486, cat_122, mul_1487, add_562, convert_element_type_1980, mul_1490], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf1678, arg6_1, buf1526, buf1689, 16, 128, stream=stream0)
        buf1690 = buf831; del buf831  # reuse
        # Topologically Sorted Source Nodes: [mul_1492, sum_556], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf1689, arg10_1, buf1690, 15360, 128, stream=stream0)
        buf1699 = buf1668; del buf1668  # reuse
        # Topologically Sorted Source Nodes: [full_default_285, full_default_283, where_122, add_564, eq_61, logical_not_122, any_62, logical_not_123, full_default_287, , sub, exp, div_61, where_123], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_43.run(buf1690, arg1_1, buf1699, 16, 960, stream=stream0)
        buf1700 = buf841; del buf841  # reuse
        # Topologically Sorted Source Nodes: [mul_1493, sum_558], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf1699, arg11_1, buf1700, 16384, 120, stream=stream0)
        buf1701 = reinterpret_tensor(buf1689, (16, 1, 128), (128, 128, 1), 0); del buf1689  # reuse
        # Topologically Sorted Source Nodes: [mul_1493, sum_558], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf1700, buf1701, 2048, 8, stream=stream0)
        buf1703 = buf1642; del buf1642  # reuse
        # Topologically Sorted Source Nodes: [add_549, add_556, add_558, mul_1494, sum_559, add_565], Original ATen: [aten.add, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_sum_32.run(buf1703, buf1701, arg12_1, buf1645, buf1671, buf1675, 2048, 2048, stream=stream0)
        buf1704 = buf1646; del buf1646  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_1987, pow_250, mean_249], Original ATen: [prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_mean_pow_25.run(buf1703, buf1704, 1, 2048, stream=stream0)
        buf1705 = buf1674; del buf1674  # reuse
        # Topologically Sorted Source Nodes: [mul_1497, sum_560], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_26.run(arg13_1, buf1703, buf1704, arg14_1, buf1705, 12288, 2048, stream=stream0)
        buf1706 = reinterpret_tensor(buf1701, (1, 2048), (2048, 1), 0); del buf1701  # reuse
        # Topologically Sorted Source Nodes: [mul_1500, sum_561], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf1705, arg15_1, buf1706, 2048, 6144, stream=stream0)
        buf1707 = buf1704; del buf1704  # reuse
        # Topologically Sorted Source Nodes: [add_567, convert_element_type_1997, pow_251, mean_250], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_27.run(buf1703, buf1706, buf1707, 1, 2048, stream=stream0)
        buf1711 = buf1681; del buf1681  # reuse
        # Topologically Sorted Source Nodes: [mul_1506, sum_563, mul_1509, sum_564, index_put_125], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_45.run(arg16_1, buf1703, buf1706, buf1707, arg19_1, arg21_1, arg1_1, buf1711, arg23_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_2007, pow_253, mean_252, mul_1512, cat_125, mul_1513, add_572, index_put_124], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_42.run(buf1711, arg20_1, arg1_1, buf1526, arg22_1, 8, 128, stream=stream0)
        buf1708 = buf1675; del buf1675  # reuse
        # Topologically Sorted Source Nodes: [mul_1503, sum_562], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_28.run(arg16_1, buf1703, buf1706, buf1707, arg17_1, buf1708, 2048, 2048, stream=stream0)
        buf1719 = reinterpret_tensor(buf1671, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf1671  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_2002, pow_252, mean_251, mul_1510, cat_124, mul_1511, add_571, convert_element_type_2012, mul_1514], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf1708, arg18_1, buf1526, buf1719, 16, 128, stream=stream0)
        buf1720 = buf861; del buf861  # reuse
        # Topologically Sorted Source Nodes: [mul_1516, sum_565], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf1719, arg22_1, buf1720, 15360, 128, stream=stream0)
        buf1729 = buf1699; del buf1699  # reuse
        # Topologically Sorted Source Nodes: [full_default_291, full_default_289, where_124, add_573, eq_62, logical_not_124, any_63, logical_not_125, full_default_293, , sub, exp, div_62, where_125], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_43.run(buf1720, arg1_1, buf1729, 16, 960, stream=stream0)
        buf1730 = buf871; del buf871  # reuse
        # Topologically Sorted Source Nodes: [mul_1517, sum_567], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf1729, arg23_1, buf1730, 16384, 120, stream=stream0)
        buf1731 = reinterpret_tensor(buf1719, (16, 1, 128), (128, 128, 1), 0); del buf1719  # reuse
        # Topologically Sorted Source Nodes: [mul_1517, sum_567], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf1730, buf1731, 2048, 8, stream=stream0)
        buf1732 = buf1708; del buf1708  # reuse
        # Topologically Sorted Source Nodes: [mul_1518, sum_568], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_17.run(buf1731, arg24_1, buf1732, 2048, 2048, stream=stream0)
        buf1734 = reinterpret_tensor(buf1731, (1, 2048), (2048, 1), 0); del buf1731  # reuse
        # Topologically Sorted Source Nodes: [add_567, add_574, convert_element_type_2019, pow_254, mean_253, convert_element_type_2021], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_30.run(buf1703, buf1706, buf1732, arg25_1, buf1734, 1, 2048, stream=stream0)
        buf1735 = buf1705; del buf1705  # reuse
        # Topologically Sorted Source Nodes: [mul_1521, sum_569], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_19.run(buf1734, arg26_1, buf1735, 12288, 2048, stream=stream0)
        buf1736 = buf1734; del buf1734  # reuse
        # Topologically Sorted Source Nodes: [mul_1524, sum_570], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf1735, arg27_1, buf1736, 2048, 6144, stream=stream0)
        buf1738 = buf1677; del buf1677  # reuse
        # Topologically Sorted Source Nodes: [add_567, add_574, add_576, convert_element_type_2029, pow_255, mean_254, add_577, rsqrt_254, mul_1525, convert_element_type_2030, mul_1526], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_mul_pow_rsqrt_31.run(buf1703, buf1706, buf1732, buf1736, arg28_1, buf1738, 1, 2048, stream=stream0)
        buf1742 = buf1711; del buf1711  # reuse
        # Topologically Sorted Source Nodes: [mul_1530, sum_572, mul_1533, sum_573, index_put_127], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_44.run(buf1738, arg31_1, arg33_1, arg1_1, buf1742, arg35_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_2039, pow_257, mean_256, mul_1536, cat_127, mul_1537, add_581, index_put_126], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_42.run(buf1742, arg32_1, arg1_1, buf1526, arg34_1, 8, 128, stream=stream0)
        buf1739 = buf1645; del buf1645  # reuse
        # Topologically Sorted Source Nodes: [mul_1527, sum_571], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_22.run(buf1738, arg29_1, buf1739, 2048, 2048, stream=stream0)
        buf1750 = reinterpret_tensor(buf1678, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf1678  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_2034, pow_256, mean_255, mul_1534, cat_126, mul_1535, add_580, convert_element_type_2044, mul_1538], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf1739, arg30_1, buf1526, buf1750, 16, 128, stream=stream0)
        buf1751 = buf892; del buf892  # reuse
        # Topologically Sorted Source Nodes: [mul_1540, sum_574], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf1750, arg34_1, buf1751, 15360, 128, stream=stream0)
        buf1760 = buf1729; del buf1729  # reuse
        # Topologically Sorted Source Nodes: [full_default_297, full_default_295, where_126, add_582, eq_63, logical_not_126, any_64, logical_not_127, full_default_299, , sub, exp, div_63, where_127], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_43.run(buf1751, arg1_1, buf1760, 16, 960, stream=stream0)
        buf1761 = buf902; del buf902  # reuse
        # Topologically Sorted Source Nodes: [mul_1541, sum_576], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf1760, arg35_1, buf1761, 16384, 120, stream=stream0)
        buf1762 = reinterpret_tensor(buf1750, (16, 1, 128), (128, 128, 1), 0); del buf1750  # reuse
        # Topologically Sorted Source Nodes: [mul_1541, sum_576], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf1761, buf1762, 2048, 8, stream=stream0)
        buf1764 = buf1703; del buf1703  # reuse
        # Topologically Sorted Source Nodes: [add_567, add_574, add_576, mul_1542, sum_577, add_583], Original ATen: [aten.add, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_sum_32.run(buf1764, buf1762, arg36_1, buf1706, buf1732, buf1736, 2048, 2048, stream=stream0)
        buf1765 = buf1707; del buf1707  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_2051, pow_258, mean_257], Original ATen: [prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_mean_pow_25.run(buf1764, buf1765, 1, 2048, stream=stream0)
        buf1766 = buf1735; del buf1735  # reuse
        # Topologically Sorted Source Nodes: [mul_1545, sum_578], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_26.run(arg37_1, buf1764, buf1765, arg38_1, buf1766, 12288, 2048, stream=stream0)
        buf1767 = reinterpret_tensor(buf1762, (1, 2048), (2048, 1), 0); del buf1762  # reuse
        # Topologically Sorted Source Nodes: [mul_1548, sum_579], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf1766, arg39_1, buf1767, 2048, 6144, stream=stream0)
        buf1768 = buf1765; del buf1765  # reuse
        # Topologically Sorted Source Nodes: [add_585, convert_element_type_2061, pow_259, mean_258], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_27.run(buf1764, buf1767, buf1768, 1, 2048, stream=stream0)
        buf1772 = buf1742; del buf1742  # reuse
        # Topologically Sorted Source Nodes: [mul_1554, sum_581, mul_1557, sum_582, index_put_129], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_45.run(arg40_1, buf1764, buf1767, buf1768, arg43_1, arg45_1, arg1_1, buf1772, arg47_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_2071, pow_261, mean_260, mul_1560, cat_129, mul_1561, add_590, index_put_128], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_42.run(buf1772, arg44_1, arg1_1, buf1526, arg46_1, 8, 128, stream=stream0)
        buf1769 = buf1736; del buf1736  # reuse
        # Topologically Sorted Source Nodes: [mul_1551, sum_580], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_28.run(arg40_1, buf1764, buf1767, buf1768, arg41_1, buf1769, 2048, 2048, stream=stream0)
        buf1780 = reinterpret_tensor(buf1732, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf1732  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_2066, pow_260, mean_259, mul_1558, cat_128, mul_1559, add_589, convert_element_type_2076, mul_1562], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf1769, arg42_1, buf1526, buf1780, 16, 128, stream=stream0)
        buf1781 = buf922; del buf922  # reuse
        # Topologically Sorted Source Nodes: [mul_1564, sum_583], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf1780, arg46_1, buf1781, 15360, 128, stream=stream0)
        buf1790 = buf1760; del buf1760  # reuse
        # Topologically Sorted Source Nodes: [full_default_303, full_default_301, where_128, add_591, eq_64, logical_not_128, any_65, logical_not_129, full_default_305, , sub, exp, div_64, where_129], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_43.run(buf1781, arg1_1, buf1790, 16, 960, stream=stream0)
        buf1791 = buf932; del buf932  # reuse
        # Topologically Sorted Source Nodes: [mul_1565, sum_585], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf1790, arg47_1, buf1791, 16384, 120, stream=stream0)
        buf1792 = reinterpret_tensor(buf1780, (16, 1, 128), (128, 128, 1), 0); del buf1780  # reuse
        # Topologically Sorted Source Nodes: [mul_1565, sum_585], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf1791, buf1792, 2048, 8, stream=stream0)
        buf1793 = buf1769; del buf1769  # reuse
        # Topologically Sorted Source Nodes: [mul_1566, sum_586], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_17.run(buf1792, arg48_1, buf1793, 2048, 2048, stream=stream0)
        buf1795 = reinterpret_tensor(buf1792, (1, 2048), (2048, 1), 0); del buf1792  # reuse
        # Topologically Sorted Source Nodes: [add_585, add_592, convert_element_type_2083, pow_262, mean_261, convert_element_type_2085], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_30.run(buf1764, buf1767, buf1793, arg49_1, buf1795, 1, 2048, stream=stream0)
        buf1796 = buf1766; del buf1766  # reuse
        # Topologically Sorted Source Nodes: [mul_1569, sum_587], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_19.run(buf1795, arg50_1, buf1796, 12288, 2048, stream=stream0)
        buf1797 = buf1795; del buf1795  # reuse
        # Topologically Sorted Source Nodes: [mul_1572, sum_588], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf1796, arg51_1, buf1797, 2048, 6144, stream=stream0)
        buf1799 = buf1738; del buf1738  # reuse
        # Topologically Sorted Source Nodes: [add_585, add_592, add_594, convert_element_type_2093, pow_263, mean_262, add_595, rsqrt_262, mul_1573, convert_element_type_2094, mul_1574], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_mul_pow_rsqrt_31.run(buf1764, buf1767, buf1793, buf1797, arg52_1, buf1799, 1, 2048, stream=stream0)
        buf1803 = buf1772; del buf1772  # reuse
        # Topologically Sorted Source Nodes: [mul_1578, sum_590, mul_1581, sum_591, index_put_131], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_44.run(buf1799, arg55_1, arg57_1, arg1_1, buf1803, arg59_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_2103, pow_265, mean_264, mul_1584, cat_131, mul_1585, add_599, index_put_130], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_42.run(buf1803, arg56_1, arg1_1, buf1526, arg58_1, 8, 128, stream=stream0)
        buf1800 = buf1706; del buf1706  # reuse
        # Topologically Sorted Source Nodes: [mul_1575, sum_589], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_22.run(buf1799, arg53_1, buf1800, 2048, 2048, stream=stream0)
        buf1811 = reinterpret_tensor(buf1739, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf1739  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_2098, pow_264, mean_263, mul_1582, cat_130, mul_1583, add_598, convert_element_type_2108, mul_1586], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf1800, arg54_1, buf1526, buf1811, 16, 128, stream=stream0)
        buf1812 = buf953; del buf953  # reuse
        # Topologically Sorted Source Nodes: [mul_1588, sum_592], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf1811, arg58_1, buf1812, 15360, 128, stream=stream0)
        buf1821 = buf1790; del buf1790  # reuse
        # Topologically Sorted Source Nodes: [full_default_309, full_default_307, where_130, add_600, eq_65, logical_not_130, any_66, logical_not_131, full_default_311, , sub, exp, div_65, where_131], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_43.run(buf1812, arg1_1, buf1821, 16, 960, stream=stream0)
        buf1822 = buf963; del buf963  # reuse
        # Topologically Sorted Source Nodes: [mul_1589, sum_594], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf1821, arg59_1, buf1822, 16384, 120, stream=stream0)
        buf1823 = reinterpret_tensor(buf1811, (16, 1, 128), (128, 128, 1), 0); del buf1811  # reuse
        # Topologically Sorted Source Nodes: [mul_1589, sum_594], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf1822, buf1823, 2048, 8, stream=stream0)
        buf1825 = buf1764; del buf1764  # reuse
        # Topologically Sorted Source Nodes: [add_585, add_592, add_594, mul_1590, sum_595, add_601], Original ATen: [aten.add, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_sum_32.run(buf1825, buf1823, arg60_1, buf1767, buf1793, buf1797, 2048, 2048, stream=stream0)
        buf1826 = buf1768; del buf1768  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_2115, pow_266, mean_265], Original ATen: [prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_mean_pow_25.run(buf1825, buf1826, 1, 2048, stream=stream0)
        buf1827 = buf1796; del buf1796  # reuse
        # Topologically Sorted Source Nodes: [mul_1593, sum_596], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_26.run(arg61_1, buf1825, buf1826, arg62_1, buf1827, 12288, 2048, stream=stream0)
        buf1828 = reinterpret_tensor(buf1823, (1, 2048), (2048, 1), 0); del buf1823  # reuse
        # Topologically Sorted Source Nodes: [mul_1596, sum_597], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf1827, arg63_1, buf1828, 2048, 6144, stream=stream0)
        buf1829 = buf1826; del buf1826  # reuse
        # Topologically Sorted Source Nodes: [add_603, convert_element_type_2125, pow_267, mean_266], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_27.run(buf1825, buf1828, buf1829, 1, 2048, stream=stream0)
        buf1833 = buf1803; del buf1803  # reuse
        # Topologically Sorted Source Nodes: [mul_1602, sum_599, mul_1605, sum_600, index_put_133], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_45.run(arg64_1, buf1825, buf1828, buf1829, arg67_1, arg69_1, arg1_1, buf1833, arg71_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_2135, pow_269, mean_268, mul_1608, cat_133, mul_1609, add_608, index_put_132], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_42.run(buf1833, arg68_1, arg1_1, buf1526, arg70_1, 8, 128, stream=stream0)
        buf1830 = buf1797; del buf1797  # reuse
        # Topologically Sorted Source Nodes: [mul_1599, sum_598], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_28.run(arg64_1, buf1825, buf1828, buf1829, arg65_1, buf1830, 2048, 2048, stream=stream0)
        buf1841 = reinterpret_tensor(buf1793, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf1793  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_2130, pow_268, mean_267, mul_1606, cat_132, mul_1607, add_607, convert_element_type_2140, mul_1610], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf1830, arg66_1, buf1526, buf1841, 16, 128, stream=stream0)
        buf1842 = buf983; del buf983  # reuse
        # Topologically Sorted Source Nodes: [mul_1612, sum_601], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf1841, arg70_1, buf1842, 15360, 128, stream=stream0)
        buf1851 = buf1821; del buf1821  # reuse
        # Topologically Sorted Source Nodes: [full_default_315, full_default_313, where_132, add_609, eq_66, logical_not_132, any_67, logical_not_133, full_default_317, , sub, exp, div_66, where_133], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_43.run(buf1842, arg1_1, buf1851, 16, 960, stream=stream0)
        buf1852 = buf993; del buf993  # reuse
        # Topologically Sorted Source Nodes: [mul_1613, sum_603], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf1851, arg71_1, buf1852, 16384, 120, stream=stream0)
        buf1853 = reinterpret_tensor(buf1841, (16, 1, 128), (128, 128, 1), 0); del buf1841  # reuse
        # Topologically Sorted Source Nodes: [mul_1613, sum_603], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf1852, buf1853, 2048, 8, stream=stream0)
        buf1854 = buf1830; del buf1830  # reuse
        # Topologically Sorted Source Nodes: [mul_1614, sum_604], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_17.run(buf1853, arg72_1, buf1854, 2048, 2048, stream=stream0)
        buf1856 = reinterpret_tensor(buf1853, (1, 2048), (2048, 1), 0); del buf1853  # reuse
        # Topologically Sorted Source Nodes: [add_603, add_610, convert_element_type_2147, pow_270, mean_269, convert_element_type_2149], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_30.run(buf1825, buf1828, buf1854, arg73_1, buf1856, 1, 2048, stream=stream0)
        buf1857 = buf1827; del buf1827  # reuse
        # Topologically Sorted Source Nodes: [mul_1617, sum_605], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_19.run(buf1856, arg74_1, buf1857, 12288, 2048, stream=stream0)
        buf1858 = buf1856; del buf1856  # reuse
        # Topologically Sorted Source Nodes: [mul_1620, sum_606], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf1857, arg75_1, buf1858, 2048, 6144, stream=stream0)
        buf1860 = buf1799; del buf1799  # reuse
        # Topologically Sorted Source Nodes: [add_603, add_610, add_612, convert_element_type_2157, pow_271, mean_270, add_613, rsqrt_270, mul_1621, convert_element_type_2158, mul_1622], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_mul_pow_rsqrt_31.run(buf1825, buf1828, buf1854, buf1858, arg76_1, buf1860, 1, 2048, stream=stream0)
        buf1864 = buf1833; del buf1833  # reuse
        # Topologically Sorted Source Nodes: [mul_1626, sum_608, mul_1629, sum_609, index_put_135], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_44.run(buf1860, arg79_1, arg81_1, arg1_1, buf1864, arg83_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_2167, pow_273, mean_272, mul_1632, cat_135, mul_1633, add_617, index_put_134], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_42.run(buf1864, arg80_1, arg1_1, buf1526, arg82_1, 8, 128, stream=stream0)
        buf1861 = buf1767; del buf1767  # reuse
        # Topologically Sorted Source Nodes: [mul_1623, sum_607], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_22.run(buf1860, arg77_1, buf1861, 2048, 2048, stream=stream0)
        buf1872 = reinterpret_tensor(buf1800, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf1800  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_2162, pow_272, mean_271, mul_1630, cat_134, mul_1631, add_616, convert_element_type_2172, mul_1634], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf1861, arg78_1, buf1526, buf1872, 16, 128, stream=stream0)
        buf1873 = buf1014; del buf1014  # reuse
        # Topologically Sorted Source Nodes: [mul_1636, sum_610], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf1872, arg82_1, buf1873, 15360, 128, stream=stream0)
        buf1882 = buf1851; del buf1851  # reuse
        # Topologically Sorted Source Nodes: [full_default_321, full_default_319, where_134, add_618, eq_67, logical_not_134, any_68, logical_not_135, full_default_323, , sub, exp, div_67, where_135], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_43.run(buf1873, arg1_1, buf1882, 16, 960, stream=stream0)
        buf1883 = buf1024; del buf1024  # reuse
        # Topologically Sorted Source Nodes: [mul_1637, sum_612], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf1882, arg83_1, buf1883, 16384, 120, stream=stream0)
        buf1884 = reinterpret_tensor(buf1872, (16, 1, 128), (128, 128, 1), 0); del buf1872  # reuse
        # Topologically Sorted Source Nodes: [mul_1637, sum_612], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf1883, buf1884, 2048, 8, stream=stream0)
        buf1886 = buf1825; del buf1825  # reuse
        # Topologically Sorted Source Nodes: [add_603, add_610, add_612, mul_1638, sum_613, add_619], Original ATen: [aten.add, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_sum_32.run(buf1886, buf1884, arg84_1, buf1828, buf1854, buf1858, 2048, 2048, stream=stream0)
        buf1887 = buf1829; del buf1829  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_2179, pow_274, mean_273], Original ATen: [prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_mean_pow_25.run(buf1886, buf1887, 1, 2048, stream=stream0)
        buf1888 = buf1857; del buf1857  # reuse
        # Topologically Sorted Source Nodes: [mul_1641, sum_614], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_26.run(arg85_1, buf1886, buf1887, arg86_1, buf1888, 12288, 2048, stream=stream0)
        buf1889 = reinterpret_tensor(buf1884, (1, 2048), (2048, 1), 0); del buf1884  # reuse
        # Topologically Sorted Source Nodes: [mul_1644, sum_615], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf1888, arg87_1, buf1889, 2048, 6144, stream=stream0)
        buf1890 = buf1887; del buf1887  # reuse
        # Topologically Sorted Source Nodes: [add_621, convert_element_type_2189, pow_275, mean_274], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_27.run(buf1886, buf1889, buf1890, 1, 2048, stream=stream0)
        buf1894 = buf1864; del buf1864  # reuse
        # Topologically Sorted Source Nodes: [mul_1650, sum_617, mul_1653, sum_618, index_put_137], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_45.run(arg88_1, buf1886, buf1889, buf1890, arg91_1, arg93_1, arg1_1, buf1894, arg95_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_2199, pow_277, mean_276, mul_1656, cat_137, mul_1657, add_626, index_put_136], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_42.run(buf1894, arg92_1, arg1_1, buf1526, arg94_1, 8, 128, stream=stream0)
        buf1891 = buf1858; del buf1858  # reuse
        # Topologically Sorted Source Nodes: [mul_1647, sum_616], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_28.run(arg88_1, buf1886, buf1889, buf1890, arg89_1, buf1891, 2048, 2048, stream=stream0)
        buf1902 = reinterpret_tensor(buf1854, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf1854  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_2194, pow_276, mean_275, mul_1654, cat_136, mul_1655, add_625, convert_element_type_2204, mul_1658], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf1891, arg90_1, buf1526, buf1902, 16, 128, stream=stream0)
        buf1903 = buf1044; del buf1044  # reuse
        # Topologically Sorted Source Nodes: [mul_1660, sum_619], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf1902, arg94_1, buf1903, 15360, 128, stream=stream0)
        buf1912 = buf1882; del buf1882  # reuse
        # Topologically Sorted Source Nodes: [full_default_327, full_default_325, where_136, add_627, eq_68, logical_not_136, any_69, logical_not_137, full_default_329, , sub, exp, div_68, where_137], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_43.run(buf1903, arg1_1, buf1912, 16, 960, stream=stream0)
        buf1913 = buf1054; del buf1054  # reuse
        # Topologically Sorted Source Nodes: [mul_1661, sum_621], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf1912, arg95_1, buf1913, 16384, 120, stream=stream0)
        buf1914 = reinterpret_tensor(buf1902, (16, 1, 128), (128, 128, 1), 0); del buf1902  # reuse
        # Topologically Sorted Source Nodes: [mul_1661, sum_621], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf1913, buf1914, 2048, 8, stream=stream0)
        buf1915 = buf1891; del buf1891  # reuse
        # Topologically Sorted Source Nodes: [mul_1662, sum_622], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_17.run(buf1914, arg96_1, buf1915, 2048, 2048, stream=stream0)
        buf1917 = reinterpret_tensor(buf1914, (1, 2048), (2048, 1), 0); del buf1914  # reuse
        # Topologically Sorted Source Nodes: [add_621, add_628, convert_element_type_2211, pow_278, mean_277, convert_element_type_2213], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_30.run(buf1886, buf1889, buf1915, arg97_1, buf1917, 1, 2048, stream=stream0)
        buf1918 = buf1888; del buf1888  # reuse
        # Topologically Sorted Source Nodes: [mul_1665, sum_623], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_19.run(buf1917, arg98_1, buf1918, 12288, 2048, stream=stream0)
        buf1919 = buf1917; del buf1917  # reuse
        # Topologically Sorted Source Nodes: [mul_1668, sum_624], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf1918, arg99_1, buf1919, 2048, 6144, stream=stream0)
        buf1921 = buf1860; del buf1860  # reuse
        # Topologically Sorted Source Nodes: [add_621, add_628, add_630, convert_element_type_2221, pow_279, mean_278, add_631, rsqrt_278, mul_1669, convert_element_type_2222, mul_1670], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_mul_pow_rsqrt_31.run(buf1886, buf1889, buf1915, buf1919, arg100_1, buf1921, 1, 2048, stream=stream0)
        buf1925 = buf1894; del buf1894  # reuse
        # Topologically Sorted Source Nodes: [mul_1674, sum_626, mul_1677, sum_627, index_put_139], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_44.run(buf1921, arg103_1, arg105_1, arg1_1, buf1925, arg107_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_2231, pow_281, mean_280, mul_1680, cat_139, mul_1681, add_635, index_put_138], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_42.run(buf1925, arg104_1, arg1_1, buf1526, arg106_1, 8, 128, stream=stream0)
        buf1922 = buf1828; del buf1828  # reuse
        # Topologically Sorted Source Nodes: [mul_1671, sum_625], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_22.run(buf1921, arg101_1, buf1922, 2048, 2048, stream=stream0)
        buf1933 = reinterpret_tensor(buf1861, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf1861  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_2226, pow_280, mean_279, mul_1678, cat_138, mul_1679, add_634, convert_element_type_2236, mul_1682], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf1922, arg102_1, buf1526, buf1933, 16, 128, stream=stream0)
        buf1934 = buf1075; del buf1075  # reuse
        # Topologically Sorted Source Nodes: [mul_1684, sum_628], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf1933, arg106_1, buf1934, 15360, 128, stream=stream0)
        buf1943 = buf1912; del buf1912  # reuse
        # Topologically Sorted Source Nodes: [full_default_333, full_default_331, where_138, add_636, eq_69, logical_not_138, any_70, logical_not_139, full_default_335, , sub, exp, div_69, where_139], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_43.run(buf1934, arg1_1, buf1943, 16, 960, stream=stream0)
        buf1944 = buf1085; del buf1085  # reuse
        # Topologically Sorted Source Nodes: [mul_1685, sum_630], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf1943, arg107_1, buf1944, 16384, 120, stream=stream0)
        buf1945 = reinterpret_tensor(buf1933, (16, 1, 128), (128, 128, 1), 0); del buf1933  # reuse
        # Topologically Sorted Source Nodes: [mul_1685, sum_630], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf1944, buf1945, 2048, 8, stream=stream0)
        buf1947 = buf1886; del buf1886  # reuse
        # Topologically Sorted Source Nodes: [add_621, add_628, add_630, mul_1686, sum_631, add_637], Original ATen: [aten.add, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_sum_32.run(buf1947, buf1945, arg108_1, buf1889, buf1915, buf1919, 2048, 2048, stream=stream0)
        buf1948 = buf1890; del buf1890  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_2243, pow_282, mean_281], Original ATen: [prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_mean_pow_25.run(buf1947, buf1948, 1, 2048, stream=stream0)
        buf1949 = buf1918; del buf1918  # reuse
        # Topologically Sorted Source Nodes: [mul_1689, sum_632], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_26.run(arg109_1, buf1947, buf1948, arg110_1, buf1949, 12288, 2048, stream=stream0)
        buf1950 = reinterpret_tensor(buf1945, (1, 2048), (2048, 1), 0); del buf1945  # reuse
        # Topologically Sorted Source Nodes: [mul_1692, sum_633], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf1949, arg111_1, buf1950, 2048, 6144, stream=stream0)
        buf1951 = buf1948; del buf1948  # reuse
        # Topologically Sorted Source Nodes: [add_639, convert_element_type_2253, pow_283, mean_282], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_27.run(buf1947, buf1950, buf1951, 1, 2048, stream=stream0)
        buf1955 = buf1925; del buf1925  # reuse
        # Topologically Sorted Source Nodes: [mul_1698, sum_635, mul_1701, sum_636, index_put_141], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_45.run(arg112_1, buf1947, buf1950, buf1951, arg115_1, arg117_1, arg1_1, buf1955, arg119_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_2263, pow_285, mean_284, mul_1704, cat_141, mul_1705, add_644, index_put_140], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_42.run(buf1955, arg116_1, arg1_1, buf1526, arg118_1, 8, 128, stream=stream0)
        buf1952 = buf1919; del buf1919  # reuse
        # Topologically Sorted Source Nodes: [mul_1695, sum_634], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_28.run(arg112_1, buf1947, buf1950, buf1951, arg113_1, buf1952, 2048, 2048, stream=stream0)
        buf1963 = reinterpret_tensor(buf1915, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf1915  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_2258, pow_284, mean_283, mul_1702, cat_140, mul_1703, add_643, convert_element_type_2268, mul_1706], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf1952, arg114_1, buf1526, buf1963, 16, 128, stream=stream0)
        buf1964 = buf1105; del buf1105  # reuse
        # Topologically Sorted Source Nodes: [mul_1708, sum_637], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf1963, arg118_1, buf1964, 15360, 128, stream=stream0)
        buf1973 = buf1943; del buf1943  # reuse
        # Topologically Sorted Source Nodes: [full_default_339, full_default_337, where_140, add_645, eq_70, logical_not_140, any_71, logical_not_141, full_default_341, , sub, exp, div_70, where_141], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_43.run(buf1964, arg1_1, buf1973, 16, 960, stream=stream0)
        buf1974 = buf1115; del buf1115  # reuse
        # Topologically Sorted Source Nodes: [mul_1709, sum_639], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf1973, arg119_1, buf1974, 16384, 120, stream=stream0)
        buf1975 = reinterpret_tensor(buf1963, (16, 1, 128), (128, 128, 1), 0); del buf1963  # reuse
        # Topologically Sorted Source Nodes: [mul_1709, sum_639], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf1974, buf1975, 2048, 8, stream=stream0)
        buf1976 = buf1952; del buf1952  # reuse
        # Topologically Sorted Source Nodes: [mul_1710, sum_640], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_17.run(buf1975, arg120_1, buf1976, 2048, 2048, stream=stream0)
        buf1978 = reinterpret_tensor(buf1975, (1, 2048), (2048, 1), 0); del buf1975  # reuse
        # Topologically Sorted Source Nodes: [add_639, add_646, convert_element_type_2275, pow_286, mean_285, convert_element_type_2277], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_30.run(buf1947, buf1950, buf1976, arg121_1, buf1978, 1, 2048, stream=stream0)
        buf1979 = buf1949; del buf1949  # reuse
        # Topologically Sorted Source Nodes: [mul_1713, sum_641], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_19.run(buf1978, arg122_1, buf1979, 12288, 2048, stream=stream0)
        buf1980 = buf1978; del buf1978  # reuse
        # Topologically Sorted Source Nodes: [mul_1716, sum_642], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf1979, arg123_1, buf1980, 2048, 6144, stream=stream0)
        buf1982 = buf1921; del buf1921  # reuse
        # Topologically Sorted Source Nodes: [add_639, add_646, add_648, convert_element_type_2285, pow_287, mean_286, add_649, rsqrt_286, mul_1717, convert_element_type_2286, mul_1718], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_mul_pow_rsqrt_31.run(buf1947, buf1950, buf1976, buf1980, arg124_1, buf1982, 1, 2048, stream=stream0)
        buf1986 = buf1955; del buf1955  # reuse
        # Topologically Sorted Source Nodes: [mul_1722, sum_644, mul_1725, sum_645, index_put_143], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_44.run(buf1982, arg127_1, arg129_1, arg1_1, buf1986, arg131_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_2295, pow_289, mean_288, mul_1728, cat_143, mul_1729, add_653, index_put_142], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_42.run(buf1986, arg128_1, arg1_1, buf1526, arg130_1, 8, 128, stream=stream0)
        buf1983 = buf1889; del buf1889  # reuse
        # Topologically Sorted Source Nodes: [mul_1719, sum_643], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_22.run(buf1982, arg125_1, buf1983, 2048, 2048, stream=stream0)
        buf1994 = reinterpret_tensor(buf1922, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf1922  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_2290, pow_288, mean_287, mul_1726, cat_142, mul_1727, add_652, convert_element_type_2300, mul_1730], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf1983, arg126_1, buf1526, buf1994, 16, 128, stream=stream0)
        buf1995 = buf1136; del buf1136  # reuse
        # Topologically Sorted Source Nodes: [mul_1732, sum_646], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf1994, arg130_1, buf1995, 15360, 128, stream=stream0)
        buf2004 = buf1973; del buf1973  # reuse
        # Topologically Sorted Source Nodes: [full_default_345, full_default_343, where_142, add_654, eq_71, logical_not_142, any_72, logical_not_143, full_default_347, , sub, exp, div_71, where_143], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_43.run(buf1995, arg1_1, buf2004, 16, 960, stream=stream0)
        buf2005 = buf1146; del buf1146  # reuse
        # Topologically Sorted Source Nodes: [mul_1733, sum_648], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf2004, arg131_1, buf2005, 16384, 120, stream=stream0)
        buf2006 = reinterpret_tensor(buf1994, (16, 1, 128), (128, 128, 1), 0); del buf1994  # reuse
        # Topologically Sorted Source Nodes: [mul_1733, sum_648], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf2005, buf2006, 2048, 8, stream=stream0)
        buf2008 = buf1947; del buf1947  # reuse
        # Topologically Sorted Source Nodes: [add_639, add_646, add_648, mul_1734, sum_649, add_655], Original ATen: [aten.add, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_sum_32.run(buf2008, buf2006, arg132_1, buf1950, buf1976, buf1980, 2048, 2048, stream=stream0)
        buf2009 = buf1951; del buf1951  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_2307, pow_290, mean_289], Original ATen: [prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_mean_pow_25.run(buf2008, buf2009, 1, 2048, stream=stream0)
        buf2010 = buf1979; del buf1979  # reuse
        # Topologically Sorted Source Nodes: [mul_1737, sum_650], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_26.run(arg133_1, buf2008, buf2009, arg134_1, buf2010, 12288, 2048, stream=stream0)
        buf2011 = reinterpret_tensor(buf2006, (1, 2048), (2048, 1), 0); del buf2006  # reuse
        # Topologically Sorted Source Nodes: [mul_1740, sum_651], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf2010, arg135_1, buf2011, 2048, 6144, stream=stream0)
        buf2012 = buf2009; del buf2009  # reuse
        # Topologically Sorted Source Nodes: [add_657, convert_element_type_2317, pow_291, mean_290], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_27.run(buf2008, buf2011, buf2012, 1, 2048, stream=stream0)
        buf2016 = buf1986; del buf1986  # reuse
        # Topologically Sorted Source Nodes: [mul_1746, sum_653, mul_1749, sum_654, index_put_145], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_45.run(arg136_1, buf2008, buf2011, buf2012, arg139_1, arg141_1, arg1_1, buf2016, arg143_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_2327, pow_293, mean_292, mul_1752, cat_145, mul_1753, add_662, index_put_144], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_42.run(buf2016, arg140_1, arg1_1, buf1526, arg142_1, 8, 128, stream=stream0)
        buf2013 = buf1980; del buf1980  # reuse
        # Topologically Sorted Source Nodes: [mul_1743, sum_652], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_28.run(arg136_1, buf2008, buf2011, buf2012, arg137_1, buf2013, 2048, 2048, stream=stream0)
        buf2024 = reinterpret_tensor(buf1976, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf1976  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_2322, pow_292, mean_291, mul_1750, cat_144, mul_1751, add_661, convert_element_type_2332, mul_1754], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf2013, arg138_1, buf1526, buf2024, 16, 128, stream=stream0)
        buf2025 = buf1166; del buf1166  # reuse
        # Topologically Sorted Source Nodes: [mul_1756, sum_655], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf2024, arg142_1, buf2025, 15360, 128, stream=stream0)
        buf2034 = buf2004; del buf2004  # reuse
        # Topologically Sorted Source Nodes: [full_default_351, full_default_349, where_144, add_663, eq_72, logical_not_144, any_73, logical_not_145, full_default_353, , sub, exp, div_72, where_145], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_43.run(buf2025, arg1_1, buf2034, 16, 960, stream=stream0)
        buf2035 = buf1176; del buf1176  # reuse
        # Topologically Sorted Source Nodes: [mul_1757, sum_657], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf2034, arg143_1, buf2035, 16384, 120, stream=stream0)
        buf2036 = reinterpret_tensor(buf2024, (16, 1, 128), (128, 128, 1), 0); del buf2024  # reuse
        # Topologically Sorted Source Nodes: [mul_1757, sum_657], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf2035, buf2036, 2048, 8, stream=stream0)
        buf2037 = buf2013; del buf2013  # reuse
        # Topologically Sorted Source Nodes: [mul_1758, sum_658], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_17.run(buf2036, arg144_1, buf2037, 2048, 2048, stream=stream0)
        buf2039 = reinterpret_tensor(buf2036, (1, 2048), (2048, 1), 0); del buf2036  # reuse
        # Topologically Sorted Source Nodes: [add_657, add_664, convert_element_type_2339, pow_294, mean_293, convert_element_type_2341], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_30.run(buf2008, buf2011, buf2037, arg145_1, buf2039, 1, 2048, stream=stream0)
        buf2040 = buf2010; del buf2010  # reuse
        # Topologically Sorted Source Nodes: [mul_1761, sum_659], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_19.run(buf2039, arg146_1, buf2040, 12288, 2048, stream=stream0)
        buf2041 = buf2039; del buf2039  # reuse
        # Topologically Sorted Source Nodes: [mul_1764, sum_660], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf2040, arg147_1, buf2041, 2048, 6144, stream=stream0)
        buf2043 = buf1982; del buf1982  # reuse
        # Topologically Sorted Source Nodes: [add_657, add_664, add_666, convert_element_type_2349, pow_295, mean_294, add_667, rsqrt_294, mul_1765, convert_element_type_2350, mul_1766], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_mul_pow_rsqrt_31.run(buf2008, buf2011, buf2037, buf2041, arg148_1, buf2043, 1, 2048, stream=stream0)
        buf2047 = buf2016; del buf2016  # reuse
        # Topologically Sorted Source Nodes: [mul_1770, sum_662, mul_1773, sum_663, index_put_147], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_44.run(buf2043, arg151_1, arg153_1, arg1_1, buf2047, arg155_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_2359, pow_297, mean_296, mul_1776, cat_147, mul_1777, add_671, index_put_146], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_42.run(buf2047, arg152_1, arg1_1, buf1526, arg154_1, 8, 128, stream=stream0)
        buf2044 = buf1950; del buf1950  # reuse
        # Topologically Sorted Source Nodes: [mul_1767, sum_661], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_22.run(buf2043, arg149_1, buf2044, 2048, 2048, stream=stream0)
        buf2055 = reinterpret_tensor(buf1983, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf1983  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_2354, pow_296, mean_295, mul_1774, cat_146, mul_1775, add_670, convert_element_type_2364, mul_1778], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf2044, arg150_1, buf1526, buf2055, 16, 128, stream=stream0)
        buf2056 = buf1197; del buf1197  # reuse
        # Topologically Sorted Source Nodes: [mul_1780, sum_664], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf2055, arg154_1, buf2056, 15360, 128, stream=stream0)
        buf2065 = buf2034; del buf2034  # reuse
        # Topologically Sorted Source Nodes: [full_default_357, full_default_355, where_146, add_672, eq_73, logical_not_146, any_74, logical_not_147, full_default_359, , sub, exp, div_73, where_147], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_43.run(buf2056, arg1_1, buf2065, 16, 960, stream=stream0)
        buf2066 = buf1207; del buf1207  # reuse
        # Topologically Sorted Source Nodes: [mul_1781, sum_666], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf2065, arg155_1, buf2066, 16384, 120, stream=stream0)
        buf2067 = reinterpret_tensor(buf2055, (16, 1, 128), (128, 128, 1), 0); del buf2055  # reuse
        # Topologically Sorted Source Nodes: [mul_1781, sum_666], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf2066, buf2067, 2048, 8, stream=stream0)
        buf2069 = buf2008; del buf2008  # reuse
        # Topologically Sorted Source Nodes: [add_657, add_664, add_666, mul_1782, sum_667, add_673], Original ATen: [aten.add, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_sum_32.run(buf2069, buf2067, arg156_1, buf2011, buf2037, buf2041, 2048, 2048, stream=stream0)
        buf2070 = buf2012; del buf2012  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_2371, pow_298, mean_297], Original ATen: [prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_mean_pow_25.run(buf2069, buf2070, 1, 2048, stream=stream0)
        buf2071 = buf2040; del buf2040  # reuse
        # Topologically Sorted Source Nodes: [mul_1785, sum_668], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_26.run(arg157_1, buf2069, buf2070, arg158_1, buf2071, 12288, 2048, stream=stream0)
        buf2072 = reinterpret_tensor(buf2067, (1, 2048), (2048, 1), 0); del buf2067  # reuse
        # Topologically Sorted Source Nodes: [mul_1788, sum_669], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf2071, arg159_1, buf2072, 2048, 6144, stream=stream0)
        buf2073 = buf2070; del buf2070  # reuse
        # Topologically Sorted Source Nodes: [add_675, convert_element_type_2381, pow_299, mean_298], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_27.run(buf2069, buf2072, buf2073, 1, 2048, stream=stream0)
        buf2077 = buf2047; del buf2047  # reuse
        # Topologically Sorted Source Nodes: [mul_1794, sum_671, mul_1797, sum_672, index_put_149], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_45.run(arg160_1, buf2069, buf2072, buf2073, arg163_1, arg165_1, arg1_1, buf2077, arg167_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_2391, pow_301, mean_300, mul_1800, cat_149, mul_1801, add_680, index_put_148], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_42.run(buf2077, arg164_1, arg1_1, buf1526, arg166_1, 8, 128, stream=stream0)
        buf2074 = buf2041; del buf2041  # reuse
        # Topologically Sorted Source Nodes: [mul_1791, sum_670], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_28.run(arg160_1, buf2069, buf2072, buf2073, arg161_1, buf2074, 2048, 2048, stream=stream0)
        buf2085 = reinterpret_tensor(buf2037, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf2037  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_2386, pow_300, mean_299, mul_1798, cat_148, mul_1799, add_679, convert_element_type_2396, mul_1802], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf2074, arg162_1, buf1526, buf2085, 16, 128, stream=stream0)
        buf2086 = buf1227; del buf1227  # reuse
        # Topologically Sorted Source Nodes: [mul_1804, sum_673], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf2085, arg166_1, buf2086, 15360, 128, stream=stream0)
        buf2095 = buf2065; del buf2065  # reuse
        # Topologically Sorted Source Nodes: [full_default_363, full_default_361, where_148, add_681, eq_74, logical_not_148, any_75, logical_not_149, full_default_365, , sub, exp, div_74, where_149], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_43.run(buf2086, arg1_1, buf2095, 16, 960, stream=stream0)
        buf2096 = buf1237; del buf1237  # reuse
        # Topologically Sorted Source Nodes: [mul_1805, sum_675], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf2095, arg167_1, buf2096, 16384, 120, stream=stream0)
        buf2097 = reinterpret_tensor(buf2085, (16, 1, 128), (128, 128, 1), 0); del buf2085  # reuse
        # Topologically Sorted Source Nodes: [mul_1805, sum_675], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf2096, buf2097, 2048, 8, stream=stream0)
        buf2098 = buf2074; del buf2074  # reuse
        # Topologically Sorted Source Nodes: [mul_1806, sum_676], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_17.run(buf2097, arg168_1, buf2098, 2048, 2048, stream=stream0)
        buf2100 = reinterpret_tensor(buf2097, (1, 2048), (2048, 1), 0); del buf2097  # reuse
        # Topologically Sorted Source Nodes: [add_675, add_682, convert_element_type_2403, pow_302, mean_301, convert_element_type_2405], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_30.run(buf2069, buf2072, buf2098, arg169_1, buf2100, 1, 2048, stream=stream0)
        buf2101 = buf2071; del buf2071  # reuse
        # Topologically Sorted Source Nodes: [mul_1809, sum_677], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_19.run(buf2100, arg170_1, buf2101, 12288, 2048, stream=stream0)
        buf2102 = buf2100; del buf2100  # reuse
        # Topologically Sorted Source Nodes: [mul_1812, sum_678], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf2101, arg171_1, buf2102, 2048, 6144, stream=stream0)
        buf2104 = buf2043; del buf2043  # reuse
        # Topologically Sorted Source Nodes: [add_675, add_682, add_684, convert_element_type_2413, pow_303, mean_302, add_685, rsqrt_302, mul_1813, convert_element_type_2414, mul_1814], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_mul_pow_rsqrt_31.run(buf2069, buf2072, buf2098, buf2102, arg172_1, buf2104, 1, 2048, stream=stream0)
        buf2108 = buf2077; del buf2077  # reuse
        # Topologically Sorted Source Nodes: [mul_1818, sum_680, mul_1821, sum_681, index_put_151], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_44.run(buf2104, arg175_1, arg177_1, arg1_1, buf2108, arg179_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_2423, pow_305, mean_304, mul_1824, cat_151, mul_1825, add_689, index_put_150], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_42.run(buf2108, arg176_1, arg1_1, buf1526, arg178_1, 8, 128, stream=stream0)
        buf2105 = buf2011; del buf2011  # reuse
        # Topologically Sorted Source Nodes: [mul_1815, sum_679], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_22.run(buf2104, arg173_1, buf2105, 2048, 2048, stream=stream0)
        buf2116 = reinterpret_tensor(buf2044, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf2044  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_2418, pow_304, mean_303, mul_1822, cat_150, mul_1823, add_688, convert_element_type_2428, mul_1826], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf2105, arg174_1, buf1526, buf2116, 16, 128, stream=stream0)
        buf2117 = buf1258; del buf1258  # reuse
        # Topologically Sorted Source Nodes: [mul_1828, sum_682], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf2116, arg178_1, buf2117, 15360, 128, stream=stream0)
        buf2126 = buf2095; del buf2095  # reuse
        # Topologically Sorted Source Nodes: [full_default_369, full_default_367, where_150, add_690, eq_75, logical_not_150, any_76, logical_not_151, full_default_371, , sub, exp, div_75, where_151], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_43.run(buf2117, arg1_1, buf2126, 16, 960, stream=stream0)
        buf2127 = buf1268; del buf1268  # reuse
        # Topologically Sorted Source Nodes: [mul_1829, sum_684], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf2126, arg179_1, buf2127, 16384, 120, stream=stream0)
        buf2128 = reinterpret_tensor(buf2116, (16, 1, 128), (128, 128, 1), 0); del buf2116  # reuse
        # Topologically Sorted Source Nodes: [mul_1829, sum_684], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf2127, buf2128, 2048, 8, stream=stream0)
        buf2130 = buf2069; del buf2069  # reuse
        # Topologically Sorted Source Nodes: [add_675, add_682, add_684, mul_1830, sum_685, add_691], Original ATen: [aten.add, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_sum_32.run(buf2130, buf2128, arg180_1, buf2072, buf2098, buf2102, 2048, 2048, stream=stream0)
        buf2131 = buf2073; del buf2073  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_2435, pow_306, mean_305], Original ATen: [prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_mean_pow_25.run(buf2130, buf2131, 1, 2048, stream=stream0)
        buf2132 = buf2101; del buf2101  # reuse
        # Topologically Sorted Source Nodes: [mul_1833, sum_686], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_26.run(arg181_1, buf2130, buf2131, arg182_1, buf2132, 12288, 2048, stream=stream0)
        buf2133 = reinterpret_tensor(buf2128, (1, 2048), (2048, 1), 0); del buf2128  # reuse
        # Topologically Sorted Source Nodes: [mul_1836, sum_687], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf2132, arg183_1, buf2133, 2048, 6144, stream=stream0)
        buf2134 = buf2131; del buf2131  # reuse
        # Topologically Sorted Source Nodes: [add_693, convert_element_type_2445, pow_307, mean_306], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_27.run(buf2130, buf2133, buf2134, 1, 2048, stream=stream0)
        buf2138 = buf2108; del buf2108  # reuse
        # Topologically Sorted Source Nodes: [mul_1842, sum_689, mul_1845, sum_690, index_put_153], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_45.run(arg184_1, buf2130, buf2133, buf2134, arg187_1, arg189_1, arg1_1, buf2138, arg191_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_2455, pow_309, mean_308, mul_1848, cat_153, mul_1849, add_698, index_put_152], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_42.run(buf2138, arg188_1, arg1_1, buf1526, arg190_1, 8, 128, stream=stream0)
        buf2135 = buf2102; del buf2102  # reuse
        # Topologically Sorted Source Nodes: [mul_1839, sum_688], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_28.run(arg184_1, buf2130, buf2133, buf2134, arg185_1, buf2135, 2048, 2048, stream=stream0)
        buf2146 = reinterpret_tensor(buf2098, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf2098  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_2450, pow_308, mean_307, mul_1846, cat_152, mul_1847, add_697, convert_element_type_2460, mul_1850], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf2135, arg186_1, buf1526, buf2146, 16, 128, stream=stream0)
        buf2147 = buf1288; del buf1288  # reuse
        # Topologically Sorted Source Nodes: [mul_1852, sum_691], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf2146, arg190_1, buf2147, 15360, 128, stream=stream0)
        buf2156 = buf2126; del buf2126  # reuse
        # Topologically Sorted Source Nodes: [full_default_375, full_default_373, where_152, add_699, eq_76, logical_not_152, any_77, logical_not_153, full_default_377, , sub, exp, div_76, where_153], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_43.run(buf2147, arg1_1, buf2156, 16, 960, stream=stream0)
        buf2157 = buf1298; del buf1298  # reuse
        # Topologically Sorted Source Nodes: [mul_1853, sum_693], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf2156, arg191_1, buf2157, 16384, 120, stream=stream0)
        buf2158 = reinterpret_tensor(buf2146, (16, 1, 128), (128, 128, 1), 0); del buf2146  # reuse
        # Topologically Sorted Source Nodes: [mul_1853, sum_693], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf2157, buf2158, 2048, 8, stream=stream0)
        buf2159 = buf2135; del buf2135  # reuse
        # Topologically Sorted Source Nodes: [mul_1854, sum_694], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_17.run(buf2158, arg192_1, buf2159, 2048, 2048, stream=stream0)
        buf2161 = reinterpret_tensor(buf2158, (1, 2048), (2048, 1), 0); del buf2158  # reuse
        # Topologically Sorted Source Nodes: [add_693, add_700, convert_element_type_2467, pow_310, mean_309, convert_element_type_2469], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_30.run(buf2130, buf2133, buf2159, arg193_1, buf2161, 1, 2048, stream=stream0)
        buf2162 = buf2132; del buf2132  # reuse
        # Topologically Sorted Source Nodes: [mul_1857, sum_695], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_19.run(buf2161, arg194_1, buf2162, 12288, 2048, stream=stream0)
        buf2163 = buf2161; del buf2161  # reuse
        # Topologically Sorted Source Nodes: [mul_1860, sum_696], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf2162, arg195_1, buf2163, 2048, 6144, stream=stream0)
        buf2165 = buf2104; del buf2104  # reuse
        # Topologically Sorted Source Nodes: [add_693, add_700, add_702, convert_element_type_2477, pow_311, mean_310, add_703, rsqrt_310, mul_1861, convert_element_type_2478, mul_1862], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_mul_pow_rsqrt_31.run(buf2130, buf2133, buf2159, buf2163, arg196_1, buf2165, 1, 2048, stream=stream0)
        buf2169 = buf2138; del buf2138  # reuse
        # Topologically Sorted Source Nodes: [mul_1866, sum_698, mul_1869, sum_699, index_put_155], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_44.run(buf2165, arg199_1, arg201_1, arg1_1, buf2169, arg203_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_2487, pow_313, mean_312, mul_1872, cat_155, mul_1873, add_707, index_put_154], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_42.run(buf2169, arg200_1, arg1_1, buf1526, arg202_1, 8, 128, stream=stream0)
        buf2166 = buf2072; del buf2072  # reuse
        # Topologically Sorted Source Nodes: [mul_1863, sum_697], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_22.run(buf2165, arg197_1, buf2166, 2048, 2048, stream=stream0)
        buf2177 = reinterpret_tensor(buf2105, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf2105  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_2482, pow_312, mean_311, mul_1870, cat_154, mul_1871, add_706, convert_element_type_2492, mul_1874], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf2166, arg198_1, buf1526, buf2177, 16, 128, stream=stream0)
        buf2178 = buf1319; del buf1319  # reuse
        # Topologically Sorted Source Nodes: [mul_1876, sum_700], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf2177, arg202_1, buf2178, 15360, 128, stream=stream0)
        buf2187 = buf2156; del buf2156  # reuse
        # Topologically Sorted Source Nodes: [full_default_381, full_default_379, where_154, add_708, eq_77, logical_not_154, any_78, logical_not_155, full_default_383, , sub, exp, div_77, where_155], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_43.run(buf2178, arg1_1, buf2187, 16, 960, stream=stream0)
        buf2188 = buf1329; del buf1329  # reuse
        # Topologically Sorted Source Nodes: [mul_1877, sum_702], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf2187, arg203_1, buf2188, 16384, 120, stream=stream0)
        buf2189 = reinterpret_tensor(buf2177, (16, 1, 128), (128, 128, 1), 0); del buf2177  # reuse
        # Topologically Sorted Source Nodes: [mul_1877, sum_702], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf2188, buf2189, 2048, 8, stream=stream0)
        buf2191 = buf2130; del buf2130  # reuse
        # Topologically Sorted Source Nodes: [add_693, add_700, add_702, mul_1878, sum_703, add_709], Original ATen: [aten.add, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_sum_32.run(buf2191, buf2189, arg204_1, buf2133, buf2159, buf2163, 2048, 2048, stream=stream0)
        buf2192 = buf2134; del buf2134  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_2499, pow_314, mean_313], Original ATen: [prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_mean_pow_25.run(buf2191, buf2192, 1, 2048, stream=stream0)
        buf2193 = buf2162; del buf2162  # reuse
        # Topologically Sorted Source Nodes: [mul_1881, sum_704], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_26.run(arg205_1, buf2191, buf2192, arg206_1, buf2193, 12288, 2048, stream=stream0)
        buf2194 = reinterpret_tensor(buf2189, (1, 2048), (2048, 1), 0); del buf2189  # reuse
        # Topologically Sorted Source Nodes: [mul_1884, sum_705], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf2193, arg207_1, buf2194, 2048, 6144, stream=stream0)
        buf2195 = buf2192; del buf2192  # reuse
        # Topologically Sorted Source Nodes: [add_711, convert_element_type_2509, pow_315, mean_314], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_27.run(buf2191, buf2194, buf2195, 1, 2048, stream=stream0)
        buf2199 = buf2169; del buf2169  # reuse
        # Topologically Sorted Source Nodes: [mul_1890, sum_707, mul_1893, sum_708, index_put_157], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_45.run(arg208_1, buf2191, buf2194, buf2195, arg211_1, arg213_1, arg1_1, buf2199, arg215_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_2519, pow_317, mean_316, mul_1896, cat_157, mul_1897, add_716, index_put_156], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_42.run(buf2199, arg212_1, arg1_1, buf1526, arg214_1, 8, 128, stream=stream0)
        buf2196 = buf2163; del buf2163  # reuse
        # Topologically Sorted Source Nodes: [mul_1887, sum_706], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_28.run(arg208_1, buf2191, buf2194, buf2195, arg209_1, buf2196, 2048, 2048, stream=stream0)
        buf2207 = reinterpret_tensor(buf2159, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf2159  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_2514, pow_316, mean_315, mul_1894, cat_156, mul_1895, add_715, convert_element_type_2524, mul_1898], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf2196, arg210_1, buf1526, buf2207, 16, 128, stream=stream0)
        buf2208 = buf1349; del buf1349  # reuse
        # Topologically Sorted Source Nodes: [mul_1900, sum_709], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf2207, arg214_1, buf2208, 15360, 128, stream=stream0)
        buf2217 = buf2187; del buf2187  # reuse
        # Topologically Sorted Source Nodes: [full_default_387, full_default_385, where_156, add_717, eq_78, logical_not_156, any_79, logical_not_157, full_default_389, , sub, exp, div_78, where_157], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_43.run(buf2208, arg1_1, buf2217, 16, 960, stream=stream0)
        buf2218 = buf1359; del buf1359  # reuse
        # Topologically Sorted Source Nodes: [mul_1901, sum_711], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf2217, arg215_1, buf2218, 16384, 120, stream=stream0)
        buf2219 = reinterpret_tensor(buf2207, (16, 1, 128), (128, 128, 1), 0); del buf2207  # reuse
        # Topologically Sorted Source Nodes: [mul_1901, sum_711], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf2218, buf2219, 2048, 8, stream=stream0)
        buf2220 = buf2196; del buf2196  # reuse
        # Topologically Sorted Source Nodes: [mul_1902, sum_712], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_17.run(buf2219, arg216_1, buf2220, 2048, 2048, stream=stream0)
        buf2222 = reinterpret_tensor(buf2219, (1, 2048), (2048, 1), 0); del buf2219  # reuse
        # Topologically Sorted Source Nodes: [add_711, add_718, convert_element_type_2531, pow_318, mean_317, convert_element_type_2533], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_30.run(buf2191, buf2194, buf2220, arg217_1, buf2222, 1, 2048, stream=stream0)
        buf2223 = buf2193; del buf2193  # reuse
        # Topologically Sorted Source Nodes: [mul_1905, sum_713], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_19.run(buf2222, arg218_1, buf2223, 12288, 2048, stream=stream0)
        buf2224 = buf2222; del buf2222  # reuse
        # Topologically Sorted Source Nodes: [mul_1908, sum_714], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf2223, arg219_1, buf2224, 2048, 6144, stream=stream0)
        buf2226 = buf2165; del buf2165  # reuse
        # Topologically Sorted Source Nodes: [add_711, add_718, add_720, convert_element_type_2541, pow_319, mean_318, add_721, rsqrt_318, mul_1909, convert_element_type_2542, mul_1910], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_mul_pow_rsqrt_31.run(buf2191, buf2194, buf2220, buf2224, arg220_1, buf2226, 1, 2048, stream=stream0)
        buf2230 = buf2199; del buf2199  # reuse
        # Topologically Sorted Source Nodes: [mul_1914, sum_716, mul_1917, sum_717, index_put_159], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_44.run(buf2226, arg223_1, arg225_1, arg1_1, buf2230, arg227_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_2551, pow_321, mean_320, mul_1920, cat_159, mul_1921, add_725, index_put_158], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_42.run(buf2230, arg224_1, arg1_1, buf1526, arg226_1, 8, 128, stream=stream0)
        buf2227 = buf2133; del buf2133  # reuse
        # Topologically Sorted Source Nodes: [mul_1911, sum_715], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_22.run(buf2226, arg221_1, buf2227, 2048, 2048, stream=stream0)
        buf2238 = reinterpret_tensor(buf2166, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf2166  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_2546, pow_320, mean_319, mul_1918, cat_158, mul_1919, add_724, convert_element_type_2556, mul_1922], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf2227, arg222_1, buf1526, buf2238, 16, 128, stream=stream0)
        buf2239 = buf1380; del buf1380  # reuse
        # Topologically Sorted Source Nodes: [mul_1924, sum_718], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf2238, arg226_1, buf2239, 15360, 128, stream=stream0)
        buf2248 = buf2217; del buf2217  # reuse
        # Topologically Sorted Source Nodes: [full_default_393, full_default_391, where_158, add_726, eq_79, logical_not_158, any_80, logical_not_159, full_default_395, , sub, exp, div_79, where_159], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_43.run(buf2239, arg1_1, buf2248, 16, 960, stream=stream0)
        buf2249 = buf1390; del buf1390  # reuse
        # Topologically Sorted Source Nodes: [mul_1925, sum_720], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf2248, arg227_1, buf2249, 16384, 120, stream=stream0)
        buf2250 = reinterpret_tensor(buf2238, (16, 1, 128), (128, 128, 1), 0); del buf2238  # reuse
        # Topologically Sorted Source Nodes: [mul_1925, sum_720], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf2249, buf2250, 2048, 8, stream=stream0)
        buf2252 = buf2191; del buf2191  # reuse
        # Topologically Sorted Source Nodes: [add_711, add_718, add_720, mul_1926, sum_721, add_727], Original ATen: [aten.add, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_sum_32.run(buf2252, buf2250, arg228_1, buf2194, buf2220, buf2224, 2048, 2048, stream=stream0)
        buf2253 = buf2195; del buf2195  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_2563, pow_322, mean_321], Original ATen: [prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_mean_pow_25.run(buf2252, buf2253, 1, 2048, stream=stream0)
        buf2254 = buf2223; del buf2223  # reuse
        # Topologically Sorted Source Nodes: [mul_1929, sum_722], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_26.run(arg229_1, buf2252, buf2253, arg230_1, buf2254, 12288, 2048, stream=stream0)
        buf2255 = reinterpret_tensor(buf2250, (1, 2048), (2048, 1), 0); del buf2250  # reuse
        # Topologically Sorted Source Nodes: [mul_1932, sum_723], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf2254, arg231_1, buf2255, 2048, 6144, stream=stream0)
        buf2256 = buf2253; del buf2253  # reuse
        # Topologically Sorted Source Nodes: [add_729, convert_element_type_2573, pow_323, mean_322], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_27.run(buf2252, buf2255, buf2256, 1, 2048, stream=stream0)
        buf2260 = buf2230; del buf2230  # reuse
        # Topologically Sorted Source Nodes: [mul_1938, sum_725, mul_1941, sum_726, index_put_161], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_45.run(arg232_1, buf2252, buf2255, buf2256, arg235_1, arg237_1, arg1_1, buf2260, arg239_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_2583, pow_325, mean_324, mul_1944, cat_161, mul_1945, add_734, index_put_160], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_42.run(buf2260, arg236_1, arg1_1, buf1526, arg238_1, 8, 128, stream=stream0)
        buf2257 = buf2224; del buf2224  # reuse
        # Topologically Sorted Source Nodes: [mul_1935, sum_724], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_28.run(arg232_1, buf2252, buf2255, buf2256, arg233_1, buf2257, 2048, 2048, stream=stream0)
        buf2268 = reinterpret_tensor(buf2220, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf2220  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_2578, pow_324, mean_323, mul_1942, cat_160, mul_1943, add_733, convert_element_type_2588, mul_1946], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf2257, arg234_1, buf1526, buf2268, 16, 128, stream=stream0)
        buf2269 = buf1410; del buf1410  # reuse
        # Topologically Sorted Source Nodes: [mul_1948, sum_727], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf2268, arg238_1, buf2269, 15360, 128, stream=stream0)
        buf2278 = buf2248; del buf2248  # reuse
        # Topologically Sorted Source Nodes: [full_default_399, full_default_397, where_160, add_735, eq_80, logical_not_160, any_81, logical_not_161, full_default_401, , sub, exp, div_80, where_161], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_43.run(buf2269, arg1_1, buf2278, 16, 960, stream=stream0)
        buf2279 = buf1420; del buf1420  # reuse
        # Topologically Sorted Source Nodes: [mul_1949, sum_729], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf2278, arg239_1, buf2279, 16384, 120, stream=stream0)
        buf2280 = reinterpret_tensor(buf2268, (16, 1, 128), (128, 128, 1), 0); del buf2268  # reuse
        # Topologically Sorted Source Nodes: [mul_1949, sum_729], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf2279, buf2280, 2048, 8, stream=stream0)
        buf2281 = buf2257; del buf2257  # reuse
        # Topologically Sorted Source Nodes: [mul_1950, sum_730], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_17.run(buf2280, arg240_1, buf2281, 2048, 2048, stream=stream0)
        buf2283 = reinterpret_tensor(buf2280, (1, 2048), (2048, 1), 0); del buf2280  # reuse
        # Topologically Sorted Source Nodes: [add_729, add_736, convert_element_type_2595, pow_326, mean_325, convert_element_type_2597], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_30.run(buf2252, buf2255, buf2281, arg241_1, buf2283, 1, 2048, stream=stream0)
        buf2284 = buf2254; del buf2254  # reuse
        # Topologically Sorted Source Nodes: [mul_1953, sum_731], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_19.run(buf2283, arg242_1, buf2284, 12288, 2048, stream=stream0)
        buf2285 = buf2283; del buf2283  # reuse
        # Topologically Sorted Source Nodes: [mul_1956, sum_732], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf2284, arg243_1, buf2285, 2048, 6144, stream=stream0)
        buf2287 = buf2226; del buf2226  # reuse
        # Topologically Sorted Source Nodes: [add_729, add_736, add_738, convert_element_type_2605, pow_327, mean_326, add_739, rsqrt_326, mul_1957, convert_element_type_2606, mul_1958], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_mul_pow_rsqrt_31.run(buf2252, buf2255, buf2281, buf2285, arg244_1, buf2287, 1, 2048, stream=stream0)
        buf2291 = buf2260; del buf2260  # reuse
        # Topologically Sorted Source Nodes: [mul_1962, sum_734, mul_1965, sum_735, index_put_163], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_44.run(buf2287, arg247_1, arg249_1, arg1_1, buf2291, arg251_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_2615, pow_329, mean_328, mul_1968, cat_163, mul_1969, add_743, index_put_162], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_42.run(buf2291, arg248_1, arg1_1, buf1526, arg250_1, 8, 128, stream=stream0)
        buf2288 = buf2194; del buf2194  # reuse
        # Topologically Sorted Source Nodes: [mul_1959, sum_733], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_22.run(buf2287, arg245_1, buf2288, 2048, 2048, stream=stream0)
        buf2299 = reinterpret_tensor(buf2227, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf2227  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_2610, pow_328, mean_327, mul_1966, cat_162, mul_1967, add_742, convert_element_type_2620, mul_1970], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf2288, arg246_1, buf1526, buf2299, 16, 128, stream=stream0)
        buf2300 = buf1441; del buf1441  # reuse
        # Topologically Sorted Source Nodes: [mul_1972, sum_736], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf2299, arg250_1, buf2300, 15360, 128, stream=stream0)
        buf2309 = buf2278; del buf2278  # reuse
        # Topologically Sorted Source Nodes: [full_default_405, full_default_403, where_162, add_744, eq_81, logical_not_162, any_82, logical_not_163, full_default_407, , sub, exp, div_81, where_163], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_43.run(buf2300, arg1_1, buf2309, 16, 960, stream=stream0)
        buf2310 = buf1451; del buf1451  # reuse
        # Topologically Sorted Source Nodes: [mul_1973, sum_738], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf2309, arg251_1, buf2310, 16384, 120, stream=stream0)
        buf2311 = reinterpret_tensor(buf2299, (16, 1, 128), (128, 128, 1), 0); del buf2299  # reuse
        # Topologically Sorted Source Nodes: [mul_1973, sum_738], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf2310, buf2311, 2048, 8, stream=stream0)
        buf2313 = buf2252; del buf2252  # reuse
        # Topologically Sorted Source Nodes: [add_729, add_736, add_738, mul_1974, sum_739, add_745], Original ATen: [aten.add, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_sum_32.run(buf2313, buf2311, arg252_1, buf2255, buf2281, buf2285, 2048, 2048, stream=stream0)
        buf2314 = buf2256; del buf2256  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_2627, pow_330, mean_329], Original ATen: [prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_mean_pow_25.run(buf2313, buf2314, 1, 2048, stream=stream0)
        buf2315 = buf2284; del buf2284  # reuse
        # Topologically Sorted Source Nodes: [mul_1977, sum_740], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_26.run(arg253_1, buf2313, buf2314, arg254_1, buf2315, 12288, 2048, stream=stream0)
        buf2316 = reinterpret_tensor(buf2311, (1, 2048), (2048, 1), 0); del buf2311  # reuse
        # Topologically Sorted Source Nodes: [mul_1980, sum_741], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf2315, arg255_1, buf2316, 2048, 6144, stream=stream0)
        buf2317 = buf2314; del buf2314  # reuse
        # Topologically Sorted Source Nodes: [add_747, convert_element_type_2637, pow_331, mean_330], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_27.run(buf2313, buf2316, buf2317, 1, 2048, stream=stream0)
        buf2321 = buf2291; del buf2291  # reuse
        # Topologically Sorted Source Nodes: [mul_1986, sum_743, mul_1989, sum_744, index_put_165], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_45.run(arg256_1, buf2313, buf2316, buf2317, arg259_1, arg261_1, arg1_1, buf2321, arg263_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_2647, pow_333, mean_332, mul_1992, cat_165, mul_1993, add_752, index_put_164], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_42.run(buf2321, arg260_1, arg1_1, buf1526, arg262_1, 8, 128, stream=stream0)
        buf2318 = buf2285; del buf2285  # reuse
        # Topologically Sorted Source Nodes: [mul_1983, sum_742], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_28.run(arg256_1, buf2313, buf2316, buf2317, arg257_1, buf2318, 2048, 2048, stream=stream0)
        buf2329 = reinterpret_tensor(buf2281, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf2281  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_2642, pow_332, mean_331, mul_1990, cat_164, mul_1991, add_751, convert_element_type_2652, mul_1994], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf2318, arg258_1, buf1526, buf2329, 16, 128, stream=stream0)
        buf2330 = buf1471; del buf1471  # reuse
        # Topologically Sorted Source Nodes: [mul_1996, sum_745], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf2329, arg262_1, buf2330, 15360, 128, stream=stream0)
        buf2339 = buf2309; del buf2309  # reuse
        # Topologically Sorted Source Nodes: [full_default_411, full_default_409, where_164, add_753, eq_82, logical_not_164, any_83, logical_not_165, full_default_413, , sub, exp, div_82, where_165], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_43.run(buf2330, arg1_1, buf2339, 16, 960, stream=stream0)
        buf2340 = buf1481; del buf1481  # reuse
        # Topologically Sorted Source Nodes: [mul_1997, sum_747], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf2339, arg263_1, buf2340, 16384, 120, stream=stream0)
        buf2341 = reinterpret_tensor(buf2329, (16, 1, 128), (128, 128, 1), 0); del buf2329  # reuse
        # Topologically Sorted Source Nodes: [mul_1997, sum_747], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf2340, buf2341, 2048, 8, stream=stream0)
        buf2342 = buf2318; del buf2318  # reuse
        # Topologically Sorted Source Nodes: [mul_1998, sum_748], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_17.run(buf2341, arg264_1, buf2342, 2048, 2048, stream=stream0)
        buf2344 = reinterpret_tensor(buf2341, (1, 2048), (2048, 1), 0); del buf2341  # reuse
        # Topologically Sorted Source Nodes: [add_747, add_754, convert_element_type_2659, pow_334, mean_333, convert_element_type_2661], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_30.run(buf2313, buf2316, buf2342, arg265_1, buf2344, 1, 2048, stream=stream0)
        buf2345 = buf2315; del buf2315  # reuse
        # Topologically Sorted Source Nodes: [mul_2001, sum_749], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_19.run(buf2344, arg266_1, buf2345, 12288, 2048, stream=stream0)
        buf2346 = buf2344; del buf2344  # reuse
        # Topologically Sorted Source Nodes: [mul_2004, sum_750], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf2345, arg267_1, buf2346, 2048, 6144, stream=stream0)
        buf2348 = buf2287; del buf2287  # reuse
        # Topologically Sorted Source Nodes: [add_747, add_754, add_756, convert_element_type_2669, pow_335, mean_334, add_757, rsqrt_334, mul_2005, convert_element_type_2670, mul_2006], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_mul_pow_rsqrt_31.run(buf2313, buf2316, buf2342, buf2346, arg268_1, buf2348, 1, 2048, stream=stream0)
        buf2352 = buf2321; del buf2321  # reuse
        # Topologically Sorted Source Nodes: [mul_2010, sum_752, mul_2013, sum_753, index_put_167], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_44.run(buf2348, arg271_1, arg273_1, arg1_1, buf2352, arg275_1, 1024, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [convert_element_type_2679, pow_337, mean_336, mul_2016, cat_167, mul_2017, add_761, index_put_166], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_42.run(buf2352, arg272_1, arg1_1, buf1526, arg274_1, 8, 128, stream=stream0)
        buf2349 = buf2255; del buf2255  # reuse
        # Topologically Sorted Source Nodes: [mul_2007, sum_751], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_22.run(buf2348, arg269_1, buf2349, 2048, 2048, stream=stream0)
        buf2360 = reinterpret_tensor(buf2288, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf2288  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_2674, pow_336, mean_335, mul_2014, cat_166, mul_2015, add_760, convert_element_type_2684, mul_2018], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf2349, arg270_1, buf1526, buf2360, 16, 128, stream=stream0)
        del buf1526
        buf2361 = buf1502; del buf1502  # reuse
        # Topologically Sorted Source Nodes: [mul_2020, sum_754], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf2360, arg274_1, buf2361, 15360, 128, stream=stream0)
        buf2370 = buf2339; del buf2339  # reuse
        # Topologically Sorted Source Nodes: [full_default_417, full_default_415, where_166, add_762, eq_83, logical_not_166, any_84, logical_not_167, full_default_419, , sub, exp, div_83, where_167], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_43.run(buf2361, arg1_1, buf2370, 16, 960, stream=stream0)
        buf2371 = buf1512; del buf1512  # reuse
        # Topologically Sorted Source Nodes: [mul_2021, sum_756], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf2370, arg275_1, buf2371, 16384, 120, stream=stream0)
        del buf2370
        buf2372 = reinterpret_tensor(buf2360, (16, 1, 128), (128, 128, 1), 0); del buf2360  # reuse
        # Topologically Sorted Source Nodes: [mul_2021, sum_756], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf2371, buf2372, 2048, 8, stream=stream0)
        buf2374 = buf2313; del buf2313  # reuse
        # Topologically Sorted Source Nodes: [add_747, add_754, add_756, mul_2022, sum_757, add_763], Original ATen: [aten.add, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_sum_32.run(buf2374, buf2372, arg276_1, buf2316, buf2342, buf2346, 2048, 2048, stream=stream0)
        buf2375 = buf2317; del buf2317  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_2691, pow_338, mean_337], Original ATen: [prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_mean_pow_25.run(buf2374, buf2375, 1, 2048, stream=stream0)
        buf2376 = buf2345; del buf2345  # reuse
        # Topologically Sorted Source Nodes: [mul_2025, sum_758], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_26.run(arg277_1, buf2374, buf2375, arg278_1, buf2376, 12288, 2048, stream=stream0)
        buf2377 = reinterpret_tensor(buf2372, (1, 2048), (2048, 1), 0); del buf2372  # reuse
        # Topologically Sorted Source Nodes: [mul_2028, sum_759], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf2376, arg279_1, buf2377, 2048, 6144, stream=stream0)
        buf2378 = buf2375; del buf2375  # reuse
        # Topologically Sorted Source Nodes: [add_765, convert_element_type_2701, pow_339, mean_338], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_27.run(buf2374, buf2377, buf2378, 1, 2048, stream=stream0)
        buf2379 = buf1520; del buf1520  # reuse
        # Topologically Sorted Source Nodes: [logits_2], Original ATen: [aten.mm]
        stream0 = get_raw_stream(0)
        triton_red_fused_mm_33.run(arg280_1, buf2374, buf2377, buf2378, arg282_1, buf2379, 151936, 2048, stream=stream0)
        buf2380 = buf1521; del buf1521  # reuse
        buf3242 = reinterpret_tensor(buf3244, (1, 1), (4, 1), 2)  # alias
        # Topologically Sorted Source Nodes: [logits_2, argmax_2, cat_224], Original ATen: [aten.mm, aten.argmax, aten.cat]
        stream0 = get_raw_stream(0)
        triton_red_fused_argmax_cat_mm_40.run(buf2379, buf2380, buf3242, 1, 151936, stream=stream0)
        buf2381 = buf2378; del buf2378  # reuse
        # Topologically Sorted Source Nodes: [embedding_3, convert_element_type_2706, pow_340, mean_339], Original ATen: [aten.embedding, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_embedding_mean_pow_0.run(buf2380, arg282_1, buf2381, 1, 2048, stream=stream0)
        buf2387 = buf2352; del buf2352  # reuse
        # Topologically Sorted Source Nodes: [mul_2039, sum_762, mul_2042, sum_763, index_put_169], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_46.run(arg284_1, buf2380, arg282_1, buf2381, arg287_1, arg289_1, arg1_1, buf2387, arg291_1, 1024, 2048, stream=stream0)
        del arg287_1
        del arg289_1
        # Topologically Sorted Source Nodes: [convert_element_type_2716, pow_342, mean_341, mul_2045, cat_169, mul_2046, add_774, index_put_168], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_47.run(buf2387, arg288_1, arg1_1, buf2385, arg290_1, 8, 128, stream=stream0)
        del arg288_1
        buf2382 = buf2377; del buf2377  # reuse
        # Topologically Sorted Source Nodes: [mul_2036, sum_761], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_5.run(arg284_1, buf2380, arg282_1, buf2381, arg285_1, buf2382, 2048, 2048, stream=stream0)
        del arg284_1
        del arg285_1
        buf2395 = reinterpret_tensor(buf2346, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf2346  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_2711, pow_341, mean_340, mul_2043, cat_168, mul_2044, add_773, convert_element_type_2721, mul_2047], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf2382, arg286_1, buf2385, buf2395, 16, 128, stream=stream0)
        del arg286_1
        buf2396 = buf1537; del buf1537  # reuse
        # Topologically Sorted Source Nodes: [mul_2049, sum_764], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf2395, arg290_1, buf2396, 15360, 128, stream=stream0)
        del arg290_1
        buf2405 = reinterpret_tensor(buf2396, (1, 16, 1, 960), (15360, 960, 15360, 1), 0); del buf2396  # reuse
        # Topologically Sorted Source Nodes: [full_default_423, full_default_421, where_168, add_775, eq_84, logical_not_168, any_85, logical_not_169, full_default_425, , sub, exp, div_84, where_169], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_48.run(buf2405, arg1_1, 16, 960, stream=stream0)
        buf2406 = buf1547; del buf1547  # reuse
        # Topologically Sorted Source Nodes: [mul_2050, sum_766], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf2405, arg291_1, buf2406, 16384, 120, stream=stream0)
        del arg291_1
        del buf2405
        buf2407 = reinterpret_tensor(buf2395, (16, 1, 128), (128, 128, 1), 0); del buf2395  # reuse
        # Topologically Sorted Source Nodes: [mul_2050, sum_766], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf2406, buf2407, 2048, 8, stream=stream0)
        del buf2406
        buf2408 = buf2382; del buf2382  # reuse
        # Topologically Sorted Source Nodes: [mul_2051, sum_767], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_17.run(buf2407, arg292_1, buf2408, 2048, 2048, stream=stream0)
        del arg292_1
        buf2410 = reinterpret_tensor(buf2407, (1, 2048), (2048, 1), 0); del buf2407  # reuse
        # Topologically Sorted Source Nodes: [embedding_3, add_776, convert_element_type_2728, pow_343, mean_342, convert_element_type_2730], Original ATen: [aten.embedding, aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_embedding_mean_pow_18.run(buf2380, arg282_1, buf2408, arg293_1, buf2410, 1, 2048, stream=stream0)
        del arg293_1
        buf2411 = buf2376; del buf2376  # reuse
        # Topologically Sorted Source Nodes: [mul_2054, sum_768], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_19.run(buf2410, arg294_1, buf2411, 12288, 2048, stream=stream0)
        del arg294_1
        buf2412 = buf2410; del buf2410  # reuse
        # Topologically Sorted Source Nodes: [mul_2057, sum_769], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf2411, arg295_1, buf2412, 2048, 6144, stream=stream0)
        del arg295_1
        buf2414 = buf2374; del buf2374  # reuse
        # Topologically Sorted Source Nodes: [embedding_3, add_776, add_778, convert_element_type_2738, pow_344, mean_343, add_779, rsqrt_343, mul_2058, convert_element_type_2739, mul_2059], Original ATen: [aten.embedding, aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_embedding_mean_mul_pow_rsqrt_21.run(buf2380, arg282_1, buf2408, buf2412, arg296_1, buf2414, 1, 2048, stream=stream0)
        del arg296_1
        buf2418 = buf2387; del buf2387  # reuse
        # Topologically Sorted Source Nodes: [mul_2063, sum_771, mul_2066, sum_772, index_put_171], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_49.run(buf2414, arg299_1, arg301_1, arg1_1, buf2418, arg303_1, 1024, 2048, stream=stream0)
        del arg299_1
        del arg301_1
        # Topologically Sorted Source Nodes: [convert_element_type_2748, pow_346, mean_345, mul_2069, cat_171, mul_2070, add_783, index_put_170], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_47.run(buf2418, arg300_1, arg1_1, buf2385, arg302_1, 8, 128, stream=stream0)
        del arg300_1
        buf2415 = buf2342; del buf2342  # reuse
        # Topologically Sorted Source Nodes: [mul_2060, sum_770], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_22.run(buf2414, arg297_1, buf2415, 2048, 2048, stream=stream0)
        del arg297_1
        buf2426 = reinterpret_tensor(buf2316, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf2316  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_2743, pow_345, mean_344, mul_2067, cat_170, mul_2068, add_782, convert_element_type_2753, mul_2071], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf2415, arg298_1, buf2385, buf2426, 16, 128, stream=stream0)
        del arg298_1
        buf2427 = buf1568; del buf1568  # reuse
        # Topologically Sorted Source Nodes: [mul_2073, sum_773], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf2426, arg302_1, buf2427, 15360, 128, stream=stream0)
        del arg302_1
        buf2436 = reinterpret_tensor(buf2427, (1, 16, 1, 960), (15360, 960, 15360, 1), 0); del buf2427  # reuse
        # Topologically Sorted Source Nodes: [full_default_429, full_default_427, where_170, add_784, eq_85, logical_not_170, any_86, logical_not_171, full_default_431, , sub, exp, div_85, where_171], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_48.run(buf2436, arg1_1, 16, 960, stream=stream0)
        buf2437 = buf1578; del buf1578  # reuse
        # Topologically Sorted Source Nodes: [mul_2074, sum_775], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf2436, arg303_1, buf2437, 16384, 120, stream=stream0)
        del arg303_1
        del buf2436
        buf2438 = reinterpret_tensor(buf2426, (16, 1, 128), (128, 128, 1), 0); del buf2426  # reuse
        # Topologically Sorted Source Nodes: [mul_2074, sum_775], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf2437, buf2438, 2048, 8, stream=stream0)
        del buf2437
        buf2440 = buf2414; del buf2414  # reuse
        # Topologically Sorted Source Nodes: [embedding_3, add_776, add_778, mul_2075, sum_776, add_785], Original ATen: [aten.embedding, aten.add, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_embedding_mul_sum_24.run(buf2438, arg304_1, buf2380, arg282_1, buf2408, buf2412, buf2440, 2048, 2048, stream=stream0)
        del arg304_1
        del buf2380
        buf2441 = buf2381; del buf2381  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_2760, pow_347, mean_346], Original ATen: [prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_mean_pow_25.run(buf2440, buf2441, 1, 2048, stream=stream0)
        buf2442 = buf2411; del buf2411  # reuse
        # Topologically Sorted Source Nodes: [mul_2078, sum_777], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_26.run(arg305_1, buf2440, buf2441, arg306_1, buf2442, 12288, 2048, stream=stream0)
        del arg305_1
        del arg306_1
        buf2443 = reinterpret_tensor(buf2438, (1, 2048), (2048, 1), 0); del buf2438  # reuse
        # Topologically Sorted Source Nodes: [mul_2081, sum_778], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf2442, arg307_1, buf2443, 2048, 6144, stream=stream0)
        del arg307_1
        buf2444 = buf2441; del buf2441  # reuse
        # Topologically Sorted Source Nodes: [add_787, convert_element_type_2770, pow_348, mean_347], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_27.run(buf2440, buf2443, buf2444, 1, 2048, stream=stream0)
        buf2448 = buf2418; del buf2418  # reuse
        # Topologically Sorted Source Nodes: [mul_2087, sum_780, mul_2090, sum_781, index_put_173], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_50.run(arg308_1, buf2440, buf2443, buf2444, arg311_1, arg313_1, arg1_1, buf2448, arg315_1, 1024, 2048, stream=stream0)
        del arg311_1
        del arg313_1
        # Topologically Sorted Source Nodes: [convert_element_type_2780, pow_350, mean_349, mul_2093, cat_173, mul_2094, add_792, index_put_172], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_47.run(buf2448, arg312_1, arg1_1, buf2385, arg314_1, 8, 128, stream=stream0)
        del arg312_1
        buf2445 = buf2412; del buf2412  # reuse
        # Topologically Sorted Source Nodes: [mul_2084, sum_779], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_28.run(arg308_1, buf2440, buf2443, buf2444, arg309_1, buf2445, 2048, 2048, stream=stream0)
        del arg308_1
        del arg309_1
        buf2456 = reinterpret_tensor(buf2408, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf2408  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_2775, pow_349, mean_348, mul_2091, cat_172, mul_2092, add_791, convert_element_type_2785, mul_2095], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf2445, arg310_1, buf2385, buf2456, 16, 128, stream=stream0)
        del arg310_1
        buf2457 = buf1598; del buf1598  # reuse
        # Topologically Sorted Source Nodes: [mul_2097, sum_782], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf2456, arg314_1, buf2457, 15360, 128, stream=stream0)
        del arg314_1
        buf2466 = reinterpret_tensor(buf2457, (1, 16, 1, 960), (15360, 960, 15360, 1), 0); del buf2457  # reuse
        # Topologically Sorted Source Nodes: [full_default_435, full_default_433, where_172, add_793, eq_86, logical_not_172, any_87, logical_not_173, full_default_437, , sub, exp, div_86, where_173], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_48.run(buf2466, arg1_1, 16, 960, stream=stream0)
        buf2467 = buf1608; del buf1608  # reuse
        # Topologically Sorted Source Nodes: [mul_2098, sum_784], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf2466, arg315_1, buf2467, 16384, 120, stream=stream0)
        del arg315_1
        del buf2466
        buf2468 = reinterpret_tensor(buf2456, (16, 1, 128), (128, 128, 1), 0); del buf2456  # reuse
        # Topologically Sorted Source Nodes: [mul_2098, sum_784], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf2467, buf2468, 2048, 8, stream=stream0)
        del buf2467
        buf2469 = buf2445; del buf2445  # reuse
        # Topologically Sorted Source Nodes: [mul_2099, sum_785], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_17.run(buf2468, arg316_1, buf2469, 2048, 2048, stream=stream0)
        del arg316_1
        buf2471 = reinterpret_tensor(buf2468, (1, 2048), (2048, 1), 0); del buf2468  # reuse
        # Topologically Sorted Source Nodes: [add_787, add_794, convert_element_type_2792, pow_351, mean_350, convert_element_type_2794], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_30.run(buf2440, buf2443, buf2469, arg317_1, buf2471, 1, 2048, stream=stream0)
        del arg317_1
        buf2472 = buf2442; del buf2442  # reuse
        # Topologically Sorted Source Nodes: [mul_2102, sum_786], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_19.run(buf2471, arg318_1, buf2472, 12288, 2048, stream=stream0)
        del arg318_1
        buf2473 = buf2471; del buf2471  # reuse
        # Topologically Sorted Source Nodes: [mul_2105, sum_787], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf2472, arg319_1, buf2473, 2048, 6144, stream=stream0)
        del arg319_1
        buf2475 = buf2348; del buf2348  # reuse
        # Topologically Sorted Source Nodes: [add_787, add_794, add_796, convert_element_type_2802, pow_352, mean_351, add_797, rsqrt_351, mul_2106, convert_element_type_2803, mul_2107], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_mul_pow_rsqrt_31.run(buf2440, buf2443, buf2469, buf2473, arg320_1, buf2475, 1, 2048, stream=stream0)
        del arg320_1
        buf2479 = buf2448; del buf2448  # reuse
        # Topologically Sorted Source Nodes: [mul_2111, sum_789, mul_2114, sum_790, index_put_175], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_49.run(buf2475, arg323_1, arg325_1, arg1_1, buf2479, arg327_1, 1024, 2048, stream=stream0)
        del arg323_1
        del arg325_1
        # Topologically Sorted Source Nodes: [convert_element_type_2812, pow_354, mean_353, mul_2117, cat_175, mul_2118, add_801, index_put_174], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_47.run(buf2479, arg324_1, arg1_1, buf2385, arg326_1, 8, 128, stream=stream0)
        del arg324_1
        buf2476 = buf2415; del buf2415  # reuse
        # Topologically Sorted Source Nodes: [mul_2108, sum_788], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_22.run(buf2475, arg321_1, buf2476, 2048, 2048, stream=stream0)
        del arg321_1
        buf2487 = reinterpret_tensor(buf2349, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf2349  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_2807, pow_353, mean_352, mul_2115, cat_174, mul_2116, add_800, convert_element_type_2817, mul_2119], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf2476, arg322_1, buf2385, buf2487, 16, 128, stream=stream0)
        del arg322_1
        buf2488 = buf1629; del buf1629  # reuse
        # Topologically Sorted Source Nodes: [mul_2121, sum_791], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf2487, arg326_1, buf2488, 15360, 128, stream=stream0)
        del arg326_1
        buf2497 = reinterpret_tensor(buf2488, (1, 16, 1, 960), (15360, 960, 15360, 1), 0); del buf2488  # reuse
        # Topologically Sorted Source Nodes: [full_default_441, full_default_439, where_174, add_802, eq_87, logical_not_174, any_88, logical_not_175, full_default_443, , sub, exp, div_87, where_175], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_48.run(buf2497, arg1_1, 16, 960, stream=stream0)
        buf2498 = buf1639; del buf1639  # reuse
        # Topologically Sorted Source Nodes: [mul_2122, sum_793], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf2497, arg327_1, buf2498, 16384, 120, stream=stream0)
        del arg327_1
        del buf2497
        buf2499 = reinterpret_tensor(buf2487, (16, 1, 128), (128, 128, 1), 0); del buf2487  # reuse
        # Topologically Sorted Source Nodes: [mul_2122, sum_793], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf2498, buf2499, 2048, 8, stream=stream0)
        del buf2498
        buf2501 = buf2440; del buf2440  # reuse
        # Topologically Sorted Source Nodes: [add_787, add_794, add_796, mul_2123, sum_794, add_803], Original ATen: [aten.add, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_sum_32.run(buf2501, buf2499, arg328_1, buf2443, buf2469, buf2473, 2048, 2048, stream=stream0)
        del arg328_1
        buf2502 = buf2444; del buf2444  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_2824, pow_355, mean_354], Original ATen: [prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_mean_pow_25.run(buf2501, buf2502, 1, 2048, stream=stream0)
        buf2503 = buf2472; del buf2472  # reuse
        # Topologically Sorted Source Nodes: [mul_2126, sum_795], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_26.run(arg329_1, buf2501, buf2502, arg330_1, buf2503, 12288, 2048, stream=stream0)
        del arg329_1
        del arg330_1
        buf2504 = reinterpret_tensor(buf2499, (1, 2048), (2048, 1), 0); del buf2499  # reuse
        # Topologically Sorted Source Nodes: [mul_2129, sum_796], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf2503, arg331_1, buf2504, 2048, 6144, stream=stream0)
        del arg331_1
        buf2505 = buf2502; del buf2502  # reuse
        # Topologically Sorted Source Nodes: [add_805, convert_element_type_2834, pow_356, mean_355], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_27.run(buf2501, buf2504, buf2505, 1, 2048, stream=stream0)
        buf2509 = buf2479; del buf2479  # reuse
        # Topologically Sorted Source Nodes: [mul_2135, sum_798, mul_2138, sum_799, index_put_177], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_50.run(arg332_1, buf2501, buf2504, buf2505, arg335_1, arg337_1, arg1_1, buf2509, arg339_1, 1024, 2048, stream=stream0)
        del arg335_1
        del arg337_1
        # Topologically Sorted Source Nodes: [convert_element_type_2844, pow_358, mean_357, mul_2141, cat_177, mul_2142, add_810, index_put_176], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_47.run(buf2509, arg336_1, arg1_1, buf2385, arg338_1, 8, 128, stream=stream0)
        del arg336_1
        buf2506 = buf2473; del buf2473  # reuse
        # Topologically Sorted Source Nodes: [mul_2132, sum_797], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_28.run(arg332_1, buf2501, buf2504, buf2505, arg333_1, buf2506, 2048, 2048, stream=stream0)
        del arg332_1
        del arg333_1
        buf2517 = reinterpret_tensor(buf2469, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf2469  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_2839, pow_357, mean_356, mul_2139, cat_176, mul_2140, add_809, convert_element_type_2849, mul_2143], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf2506, arg334_1, buf2385, buf2517, 16, 128, stream=stream0)
        del arg334_1
        buf2518 = buf1659; del buf1659  # reuse
        # Topologically Sorted Source Nodes: [mul_2145, sum_800], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf2517, arg338_1, buf2518, 15360, 128, stream=stream0)
        del arg338_1
        buf2527 = reinterpret_tensor(buf2518, (1, 16, 1, 960), (15360, 960, 15360, 1), 0); del buf2518  # reuse
        # Topologically Sorted Source Nodes: [full_default_447, full_default_445, where_176, add_811, eq_88, logical_not_176, any_89, logical_not_177, full_default_449, , sub, exp, div_88, where_177], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_48.run(buf2527, arg1_1, 16, 960, stream=stream0)
        buf2528 = buf1669; del buf1669  # reuse
        # Topologically Sorted Source Nodes: [mul_2146, sum_802], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf2527, arg339_1, buf2528, 16384, 120, stream=stream0)
        del arg339_1
        del buf2527
        buf2529 = reinterpret_tensor(buf2517, (16, 1, 128), (128, 128, 1), 0); del buf2517  # reuse
        # Topologically Sorted Source Nodes: [mul_2146, sum_802], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf2528, buf2529, 2048, 8, stream=stream0)
        del buf2528
        buf2530 = buf2506; del buf2506  # reuse
        # Topologically Sorted Source Nodes: [mul_2147, sum_803], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_17.run(buf2529, arg340_1, buf2530, 2048, 2048, stream=stream0)
        del arg340_1
        buf2532 = reinterpret_tensor(buf2529, (1, 2048), (2048, 1), 0); del buf2529  # reuse
        # Topologically Sorted Source Nodes: [add_805, add_812, convert_element_type_2856, pow_359, mean_358, convert_element_type_2858], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_30.run(buf2501, buf2504, buf2530, arg341_1, buf2532, 1, 2048, stream=stream0)
        del arg341_1
        buf2533 = buf2503; del buf2503  # reuse
        # Topologically Sorted Source Nodes: [mul_2150, sum_804], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_19.run(buf2532, arg2_1, buf2533, 12288, 2048, stream=stream0)
        del arg2_1
        buf2534 = buf2532; del buf2532  # reuse
        # Topologically Sorted Source Nodes: [mul_2153, sum_805], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf2533, arg3_1, buf2534, 2048, 6144, stream=stream0)
        del arg3_1
        buf2536 = buf2475; del buf2475  # reuse
        # Topologically Sorted Source Nodes: [add_805, add_812, add_814, convert_element_type_2866, pow_360, mean_359, add_815, rsqrt_359, mul_2154, convert_element_type_2867, mul_2155], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_mul_pow_rsqrt_31.run(buf2501, buf2504, buf2530, buf2534, arg4_1, buf2536, 1, 2048, stream=stream0)
        del arg4_1
        buf2540 = buf2509; del buf2509  # reuse
        # Topologically Sorted Source Nodes: [mul_2159, sum_807, mul_2162, sum_808, index_put_179], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_49.run(buf2536, arg7_1, arg9_1, arg1_1, buf2540, arg11_1, 1024, 2048, stream=stream0)
        del arg7_1
        del arg9_1
        # Topologically Sorted Source Nodes: [convert_element_type_2876, pow_362, mean_361, mul_2165, cat_179, mul_2166, add_819, index_put_178], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_47.run(buf2540, arg8_1, arg1_1, buf2385, arg10_1, 8, 128, stream=stream0)
        del arg8_1
        buf2537 = buf2443; del buf2443  # reuse
        # Topologically Sorted Source Nodes: [mul_2156, sum_806], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_22.run(buf2536, arg5_1, buf2537, 2048, 2048, stream=stream0)
        del arg5_1
        buf2548 = reinterpret_tensor(buf2476, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf2476  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_2871, pow_361, mean_360, mul_2163, cat_178, mul_2164, add_818, convert_element_type_2881, mul_2167], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf2537, arg6_1, buf2385, buf2548, 16, 128, stream=stream0)
        del arg6_1
        buf2549 = buf1690; del buf1690  # reuse
        # Topologically Sorted Source Nodes: [mul_2169, sum_809], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf2548, arg10_1, buf2549, 15360, 128, stream=stream0)
        del arg10_1
        buf2558 = reinterpret_tensor(buf2549, (1, 16, 1, 960), (15360, 960, 15360, 1), 0); del buf2549  # reuse
        # Topologically Sorted Source Nodes: [full_default_453, full_default_451, where_178, add_820, eq_89, logical_not_178, any_90, logical_not_179, full_default_455, , sub, exp, div_89, where_179], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_48.run(buf2558, arg1_1, 16, 960, stream=stream0)
        buf2559 = buf1700; del buf1700  # reuse
        # Topologically Sorted Source Nodes: [mul_2170, sum_811], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf2558, arg11_1, buf2559, 16384, 120, stream=stream0)
        del arg11_1
        del buf2558
        buf2560 = reinterpret_tensor(buf2548, (16, 1, 128), (128, 128, 1), 0); del buf2548  # reuse
        # Topologically Sorted Source Nodes: [mul_2170, sum_811], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf2559, buf2560, 2048, 8, stream=stream0)
        del buf2559
        buf2562 = buf2501; del buf2501  # reuse
        # Topologically Sorted Source Nodes: [add_805, add_812, add_814, mul_2171, sum_812, add_821], Original ATen: [aten.add, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_sum_32.run(buf2562, buf2560, arg12_1, buf2504, buf2530, buf2534, 2048, 2048, stream=stream0)
        del arg12_1
        buf2563 = buf2505; del buf2505  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_2888, pow_363, mean_362], Original ATen: [prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_mean_pow_25.run(buf2562, buf2563, 1, 2048, stream=stream0)
        buf2564 = buf2533; del buf2533  # reuse
        # Topologically Sorted Source Nodes: [mul_2174, sum_813], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_26.run(arg13_1, buf2562, buf2563, arg14_1, buf2564, 12288, 2048, stream=stream0)
        del arg13_1
        del arg14_1
        buf2565 = reinterpret_tensor(buf2560, (1, 2048), (2048, 1), 0); del buf2560  # reuse
        # Topologically Sorted Source Nodes: [mul_2177, sum_814], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf2564, arg15_1, buf2565, 2048, 6144, stream=stream0)
        del arg15_1
        buf2566 = buf2563; del buf2563  # reuse
        # Topologically Sorted Source Nodes: [add_823, convert_element_type_2898, pow_364, mean_363], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_27.run(buf2562, buf2565, buf2566, 1, 2048, stream=stream0)
        buf2570 = buf2540; del buf2540  # reuse
        # Topologically Sorted Source Nodes: [mul_2183, sum_816, mul_2186, sum_817, index_put_181], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_50.run(arg16_1, buf2562, buf2565, buf2566, arg19_1, arg21_1, arg1_1, buf2570, arg23_1, 1024, 2048, stream=stream0)
        del arg19_1
        del arg21_1
        # Topologically Sorted Source Nodes: [convert_element_type_2908, pow_366, mean_365, mul_2189, cat_181, mul_2190, add_828, index_put_180], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_47.run(buf2570, arg20_1, arg1_1, buf2385, arg22_1, 8, 128, stream=stream0)
        del arg20_1
        buf2567 = buf2534; del buf2534  # reuse
        # Topologically Sorted Source Nodes: [mul_2180, sum_815], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_28.run(arg16_1, buf2562, buf2565, buf2566, arg17_1, buf2567, 2048, 2048, stream=stream0)
        del arg16_1
        del arg17_1
        buf2578 = reinterpret_tensor(buf2530, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf2530  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_2903, pow_365, mean_364, mul_2187, cat_180, mul_2188, add_827, convert_element_type_2913, mul_2191], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf2567, arg18_1, buf2385, buf2578, 16, 128, stream=stream0)
        del arg18_1
        buf2579 = buf1720; del buf1720  # reuse
        # Topologically Sorted Source Nodes: [mul_2193, sum_818], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf2578, arg22_1, buf2579, 15360, 128, stream=stream0)
        del arg22_1
        buf2588 = reinterpret_tensor(buf2579, (1, 16, 1, 960), (15360, 960, 15360, 1), 0); del buf2579  # reuse
        # Topologically Sorted Source Nodes: [full_default_459, full_default_457, where_180, add_829, eq_90, logical_not_180, any_91, logical_not_181, full_default_461, , sub, exp, div_90, where_181], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_48.run(buf2588, arg1_1, 16, 960, stream=stream0)
        buf2589 = buf1730; del buf1730  # reuse
        # Topologically Sorted Source Nodes: [mul_2194, sum_820], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf2588, arg23_1, buf2589, 16384, 120, stream=stream0)
        del arg23_1
        del buf2588
        buf2590 = reinterpret_tensor(buf2578, (16, 1, 128), (128, 128, 1), 0); del buf2578  # reuse
        # Topologically Sorted Source Nodes: [mul_2194, sum_820], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf2589, buf2590, 2048, 8, stream=stream0)
        del buf2589
        buf2591 = buf2567; del buf2567  # reuse
        # Topologically Sorted Source Nodes: [mul_2195, sum_821], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_17.run(buf2590, arg24_1, buf2591, 2048, 2048, stream=stream0)
        del arg24_1
        buf2593 = reinterpret_tensor(buf2590, (1, 2048), (2048, 1), 0); del buf2590  # reuse
        # Topologically Sorted Source Nodes: [add_823, add_830, convert_element_type_2920, pow_367, mean_366, convert_element_type_2922], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_30.run(buf2562, buf2565, buf2591, arg25_1, buf2593, 1, 2048, stream=stream0)
        del arg25_1
        buf2594 = buf2564; del buf2564  # reuse
        # Topologically Sorted Source Nodes: [mul_2198, sum_822], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_19.run(buf2593, arg26_1, buf2594, 12288, 2048, stream=stream0)
        del arg26_1
        buf2595 = buf2593; del buf2593  # reuse
        # Topologically Sorted Source Nodes: [mul_2201, sum_823], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf2594, arg27_1, buf2595, 2048, 6144, stream=stream0)
        del arg27_1
        buf2597 = buf2536; del buf2536  # reuse
        # Topologically Sorted Source Nodes: [add_823, add_830, add_832, convert_element_type_2930, pow_368, mean_367, add_833, rsqrt_367, mul_2202, convert_element_type_2931, mul_2203], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_mul_pow_rsqrt_31.run(buf2562, buf2565, buf2591, buf2595, arg28_1, buf2597, 1, 2048, stream=stream0)
        del arg28_1
        buf2601 = buf2570; del buf2570  # reuse
        # Topologically Sorted Source Nodes: [mul_2207, sum_825, mul_2210, sum_826, index_put_183], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_49.run(buf2597, arg31_1, arg33_1, arg1_1, buf2601, arg35_1, 1024, 2048, stream=stream0)
        del arg31_1
        del arg33_1
        # Topologically Sorted Source Nodes: [convert_element_type_2940, pow_370, mean_369, mul_2213, cat_183, mul_2214, add_837, index_put_182], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_47.run(buf2601, arg32_1, arg1_1, buf2385, arg34_1, 8, 128, stream=stream0)
        del arg32_1
        buf2598 = buf2504; del buf2504  # reuse
        # Topologically Sorted Source Nodes: [mul_2204, sum_824], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_22.run(buf2597, arg29_1, buf2598, 2048, 2048, stream=stream0)
        del arg29_1
        buf2609 = reinterpret_tensor(buf2537, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf2537  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_2935, pow_369, mean_368, mul_2211, cat_182, mul_2212, add_836, convert_element_type_2945, mul_2215], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf2598, arg30_1, buf2385, buf2609, 16, 128, stream=stream0)
        del arg30_1
        buf2610 = buf1751; del buf1751  # reuse
        # Topologically Sorted Source Nodes: [mul_2217, sum_827], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf2609, arg34_1, buf2610, 15360, 128, stream=stream0)
        del arg34_1
        buf2619 = reinterpret_tensor(buf2610, (1, 16, 1, 960), (15360, 960, 15360, 1), 0); del buf2610  # reuse
        # Topologically Sorted Source Nodes: [full_default_465, full_default_463, where_182, add_838, eq_91, logical_not_182, any_92, logical_not_183, full_default_467, , sub, exp, div_91, where_183], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_48.run(buf2619, arg1_1, 16, 960, stream=stream0)
        buf2620 = buf1761; del buf1761  # reuse
        # Topologically Sorted Source Nodes: [mul_2218, sum_829], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf2619, arg35_1, buf2620, 16384, 120, stream=stream0)
        del arg35_1
        del buf2619
        buf2621 = reinterpret_tensor(buf2609, (16, 1, 128), (128, 128, 1), 0); del buf2609  # reuse
        # Topologically Sorted Source Nodes: [mul_2218, sum_829], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf2620, buf2621, 2048, 8, stream=stream0)
        del buf2620
        buf2623 = buf2562; del buf2562  # reuse
        # Topologically Sorted Source Nodes: [add_823, add_830, add_832, mul_2219, sum_830, add_839], Original ATen: [aten.add, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_sum_32.run(buf2623, buf2621, arg36_1, buf2565, buf2591, buf2595, 2048, 2048, stream=stream0)
        del arg36_1
        buf2624 = buf2566; del buf2566  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_2952, pow_371, mean_370], Original ATen: [prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_mean_pow_25.run(buf2623, buf2624, 1, 2048, stream=stream0)
        buf2625 = buf2594; del buf2594  # reuse
        # Topologically Sorted Source Nodes: [mul_2222, sum_831], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_26.run(arg37_1, buf2623, buf2624, arg38_1, buf2625, 12288, 2048, stream=stream0)
        del arg37_1
        del arg38_1
        buf2626 = reinterpret_tensor(buf2621, (1, 2048), (2048, 1), 0); del buf2621  # reuse
        # Topologically Sorted Source Nodes: [mul_2225, sum_832], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf2625, arg39_1, buf2626, 2048, 6144, stream=stream0)
        del arg39_1
        buf2627 = buf2624; del buf2624  # reuse
        # Topologically Sorted Source Nodes: [add_841, convert_element_type_2962, pow_372, mean_371], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_27.run(buf2623, buf2626, buf2627, 1, 2048, stream=stream0)
        buf2631 = buf2601; del buf2601  # reuse
        # Topologically Sorted Source Nodes: [mul_2231, sum_834, mul_2234, sum_835, index_put_185], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_50.run(arg40_1, buf2623, buf2626, buf2627, arg43_1, arg45_1, arg1_1, buf2631, arg47_1, 1024, 2048, stream=stream0)
        del arg43_1
        del arg45_1
        # Topologically Sorted Source Nodes: [convert_element_type_2972, pow_374, mean_373, mul_2237, cat_185, mul_2238, add_846, index_put_184], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_47.run(buf2631, arg44_1, arg1_1, buf2385, arg46_1, 8, 128, stream=stream0)
        del arg44_1
        buf2628 = buf2595; del buf2595  # reuse
        # Topologically Sorted Source Nodes: [mul_2228, sum_833], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_28.run(arg40_1, buf2623, buf2626, buf2627, arg41_1, buf2628, 2048, 2048, stream=stream0)
        del arg40_1
        del arg41_1
        buf2639 = reinterpret_tensor(buf2591, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf2591  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_2967, pow_373, mean_372, mul_2235, cat_184, mul_2236, add_845, convert_element_type_2977, mul_2239], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf2628, arg42_1, buf2385, buf2639, 16, 128, stream=stream0)
        del arg42_1
        buf2640 = buf1781; del buf1781  # reuse
        # Topologically Sorted Source Nodes: [mul_2241, sum_836], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf2639, arg46_1, buf2640, 15360, 128, stream=stream0)
        del arg46_1
        buf2649 = reinterpret_tensor(buf2640, (1, 16, 1, 960), (15360, 960, 15360, 1), 0); del buf2640  # reuse
        # Topologically Sorted Source Nodes: [full_default_471, full_default_469, where_184, add_847, eq_92, logical_not_184, any_93, logical_not_185, full_default_473, , sub, exp, div_92, where_185], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_48.run(buf2649, arg1_1, 16, 960, stream=stream0)
        buf2650 = buf1791; del buf1791  # reuse
        # Topologically Sorted Source Nodes: [mul_2242, sum_838], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf2649, arg47_1, buf2650, 16384, 120, stream=stream0)
        del arg47_1
        del buf2649
        buf2651 = reinterpret_tensor(buf2639, (16, 1, 128), (128, 128, 1), 0); del buf2639  # reuse
        # Topologically Sorted Source Nodes: [mul_2242, sum_838], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf2650, buf2651, 2048, 8, stream=stream0)
        del buf2650
        buf2652 = buf2628; del buf2628  # reuse
        # Topologically Sorted Source Nodes: [mul_2243, sum_839], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_17.run(buf2651, arg48_1, buf2652, 2048, 2048, stream=stream0)
        del arg48_1
        buf2654 = reinterpret_tensor(buf2651, (1, 2048), (2048, 1), 0); del buf2651  # reuse
        # Topologically Sorted Source Nodes: [add_841, add_848, convert_element_type_2984, pow_375, mean_374, convert_element_type_2986], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_30.run(buf2623, buf2626, buf2652, arg49_1, buf2654, 1, 2048, stream=stream0)
        del arg49_1
        buf2655 = buf2625; del buf2625  # reuse
        # Topologically Sorted Source Nodes: [mul_2246, sum_840], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_19.run(buf2654, arg50_1, buf2655, 12288, 2048, stream=stream0)
        del arg50_1
        buf2656 = buf2654; del buf2654  # reuse
        # Topologically Sorted Source Nodes: [mul_2249, sum_841], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf2655, arg51_1, buf2656, 2048, 6144, stream=stream0)
        del arg51_1
        buf2658 = buf2597; del buf2597  # reuse
        # Topologically Sorted Source Nodes: [add_841, add_848, add_850, convert_element_type_2994, pow_376, mean_375, add_851, rsqrt_375, mul_2250, convert_element_type_2995, mul_2251], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_mul_pow_rsqrt_31.run(buf2623, buf2626, buf2652, buf2656, arg52_1, buf2658, 1, 2048, stream=stream0)
        del arg52_1
        buf2662 = buf2631; del buf2631  # reuse
        # Topologically Sorted Source Nodes: [mul_2255, sum_843, mul_2258, sum_844, index_put_187], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_49.run(buf2658, arg55_1, arg57_1, arg1_1, buf2662, arg59_1, 1024, 2048, stream=stream0)
        del arg55_1
        del arg57_1
        # Topologically Sorted Source Nodes: [convert_element_type_3004, pow_378, mean_377, mul_2261, cat_187, mul_2262, add_855, index_put_186], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_47.run(buf2662, arg56_1, arg1_1, buf2385, arg58_1, 8, 128, stream=stream0)
        del arg56_1
        buf2659 = buf2565; del buf2565  # reuse
        # Topologically Sorted Source Nodes: [mul_2252, sum_842], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_22.run(buf2658, arg53_1, buf2659, 2048, 2048, stream=stream0)
        del arg53_1
        buf2670 = reinterpret_tensor(buf2598, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf2598  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_2999, pow_377, mean_376, mul_2259, cat_186, mul_2260, add_854, convert_element_type_3009, mul_2263], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf2659, arg54_1, buf2385, buf2670, 16, 128, stream=stream0)
        del arg54_1
        buf2671 = buf1812; del buf1812  # reuse
        # Topologically Sorted Source Nodes: [mul_2265, sum_845], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf2670, arg58_1, buf2671, 15360, 128, stream=stream0)
        del arg58_1
        buf2680 = reinterpret_tensor(buf2671, (1, 16, 1, 960), (15360, 960, 15360, 1), 0); del buf2671  # reuse
        # Topologically Sorted Source Nodes: [full_default_477, full_default_475, where_186, add_856, eq_93, logical_not_186, any_94, logical_not_187, full_default_479, , sub, exp, div_93, where_187], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_48.run(buf2680, arg1_1, 16, 960, stream=stream0)
        buf2681 = buf1822; del buf1822  # reuse
        # Topologically Sorted Source Nodes: [mul_2266, sum_847], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf2680, arg59_1, buf2681, 16384, 120, stream=stream0)
        del arg59_1
        del buf2680
        buf2682 = reinterpret_tensor(buf2670, (16, 1, 128), (128, 128, 1), 0); del buf2670  # reuse
        # Topologically Sorted Source Nodes: [mul_2266, sum_847], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf2681, buf2682, 2048, 8, stream=stream0)
        del buf2681
        buf2684 = buf2623; del buf2623  # reuse
        # Topologically Sorted Source Nodes: [add_841, add_848, add_850, mul_2267, sum_848, add_857], Original ATen: [aten.add, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_sum_32.run(buf2684, buf2682, arg60_1, buf2626, buf2652, buf2656, 2048, 2048, stream=stream0)
        del arg60_1
        buf2685 = buf2627; del buf2627  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_3016, pow_379, mean_378], Original ATen: [prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_mean_pow_25.run(buf2684, buf2685, 1, 2048, stream=stream0)
        buf2686 = buf2655; del buf2655  # reuse
        # Topologically Sorted Source Nodes: [mul_2270, sum_849], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_26.run(arg61_1, buf2684, buf2685, arg62_1, buf2686, 12288, 2048, stream=stream0)
        del arg61_1
        del arg62_1
        buf2687 = reinterpret_tensor(buf2682, (1, 2048), (2048, 1), 0); del buf2682  # reuse
        # Topologically Sorted Source Nodes: [mul_2273, sum_850], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf2686, arg63_1, buf2687, 2048, 6144, stream=stream0)
        del arg63_1
        buf2688 = buf2685; del buf2685  # reuse
        # Topologically Sorted Source Nodes: [add_859, convert_element_type_3026, pow_380, mean_379], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_27.run(buf2684, buf2687, buf2688, 1, 2048, stream=stream0)
        buf2692 = buf2662; del buf2662  # reuse
        # Topologically Sorted Source Nodes: [mul_2279, sum_852, mul_2282, sum_853, index_put_189], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_50.run(arg64_1, buf2684, buf2687, buf2688, arg67_1, arg69_1, arg1_1, buf2692, arg71_1, 1024, 2048, stream=stream0)
        del arg67_1
        del arg69_1
        # Topologically Sorted Source Nodes: [convert_element_type_3036, pow_382, mean_381, mul_2285, cat_189, mul_2286, add_864, index_put_188], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_47.run(buf2692, arg68_1, arg1_1, buf2385, arg70_1, 8, 128, stream=stream0)
        del arg68_1
        buf2689 = buf2656; del buf2656  # reuse
        # Topologically Sorted Source Nodes: [mul_2276, sum_851], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_28.run(arg64_1, buf2684, buf2687, buf2688, arg65_1, buf2689, 2048, 2048, stream=stream0)
        del arg64_1
        del arg65_1
        buf2700 = reinterpret_tensor(buf2652, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf2652  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_3031, pow_381, mean_380, mul_2283, cat_188, mul_2284, add_863, convert_element_type_3041, mul_2287], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf2689, arg66_1, buf2385, buf2700, 16, 128, stream=stream0)
        del arg66_1
        buf2701 = buf1842; del buf1842  # reuse
        # Topologically Sorted Source Nodes: [mul_2289, sum_854], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf2700, arg70_1, buf2701, 15360, 128, stream=stream0)
        del arg70_1
        buf2710 = reinterpret_tensor(buf2701, (1, 16, 1, 960), (15360, 960, 15360, 1), 0); del buf2701  # reuse
        # Topologically Sorted Source Nodes: [full_default_483, full_default_481, where_188, add_865, eq_94, logical_not_188, any_95, logical_not_189, full_default_485, , sub, exp, div_94, where_189], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_48.run(buf2710, arg1_1, 16, 960, stream=stream0)
        buf2711 = buf1852; del buf1852  # reuse
        # Topologically Sorted Source Nodes: [mul_2290, sum_856], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf2710, arg71_1, buf2711, 16384, 120, stream=stream0)
        del arg71_1
        del buf2710
        buf2712 = reinterpret_tensor(buf2700, (16, 1, 128), (128, 128, 1), 0); del buf2700  # reuse
        # Topologically Sorted Source Nodes: [mul_2290, sum_856], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf2711, buf2712, 2048, 8, stream=stream0)
        del buf2711
        buf2713 = buf2689; del buf2689  # reuse
        # Topologically Sorted Source Nodes: [mul_2291, sum_857], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_17.run(buf2712, arg72_1, buf2713, 2048, 2048, stream=stream0)
        del arg72_1
        buf2715 = reinterpret_tensor(buf2712, (1, 2048), (2048, 1), 0); del buf2712  # reuse
        # Topologically Sorted Source Nodes: [add_859, add_866, convert_element_type_3048, pow_383, mean_382, convert_element_type_3050], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_30.run(buf2684, buf2687, buf2713, arg73_1, buf2715, 1, 2048, stream=stream0)
        del arg73_1
        buf2716 = buf2686; del buf2686  # reuse
        # Topologically Sorted Source Nodes: [mul_2294, sum_858], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_19.run(buf2715, arg74_1, buf2716, 12288, 2048, stream=stream0)
        del arg74_1
        buf2717 = buf2715; del buf2715  # reuse
        # Topologically Sorted Source Nodes: [mul_2297, sum_859], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf2716, arg75_1, buf2717, 2048, 6144, stream=stream0)
        del arg75_1
        buf2719 = buf2658; del buf2658  # reuse
        # Topologically Sorted Source Nodes: [add_859, add_866, add_868, convert_element_type_3058, pow_384, mean_383, add_869, rsqrt_383, mul_2298, convert_element_type_3059, mul_2299], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_mul_pow_rsqrt_31.run(buf2684, buf2687, buf2713, buf2717, arg76_1, buf2719, 1, 2048, stream=stream0)
        del arg76_1
        buf2723 = buf2692; del buf2692  # reuse
        # Topologically Sorted Source Nodes: [mul_2303, sum_861, mul_2306, sum_862, index_put_191], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_49.run(buf2719, arg79_1, arg81_1, arg1_1, buf2723, arg83_1, 1024, 2048, stream=stream0)
        del arg79_1
        del arg81_1
        # Topologically Sorted Source Nodes: [convert_element_type_3068, pow_386, mean_385, mul_2309, cat_191, mul_2310, add_873, index_put_190], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_47.run(buf2723, arg80_1, arg1_1, buf2385, arg82_1, 8, 128, stream=stream0)
        del arg80_1
        buf2720 = buf2626; del buf2626  # reuse
        # Topologically Sorted Source Nodes: [mul_2300, sum_860], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_22.run(buf2719, arg77_1, buf2720, 2048, 2048, stream=stream0)
        del arg77_1
        buf2731 = reinterpret_tensor(buf2659, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf2659  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_3063, pow_385, mean_384, mul_2307, cat_190, mul_2308, add_872, convert_element_type_3073, mul_2311], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf2720, arg78_1, buf2385, buf2731, 16, 128, stream=stream0)
        del arg78_1
        buf2732 = buf1873; del buf1873  # reuse
        # Topologically Sorted Source Nodes: [mul_2313, sum_863], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf2731, arg82_1, buf2732, 15360, 128, stream=stream0)
        del arg82_1
        buf2741 = reinterpret_tensor(buf2732, (1, 16, 1, 960), (15360, 960, 15360, 1), 0); del buf2732  # reuse
        # Topologically Sorted Source Nodes: [full_default_489, full_default_487, where_190, add_874, eq_95, logical_not_190, any_96, logical_not_191, full_default_491, , sub, exp, div_95, where_191], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_48.run(buf2741, arg1_1, 16, 960, stream=stream0)
        buf2742 = buf1883; del buf1883  # reuse
        # Topologically Sorted Source Nodes: [mul_2314, sum_865], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf2741, arg83_1, buf2742, 16384, 120, stream=stream0)
        del arg83_1
        del buf2741
        buf2743 = reinterpret_tensor(buf2731, (16, 1, 128), (128, 128, 1), 0); del buf2731  # reuse
        # Topologically Sorted Source Nodes: [mul_2314, sum_865], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf2742, buf2743, 2048, 8, stream=stream0)
        del buf2742
        buf2745 = buf2684; del buf2684  # reuse
        # Topologically Sorted Source Nodes: [add_859, add_866, add_868, mul_2315, sum_866, add_875], Original ATen: [aten.add, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_sum_32.run(buf2745, buf2743, arg84_1, buf2687, buf2713, buf2717, 2048, 2048, stream=stream0)
        del arg84_1
        buf2746 = buf2688; del buf2688  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_3080, pow_387, mean_386], Original ATen: [prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_mean_pow_25.run(buf2745, buf2746, 1, 2048, stream=stream0)
        buf2747 = buf2716; del buf2716  # reuse
        # Topologically Sorted Source Nodes: [mul_2318, sum_867], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_26.run(arg85_1, buf2745, buf2746, arg86_1, buf2747, 12288, 2048, stream=stream0)
        del arg85_1
        del arg86_1
        buf2748 = reinterpret_tensor(buf2743, (1, 2048), (2048, 1), 0); del buf2743  # reuse
        # Topologically Sorted Source Nodes: [mul_2321, sum_868], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf2747, arg87_1, buf2748, 2048, 6144, stream=stream0)
        del arg87_1
        buf2749 = buf2746; del buf2746  # reuse
        # Topologically Sorted Source Nodes: [add_877, convert_element_type_3090, pow_388, mean_387], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_27.run(buf2745, buf2748, buf2749, 1, 2048, stream=stream0)
        buf2753 = buf2723; del buf2723  # reuse
        # Topologically Sorted Source Nodes: [mul_2327, sum_870, mul_2330, sum_871, index_put_193], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_50.run(arg88_1, buf2745, buf2748, buf2749, arg91_1, arg93_1, arg1_1, buf2753, arg95_1, 1024, 2048, stream=stream0)
        del arg91_1
        del arg93_1
        # Topologically Sorted Source Nodes: [convert_element_type_3100, pow_390, mean_389, mul_2333, cat_193, mul_2334, add_882, index_put_192], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_47.run(buf2753, arg92_1, arg1_1, buf2385, arg94_1, 8, 128, stream=stream0)
        del arg92_1
        buf2750 = buf2717; del buf2717  # reuse
        # Topologically Sorted Source Nodes: [mul_2324, sum_869], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_28.run(arg88_1, buf2745, buf2748, buf2749, arg89_1, buf2750, 2048, 2048, stream=stream0)
        del arg88_1
        del arg89_1
        buf2761 = reinterpret_tensor(buf2713, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf2713  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_3095, pow_389, mean_388, mul_2331, cat_192, mul_2332, add_881, convert_element_type_3105, mul_2335], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf2750, arg90_1, buf2385, buf2761, 16, 128, stream=stream0)
        del arg90_1
        buf2762 = buf1903; del buf1903  # reuse
        # Topologically Sorted Source Nodes: [mul_2337, sum_872], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf2761, arg94_1, buf2762, 15360, 128, stream=stream0)
        del arg94_1
        buf2771 = reinterpret_tensor(buf2762, (1, 16, 1, 960), (15360, 960, 15360, 1), 0); del buf2762  # reuse
        # Topologically Sorted Source Nodes: [full_default_495, full_default_493, where_192, add_883, eq_96, logical_not_192, any_97, logical_not_193, full_default_497, , sub, exp, div_96, where_193], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_48.run(buf2771, arg1_1, 16, 960, stream=stream0)
        buf2772 = buf1913; del buf1913  # reuse
        # Topologically Sorted Source Nodes: [mul_2338, sum_874], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf2771, arg95_1, buf2772, 16384, 120, stream=stream0)
        del arg95_1
        del buf2771
        buf2773 = reinterpret_tensor(buf2761, (16, 1, 128), (128, 128, 1), 0); del buf2761  # reuse
        # Topologically Sorted Source Nodes: [mul_2338, sum_874], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf2772, buf2773, 2048, 8, stream=stream0)
        del buf2772
        buf2774 = buf2750; del buf2750  # reuse
        # Topologically Sorted Source Nodes: [mul_2339, sum_875], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_17.run(buf2773, arg96_1, buf2774, 2048, 2048, stream=stream0)
        del arg96_1
        buf2776 = reinterpret_tensor(buf2773, (1, 2048), (2048, 1), 0); del buf2773  # reuse
        # Topologically Sorted Source Nodes: [add_877, add_884, convert_element_type_3112, pow_391, mean_390, convert_element_type_3114], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_30.run(buf2745, buf2748, buf2774, arg97_1, buf2776, 1, 2048, stream=stream0)
        del arg97_1
        buf2777 = buf2747; del buf2747  # reuse
        # Topologically Sorted Source Nodes: [mul_2342, sum_876], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_19.run(buf2776, arg98_1, buf2777, 12288, 2048, stream=stream0)
        del arg98_1
        buf2778 = buf2776; del buf2776  # reuse
        # Topologically Sorted Source Nodes: [mul_2345, sum_877], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf2777, arg99_1, buf2778, 2048, 6144, stream=stream0)
        del arg99_1
        buf2780 = buf2719; del buf2719  # reuse
        # Topologically Sorted Source Nodes: [add_877, add_884, add_886, convert_element_type_3122, pow_392, mean_391, add_887, rsqrt_391, mul_2346, convert_element_type_3123, mul_2347], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_mul_pow_rsqrt_31.run(buf2745, buf2748, buf2774, buf2778, arg100_1, buf2780, 1, 2048, stream=stream0)
        del arg100_1
        buf2784 = buf2753; del buf2753  # reuse
        # Topologically Sorted Source Nodes: [mul_2351, sum_879, mul_2354, sum_880, index_put_195], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_49.run(buf2780, arg103_1, arg105_1, arg1_1, buf2784, arg107_1, 1024, 2048, stream=stream0)
        del arg103_1
        del arg105_1
        # Topologically Sorted Source Nodes: [convert_element_type_3132, pow_394, mean_393, mul_2357, cat_195, mul_2358, add_891, index_put_194], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_47.run(buf2784, arg104_1, arg1_1, buf2385, arg106_1, 8, 128, stream=stream0)
        del arg104_1
        buf2781 = buf2687; del buf2687  # reuse
        # Topologically Sorted Source Nodes: [mul_2348, sum_878], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_22.run(buf2780, arg101_1, buf2781, 2048, 2048, stream=stream0)
        del arg101_1
        buf2792 = reinterpret_tensor(buf2720, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf2720  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_3127, pow_393, mean_392, mul_2355, cat_194, mul_2356, add_890, convert_element_type_3137, mul_2359], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf2781, arg102_1, buf2385, buf2792, 16, 128, stream=stream0)
        del arg102_1
        buf2793 = buf1934; del buf1934  # reuse
        # Topologically Sorted Source Nodes: [mul_2361, sum_881], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf2792, arg106_1, buf2793, 15360, 128, stream=stream0)
        del arg106_1
        buf2802 = reinterpret_tensor(buf2793, (1, 16, 1, 960), (15360, 960, 15360, 1), 0); del buf2793  # reuse
        # Topologically Sorted Source Nodes: [full_default_501, full_default_499, where_194, add_892, eq_97, logical_not_194, any_98, logical_not_195, full_default_503, , sub, exp, div_97, where_195], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_48.run(buf2802, arg1_1, 16, 960, stream=stream0)
        buf2803 = buf1944; del buf1944  # reuse
        # Topologically Sorted Source Nodes: [mul_2362, sum_883], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf2802, arg107_1, buf2803, 16384, 120, stream=stream0)
        del arg107_1
        del buf2802
        buf2804 = reinterpret_tensor(buf2792, (16, 1, 128), (128, 128, 1), 0); del buf2792  # reuse
        # Topologically Sorted Source Nodes: [mul_2362, sum_883], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf2803, buf2804, 2048, 8, stream=stream0)
        del buf2803
        buf2806 = buf2745; del buf2745  # reuse
        # Topologically Sorted Source Nodes: [add_877, add_884, add_886, mul_2363, sum_884, add_893], Original ATen: [aten.add, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_sum_32.run(buf2806, buf2804, arg108_1, buf2748, buf2774, buf2778, 2048, 2048, stream=stream0)
        del arg108_1
        buf2807 = buf2749; del buf2749  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_3144, pow_395, mean_394], Original ATen: [prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_mean_pow_25.run(buf2806, buf2807, 1, 2048, stream=stream0)
        buf2808 = buf2777; del buf2777  # reuse
        # Topologically Sorted Source Nodes: [mul_2366, sum_885], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_26.run(arg109_1, buf2806, buf2807, arg110_1, buf2808, 12288, 2048, stream=stream0)
        del arg109_1
        del arg110_1
        buf2809 = reinterpret_tensor(buf2804, (1, 2048), (2048, 1), 0); del buf2804  # reuse
        # Topologically Sorted Source Nodes: [mul_2369, sum_886], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf2808, arg111_1, buf2809, 2048, 6144, stream=stream0)
        del arg111_1
        buf2810 = buf2807; del buf2807  # reuse
        # Topologically Sorted Source Nodes: [add_895, convert_element_type_3154, pow_396, mean_395], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_27.run(buf2806, buf2809, buf2810, 1, 2048, stream=stream0)
        buf2814 = buf2784; del buf2784  # reuse
        # Topologically Sorted Source Nodes: [mul_2375, sum_888, mul_2378, sum_889, index_put_197], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_50.run(arg112_1, buf2806, buf2809, buf2810, arg115_1, arg117_1, arg1_1, buf2814, arg119_1, 1024, 2048, stream=stream0)
        del arg115_1
        del arg117_1
        # Topologically Sorted Source Nodes: [convert_element_type_3164, pow_398, mean_397, mul_2381, cat_197, mul_2382, add_900, index_put_196], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_47.run(buf2814, arg116_1, arg1_1, buf2385, arg118_1, 8, 128, stream=stream0)
        del arg116_1
        buf2811 = buf2778; del buf2778  # reuse
        # Topologically Sorted Source Nodes: [mul_2372, sum_887], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_28.run(arg112_1, buf2806, buf2809, buf2810, arg113_1, buf2811, 2048, 2048, stream=stream0)
        del arg112_1
        del arg113_1
        buf2822 = reinterpret_tensor(buf2774, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf2774  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_3159, pow_397, mean_396, mul_2379, cat_196, mul_2380, add_899, convert_element_type_3169, mul_2383], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf2811, arg114_1, buf2385, buf2822, 16, 128, stream=stream0)
        del arg114_1
        buf2823 = buf1964; del buf1964  # reuse
        # Topologically Sorted Source Nodes: [mul_2385, sum_890], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf2822, arg118_1, buf2823, 15360, 128, stream=stream0)
        del arg118_1
        buf2832 = reinterpret_tensor(buf2823, (1, 16, 1, 960), (15360, 960, 15360, 1), 0); del buf2823  # reuse
        # Topologically Sorted Source Nodes: [full_default_507, full_default_505, where_196, add_901, eq_98, logical_not_196, any_99, logical_not_197, full_default_509, , sub, exp, div_98, where_197], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_48.run(buf2832, arg1_1, 16, 960, stream=stream0)
        buf2833 = buf1974; del buf1974  # reuse
        # Topologically Sorted Source Nodes: [mul_2386, sum_892], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf2832, arg119_1, buf2833, 16384, 120, stream=stream0)
        del arg119_1
        del buf2832
        buf2834 = reinterpret_tensor(buf2822, (16, 1, 128), (128, 128, 1), 0); del buf2822  # reuse
        # Topologically Sorted Source Nodes: [mul_2386, sum_892], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf2833, buf2834, 2048, 8, stream=stream0)
        del buf2833
        buf2835 = buf2811; del buf2811  # reuse
        # Topologically Sorted Source Nodes: [mul_2387, sum_893], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_17.run(buf2834, arg120_1, buf2835, 2048, 2048, stream=stream0)
        del arg120_1
        buf2837 = reinterpret_tensor(buf2834, (1, 2048), (2048, 1), 0); del buf2834  # reuse
        # Topologically Sorted Source Nodes: [add_895, add_902, convert_element_type_3176, pow_399, mean_398, convert_element_type_3178], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_30.run(buf2806, buf2809, buf2835, arg121_1, buf2837, 1, 2048, stream=stream0)
        del arg121_1
        buf2838 = buf2808; del buf2808  # reuse
        # Topologically Sorted Source Nodes: [mul_2390, sum_894], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_19.run(buf2837, arg122_1, buf2838, 12288, 2048, stream=stream0)
        del arg122_1
        buf2839 = buf2837; del buf2837  # reuse
        # Topologically Sorted Source Nodes: [mul_2393, sum_895], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf2838, arg123_1, buf2839, 2048, 6144, stream=stream0)
        del arg123_1
        buf2841 = buf2780; del buf2780  # reuse
        # Topologically Sorted Source Nodes: [add_895, add_902, add_904, convert_element_type_3186, pow_400, mean_399, add_905, rsqrt_399, mul_2394, convert_element_type_3187, mul_2395], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_mul_pow_rsqrt_31.run(buf2806, buf2809, buf2835, buf2839, arg124_1, buf2841, 1, 2048, stream=stream0)
        del arg124_1
        buf2845 = buf2814; del buf2814  # reuse
        # Topologically Sorted Source Nodes: [mul_2399, sum_897, mul_2402, sum_898, index_put_199], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_49.run(buf2841, arg127_1, arg129_1, arg1_1, buf2845, arg131_1, 1024, 2048, stream=stream0)
        del arg127_1
        del arg129_1
        # Topologically Sorted Source Nodes: [convert_element_type_3196, pow_402, mean_401, mul_2405, cat_199, mul_2406, add_909, index_put_198], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_47.run(buf2845, arg128_1, arg1_1, buf2385, arg130_1, 8, 128, stream=stream0)
        del arg128_1
        buf2842 = buf2748; del buf2748  # reuse
        # Topologically Sorted Source Nodes: [mul_2396, sum_896], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_22.run(buf2841, arg125_1, buf2842, 2048, 2048, stream=stream0)
        del arg125_1
        buf2853 = reinterpret_tensor(buf2781, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf2781  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_3191, pow_401, mean_400, mul_2403, cat_198, mul_2404, add_908, convert_element_type_3201, mul_2407], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf2842, arg126_1, buf2385, buf2853, 16, 128, stream=stream0)
        del arg126_1
        buf2854 = buf1995; del buf1995  # reuse
        # Topologically Sorted Source Nodes: [mul_2409, sum_899], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf2853, arg130_1, buf2854, 15360, 128, stream=stream0)
        del arg130_1
        buf2863 = reinterpret_tensor(buf2854, (1, 16, 1, 960), (15360, 960, 15360, 1), 0); del buf2854  # reuse
        # Topologically Sorted Source Nodes: [full_default_513, full_default_511, where_198, add_910, eq_99, logical_not_198, any_100, logical_not_199, full_default_515, , sub, exp, div_99, where_199], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_48.run(buf2863, arg1_1, 16, 960, stream=stream0)
        buf2864 = buf2005; del buf2005  # reuse
        # Topologically Sorted Source Nodes: [mul_2410, sum_901], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf2863, arg131_1, buf2864, 16384, 120, stream=stream0)
        del arg131_1
        del buf2863
        buf2865 = reinterpret_tensor(buf2853, (16, 1, 128), (128, 128, 1), 0); del buf2853  # reuse
        # Topologically Sorted Source Nodes: [mul_2410, sum_901], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf2864, buf2865, 2048, 8, stream=stream0)
        del buf2864
        buf2867 = buf2806; del buf2806  # reuse
        # Topologically Sorted Source Nodes: [add_895, add_902, add_904, mul_2411, sum_902, add_911], Original ATen: [aten.add, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_sum_32.run(buf2867, buf2865, arg132_1, buf2809, buf2835, buf2839, 2048, 2048, stream=stream0)
        del arg132_1
        buf2868 = buf2810; del buf2810  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_3208, pow_403, mean_402], Original ATen: [prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_mean_pow_25.run(buf2867, buf2868, 1, 2048, stream=stream0)
        buf2869 = buf2838; del buf2838  # reuse
        # Topologically Sorted Source Nodes: [mul_2414, sum_903], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_26.run(arg133_1, buf2867, buf2868, arg134_1, buf2869, 12288, 2048, stream=stream0)
        del arg133_1
        del arg134_1
        buf2870 = reinterpret_tensor(buf2865, (1, 2048), (2048, 1), 0); del buf2865  # reuse
        # Topologically Sorted Source Nodes: [mul_2417, sum_904], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf2869, arg135_1, buf2870, 2048, 6144, stream=stream0)
        del arg135_1
        buf2871 = buf2868; del buf2868  # reuse
        # Topologically Sorted Source Nodes: [add_913, convert_element_type_3218, pow_404, mean_403], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_27.run(buf2867, buf2870, buf2871, 1, 2048, stream=stream0)
        buf2875 = buf2845; del buf2845  # reuse
        # Topologically Sorted Source Nodes: [mul_2423, sum_906, mul_2426, sum_907, index_put_201], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_50.run(arg136_1, buf2867, buf2870, buf2871, arg139_1, arg141_1, arg1_1, buf2875, arg143_1, 1024, 2048, stream=stream0)
        del arg139_1
        del arg141_1
        # Topologically Sorted Source Nodes: [convert_element_type_3228, pow_406, mean_405, mul_2429, cat_201, mul_2430, add_918, index_put_200], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_47.run(buf2875, arg140_1, arg1_1, buf2385, arg142_1, 8, 128, stream=stream0)
        del arg140_1
        buf2872 = buf2839; del buf2839  # reuse
        # Topologically Sorted Source Nodes: [mul_2420, sum_905], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_28.run(arg136_1, buf2867, buf2870, buf2871, arg137_1, buf2872, 2048, 2048, stream=stream0)
        del arg136_1
        del arg137_1
        buf2883 = reinterpret_tensor(buf2835, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf2835  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_3223, pow_405, mean_404, mul_2427, cat_200, mul_2428, add_917, convert_element_type_3233, mul_2431], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf2872, arg138_1, buf2385, buf2883, 16, 128, stream=stream0)
        del arg138_1
        buf2884 = buf2025; del buf2025  # reuse
        # Topologically Sorted Source Nodes: [mul_2433, sum_908], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf2883, arg142_1, buf2884, 15360, 128, stream=stream0)
        del arg142_1
        buf2893 = reinterpret_tensor(buf2884, (1, 16, 1, 960), (15360, 960, 15360, 1), 0); del buf2884  # reuse
        # Topologically Sorted Source Nodes: [full_default_519, full_default_517, where_200, add_919, eq_100, logical_not_200, any_101, logical_not_201, full_default_521, , sub, exp, div_100, where_201], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_48.run(buf2893, arg1_1, 16, 960, stream=stream0)
        buf2894 = buf2035; del buf2035  # reuse
        # Topologically Sorted Source Nodes: [mul_2434, sum_910], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf2893, arg143_1, buf2894, 16384, 120, stream=stream0)
        del arg143_1
        del buf2893
        buf2895 = reinterpret_tensor(buf2883, (16, 1, 128), (128, 128, 1), 0); del buf2883  # reuse
        # Topologically Sorted Source Nodes: [mul_2434, sum_910], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf2894, buf2895, 2048, 8, stream=stream0)
        del buf2894
        buf2896 = buf2872; del buf2872  # reuse
        # Topologically Sorted Source Nodes: [mul_2435, sum_911], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_17.run(buf2895, arg144_1, buf2896, 2048, 2048, stream=stream0)
        del arg144_1
        buf2898 = reinterpret_tensor(buf2895, (1, 2048), (2048, 1), 0); del buf2895  # reuse
        # Topologically Sorted Source Nodes: [add_913, add_920, convert_element_type_3240, pow_407, mean_406, convert_element_type_3242], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_30.run(buf2867, buf2870, buf2896, arg145_1, buf2898, 1, 2048, stream=stream0)
        del arg145_1
        buf2899 = buf2869; del buf2869  # reuse
        # Topologically Sorted Source Nodes: [mul_2438, sum_912], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_19.run(buf2898, arg146_1, buf2899, 12288, 2048, stream=stream0)
        del arg146_1
        buf2900 = buf2898; del buf2898  # reuse
        # Topologically Sorted Source Nodes: [mul_2441, sum_913], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf2899, arg147_1, buf2900, 2048, 6144, stream=stream0)
        del arg147_1
        buf2902 = buf2841; del buf2841  # reuse
        # Topologically Sorted Source Nodes: [add_913, add_920, add_922, convert_element_type_3250, pow_408, mean_407, add_923, rsqrt_407, mul_2442, convert_element_type_3251, mul_2443], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_mul_pow_rsqrt_31.run(buf2867, buf2870, buf2896, buf2900, arg148_1, buf2902, 1, 2048, stream=stream0)
        del arg148_1
        buf2906 = buf2875; del buf2875  # reuse
        # Topologically Sorted Source Nodes: [mul_2447, sum_915, mul_2450, sum_916, index_put_203], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_49.run(buf2902, arg151_1, arg153_1, arg1_1, buf2906, arg155_1, 1024, 2048, stream=stream0)
        del arg151_1
        del arg153_1
        # Topologically Sorted Source Nodes: [convert_element_type_3260, pow_410, mean_409, mul_2453, cat_203, mul_2454, add_927, index_put_202], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_47.run(buf2906, arg152_1, arg1_1, buf2385, arg154_1, 8, 128, stream=stream0)
        del arg152_1
        buf2903 = buf2809; del buf2809  # reuse
        # Topologically Sorted Source Nodes: [mul_2444, sum_914], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_22.run(buf2902, arg149_1, buf2903, 2048, 2048, stream=stream0)
        del arg149_1
        buf2914 = reinterpret_tensor(buf2842, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf2842  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_3255, pow_409, mean_408, mul_2451, cat_202, mul_2452, add_926, convert_element_type_3265, mul_2455], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf2903, arg150_1, buf2385, buf2914, 16, 128, stream=stream0)
        del arg150_1
        buf2915 = buf2056; del buf2056  # reuse
        # Topologically Sorted Source Nodes: [mul_2457, sum_917], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf2914, arg154_1, buf2915, 15360, 128, stream=stream0)
        del arg154_1
        buf2924 = reinterpret_tensor(buf2915, (1, 16, 1, 960), (15360, 960, 15360, 1), 0); del buf2915  # reuse
        # Topologically Sorted Source Nodes: [full_default_525, full_default_523, where_202, add_928, eq_101, logical_not_202, any_102, logical_not_203, full_default_527, , sub, exp, div_101, where_203], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_48.run(buf2924, arg1_1, 16, 960, stream=stream0)
        buf2925 = buf2066; del buf2066  # reuse
        # Topologically Sorted Source Nodes: [mul_2458, sum_919], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf2924, arg155_1, buf2925, 16384, 120, stream=stream0)
        del arg155_1
        del buf2924
        buf2926 = reinterpret_tensor(buf2914, (16, 1, 128), (128, 128, 1), 0); del buf2914  # reuse
        # Topologically Sorted Source Nodes: [mul_2458, sum_919], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf2925, buf2926, 2048, 8, stream=stream0)
        del buf2925
        buf2928 = buf2867; del buf2867  # reuse
        # Topologically Sorted Source Nodes: [add_913, add_920, add_922, mul_2459, sum_920, add_929], Original ATen: [aten.add, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_sum_32.run(buf2928, buf2926, arg156_1, buf2870, buf2896, buf2900, 2048, 2048, stream=stream0)
        del arg156_1
        buf2929 = buf2871; del buf2871  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_3272, pow_411, mean_410], Original ATen: [prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_mean_pow_25.run(buf2928, buf2929, 1, 2048, stream=stream0)
        buf2930 = buf2899; del buf2899  # reuse
        # Topologically Sorted Source Nodes: [mul_2462, sum_921], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_26.run(arg157_1, buf2928, buf2929, arg158_1, buf2930, 12288, 2048, stream=stream0)
        del arg157_1
        del arg158_1
        buf2931 = reinterpret_tensor(buf2926, (1, 2048), (2048, 1), 0); del buf2926  # reuse
        # Topologically Sorted Source Nodes: [mul_2465, sum_922], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf2930, arg159_1, buf2931, 2048, 6144, stream=stream0)
        del arg159_1
        buf2932 = buf2929; del buf2929  # reuse
        # Topologically Sorted Source Nodes: [add_931, convert_element_type_3282, pow_412, mean_411], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_27.run(buf2928, buf2931, buf2932, 1, 2048, stream=stream0)
        buf2936 = buf2906; del buf2906  # reuse
        # Topologically Sorted Source Nodes: [mul_2471, sum_924, mul_2474, sum_925, index_put_205], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_50.run(arg160_1, buf2928, buf2931, buf2932, arg163_1, arg165_1, arg1_1, buf2936, arg167_1, 1024, 2048, stream=stream0)
        del arg163_1
        del arg165_1
        # Topologically Sorted Source Nodes: [convert_element_type_3292, pow_414, mean_413, mul_2477, cat_205, mul_2478, add_936, index_put_204], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_47.run(buf2936, arg164_1, arg1_1, buf2385, arg166_1, 8, 128, stream=stream0)
        del arg164_1
        buf2933 = buf2900; del buf2900  # reuse
        # Topologically Sorted Source Nodes: [mul_2468, sum_923], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_28.run(arg160_1, buf2928, buf2931, buf2932, arg161_1, buf2933, 2048, 2048, stream=stream0)
        del arg160_1
        del arg161_1
        buf2944 = reinterpret_tensor(buf2896, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf2896  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_3287, pow_413, mean_412, mul_2475, cat_204, mul_2476, add_935, convert_element_type_3297, mul_2479], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf2933, arg162_1, buf2385, buf2944, 16, 128, stream=stream0)
        del arg162_1
        buf2945 = buf2086; del buf2086  # reuse
        # Topologically Sorted Source Nodes: [mul_2481, sum_926], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf2944, arg166_1, buf2945, 15360, 128, stream=stream0)
        del arg166_1
        buf2954 = reinterpret_tensor(buf2945, (1, 16, 1, 960), (15360, 960, 15360, 1), 0); del buf2945  # reuse
        # Topologically Sorted Source Nodes: [full_default_531, full_default_529, where_204, add_937, eq_102, logical_not_204, any_103, logical_not_205, full_default_533, , sub, exp, div_102, where_205], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_48.run(buf2954, arg1_1, 16, 960, stream=stream0)
        buf2955 = buf2096; del buf2096  # reuse
        # Topologically Sorted Source Nodes: [mul_2482, sum_928], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf2954, arg167_1, buf2955, 16384, 120, stream=stream0)
        del arg167_1
        del buf2954
        buf2956 = reinterpret_tensor(buf2944, (16, 1, 128), (128, 128, 1), 0); del buf2944  # reuse
        # Topologically Sorted Source Nodes: [mul_2482, sum_928], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf2955, buf2956, 2048, 8, stream=stream0)
        del buf2955
        buf2957 = buf2933; del buf2933  # reuse
        # Topologically Sorted Source Nodes: [mul_2483, sum_929], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_17.run(buf2956, arg168_1, buf2957, 2048, 2048, stream=stream0)
        del arg168_1
        buf2959 = reinterpret_tensor(buf2956, (1, 2048), (2048, 1), 0); del buf2956  # reuse
        # Topologically Sorted Source Nodes: [add_931, add_938, convert_element_type_3304, pow_415, mean_414, convert_element_type_3306], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_30.run(buf2928, buf2931, buf2957, arg169_1, buf2959, 1, 2048, stream=stream0)
        del arg169_1
        buf2960 = buf2930; del buf2930  # reuse
        # Topologically Sorted Source Nodes: [mul_2486, sum_930], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_19.run(buf2959, arg170_1, buf2960, 12288, 2048, stream=stream0)
        del arg170_1
        buf2961 = buf2959; del buf2959  # reuse
        # Topologically Sorted Source Nodes: [mul_2489, sum_931], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf2960, arg171_1, buf2961, 2048, 6144, stream=stream0)
        del arg171_1
        buf2963 = buf2902; del buf2902  # reuse
        # Topologically Sorted Source Nodes: [add_931, add_938, add_940, convert_element_type_3314, pow_416, mean_415, add_941, rsqrt_415, mul_2490, convert_element_type_3315, mul_2491], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_mul_pow_rsqrt_31.run(buf2928, buf2931, buf2957, buf2961, arg172_1, buf2963, 1, 2048, stream=stream0)
        del arg172_1
        buf2967 = buf2936; del buf2936  # reuse
        # Topologically Sorted Source Nodes: [mul_2495, sum_933, mul_2498, sum_934, index_put_207], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_49.run(buf2963, arg175_1, arg177_1, arg1_1, buf2967, arg179_1, 1024, 2048, stream=stream0)
        del arg175_1
        del arg177_1
        # Topologically Sorted Source Nodes: [convert_element_type_3324, pow_418, mean_417, mul_2501, cat_207, mul_2502, add_945, index_put_206], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_47.run(buf2967, arg176_1, arg1_1, buf2385, arg178_1, 8, 128, stream=stream0)
        del arg176_1
        buf2964 = buf2870; del buf2870  # reuse
        # Topologically Sorted Source Nodes: [mul_2492, sum_932], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_22.run(buf2963, arg173_1, buf2964, 2048, 2048, stream=stream0)
        del arg173_1
        buf2975 = reinterpret_tensor(buf2903, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf2903  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_3319, pow_417, mean_416, mul_2499, cat_206, mul_2500, add_944, convert_element_type_3329, mul_2503], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf2964, arg174_1, buf2385, buf2975, 16, 128, stream=stream0)
        del arg174_1
        buf2976 = buf2117; del buf2117  # reuse
        # Topologically Sorted Source Nodes: [mul_2505, sum_935], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf2975, arg178_1, buf2976, 15360, 128, stream=stream0)
        del arg178_1
        buf2985 = reinterpret_tensor(buf2976, (1, 16, 1, 960), (15360, 960, 15360, 1), 0); del buf2976  # reuse
        # Topologically Sorted Source Nodes: [full_default_537, full_default_535, where_206, add_946, eq_103, logical_not_206, any_104, logical_not_207, full_default_539, , sub, exp, div_103, where_207], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_48.run(buf2985, arg1_1, 16, 960, stream=stream0)
        buf2986 = buf2127; del buf2127  # reuse
        # Topologically Sorted Source Nodes: [mul_2506, sum_937], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf2985, arg179_1, buf2986, 16384, 120, stream=stream0)
        del arg179_1
        del buf2985
        buf2987 = reinterpret_tensor(buf2975, (16, 1, 128), (128, 128, 1), 0); del buf2975  # reuse
        # Topologically Sorted Source Nodes: [mul_2506, sum_937], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf2986, buf2987, 2048, 8, stream=stream0)
        del buf2986
        buf2989 = buf2928; del buf2928  # reuse
        # Topologically Sorted Source Nodes: [add_931, add_938, add_940, mul_2507, sum_938, add_947], Original ATen: [aten.add, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_sum_32.run(buf2989, buf2987, arg180_1, buf2931, buf2957, buf2961, 2048, 2048, stream=stream0)
        del arg180_1
        buf2990 = buf2932; del buf2932  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_3336, pow_419, mean_418], Original ATen: [prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_mean_pow_25.run(buf2989, buf2990, 1, 2048, stream=stream0)
        buf2991 = buf2960; del buf2960  # reuse
        # Topologically Sorted Source Nodes: [mul_2510, sum_939], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_26.run(arg181_1, buf2989, buf2990, arg182_1, buf2991, 12288, 2048, stream=stream0)
        del arg181_1
        del arg182_1
        buf2992 = reinterpret_tensor(buf2987, (1, 2048), (2048, 1), 0); del buf2987  # reuse
        # Topologically Sorted Source Nodes: [mul_2513, sum_940], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf2991, arg183_1, buf2992, 2048, 6144, stream=stream0)
        del arg183_1
        buf2993 = buf2990; del buf2990  # reuse
        # Topologically Sorted Source Nodes: [add_949, convert_element_type_3346, pow_420, mean_419], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_27.run(buf2989, buf2992, buf2993, 1, 2048, stream=stream0)
        buf2997 = buf2967; del buf2967  # reuse
        # Topologically Sorted Source Nodes: [mul_2519, sum_942, mul_2522, sum_943, index_put_209], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_50.run(arg184_1, buf2989, buf2992, buf2993, arg187_1, arg189_1, arg1_1, buf2997, arg191_1, 1024, 2048, stream=stream0)
        del arg187_1
        del arg189_1
        # Topologically Sorted Source Nodes: [convert_element_type_3356, pow_422, mean_421, mul_2525, cat_209, mul_2526, add_954, index_put_208], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_47.run(buf2997, arg188_1, arg1_1, buf2385, arg190_1, 8, 128, stream=stream0)
        del arg188_1
        buf2994 = buf2961; del buf2961  # reuse
        # Topologically Sorted Source Nodes: [mul_2516, sum_941], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_28.run(arg184_1, buf2989, buf2992, buf2993, arg185_1, buf2994, 2048, 2048, stream=stream0)
        del arg184_1
        del arg185_1
        buf3005 = reinterpret_tensor(buf2957, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf2957  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_3351, pow_421, mean_420, mul_2523, cat_208, mul_2524, add_953, convert_element_type_3361, mul_2527], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf2994, arg186_1, buf2385, buf3005, 16, 128, stream=stream0)
        del arg186_1
        buf3006 = buf2147; del buf2147  # reuse
        # Topologically Sorted Source Nodes: [mul_2529, sum_944], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf3005, arg190_1, buf3006, 15360, 128, stream=stream0)
        del arg190_1
        buf3015 = reinterpret_tensor(buf3006, (1, 16, 1, 960), (15360, 960, 15360, 1), 0); del buf3006  # reuse
        # Topologically Sorted Source Nodes: [full_default_543, full_default_541, where_208, add_955, eq_104, logical_not_208, any_105, logical_not_209, full_default_545, , sub, exp, div_104, where_209], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_48.run(buf3015, arg1_1, 16, 960, stream=stream0)
        buf3016 = buf2157; del buf2157  # reuse
        # Topologically Sorted Source Nodes: [mul_2530, sum_946], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf3015, arg191_1, buf3016, 16384, 120, stream=stream0)
        del arg191_1
        del buf3015
        buf3017 = reinterpret_tensor(buf3005, (16, 1, 128), (128, 128, 1), 0); del buf3005  # reuse
        # Topologically Sorted Source Nodes: [mul_2530, sum_946], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf3016, buf3017, 2048, 8, stream=stream0)
        del buf3016
        buf3018 = buf2994; del buf2994  # reuse
        # Topologically Sorted Source Nodes: [mul_2531, sum_947], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_17.run(buf3017, arg192_1, buf3018, 2048, 2048, stream=stream0)
        del arg192_1
        buf3020 = reinterpret_tensor(buf3017, (1, 2048), (2048, 1), 0); del buf3017  # reuse
        # Topologically Sorted Source Nodes: [add_949, add_956, convert_element_type_3368, pow_423, mean_422, convert_element_type_3370], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_30.run(buf2989, buf2992, buf3018, arg193_1, buf3020, 1, 2048, stream=stream0)
        del arg193_1
        buf3021 = buf2991; del buf2991  # reuse
        # Topologically Sorted Source Nodes: [mul_2534, sum_948], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_19.run(buf3020, arg194_1, buf3021, 12288, 2048, stream=stream0)
        del arg194_1
        buf3022 = buf3020; del buf3020  # reuse
        # Topologically Sorted Source Nodes: [mul_2537, sum_949], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf3021, arg195_1, buf3022, 2048, 6144, stream=stream0)
        del arg195_1
        buf3024 = buf2963; del buf2963  # reuse
        # Topologically Sorted Source Nodes: [add_949, add_956, add_958, convert_element_type_3378, pow_424, mean_423, add_959, rsqrt_423, mul_2538, convert_element_type_3379, mul_2539], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_mul_pow_rsqrt_31.run(buf2989, buf2992, buf3018, buf3022, arg196_1, buf3024, 1, 2048, stream=stream0)
        del arg196_1
        buf3028 = buf2997; del buf2997  # reuse
        # Topologically Sorted Source Nodes: [mul_2543, sum_951, mul_2546, sum_952, index_put_211], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_49.run(buf3024, arg199_1, arg201_1, arg1_1, buf3028, arg203_1, 1024, 2048, stream=stream0)
        del arg199_1
        del arg201_1
        # Topologically Sorted Source Nodes: [convert_element_type_3388, pow_426, mean_425, mul_2549, cat_211, mul_2550, add_963, index_put_210], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_47.run(buf3028, arg200_1, arg1_1, buf2385, arg202_1, 8, 128, stream=stream0)
        del arg200_1
        buf3025 = buf2931; del buf2931  # reuse
        # Topologically Sorted Source Nodes: [mul_2540, sum_950], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_22.run(buf3024, arg197_1, buf3025, 2048, 2048, stream=stream0)
        del arg197_1
        buf3036 = reinterpret_tensor(buf2964, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf2964  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_3383, pow_425, mean_424, mul_2547, cat_210, mul_2548, add_962, convert_element_type_3393, mul_2551], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf3025, arg198_1, buf2385, buf3036, 16, 128, stream=stream0)
        del arg198_1
        buf3037 = buf2178; del buf2178  # reuse
        # Topologically Sorted Source Nodes: [mul_2553, sum_953], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf3036, arg202_1, buf3037, 15360, 128, stream=stream0)
        del arg202_1
        buf3046 = reinterpret_tensor(buf3037, (1, 16, 1, 960), (15360, 960, 15360, 1), 0); del buf3037  # reuse
        # Topologically Sorted Source Nodes: [full_default_549, full_default_547, where_210, add_964, eq_105, logical_not_210, any_106, logical_not_211, full_default_551, , sub, exp, div_105, where_211], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_48.run(buf3046, arg1_1, 16, 960, stream=stream0)
        buf3047 = buf2188; del buf2188  # reuse
        # Topologically Sorted Source Nodes: [mul_2554, sum_955], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf3046, arg203_1, buf3047, 16384, 120, stream=stream0)
        del arg203_1
        del buf3046
        buf3048 = reinterpret_tensor(buf3036, (16, 1, 128), (128, 128, 1), 0); del buf3036  # reuse
        # Topologically Sorted Source Nodes: [mul_2554, sum_955], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf3047, buf3048, 2048, 8, stream=stream0)
        del buf3047
        buf3050 = buf2989; del buf2989  # reuse
        # Topologically Sorted Source Nodes: [add_949, add_956, add_958, mul_2555, sum_956, add_965], Original ATen: [aten.add, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_sum_32.run(buf3050, buf3048, arg204_1, buf2992, buf3018, buf3022, 2048, 2048, stream=stream0)
        del arg204_1
        buf3051 = buf2993; del buf2993  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_3400, pow_427, mean_426], Original ATen: [prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_mean_pow_25.run(buf3050, buf3051, 1, 2048, stream=stream0)
        buf3052 = buf3021; del buf3021  # reuse
        # Topologically Sorted Source Nodes: [mul_2558, sum_957], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_26.run(arg205_1, buf3050, buf3051, arg206_1, buf3052, 12288, 2048, stream=stream0)
        del arg205_1
        del arg206_1
        buf3053 = reinterpret_tensor(buf3048, (1, 2048), (2048, 1), 0); del buf3048  # reuse
        # Topologically Sorted Source Nodes: [mul_2561, sum_958], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf3052, arg207_1, buf3053, 2048, 6144, stream=stream0)
        del arg207_1
        buf3054 = buf3051; del buf3051  # reuse
        # Topologically Sorted Source Nodes: [add_967, convert_element_type_3410, pow_428, mean_427], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_27.run(buf3050, buf3053, buf3054, 1, 2048, stream=stream0)
        buf3058 = buf3028; del buf3028  # reuse
        # Topologically Sorted Source Nodes: [mul_2567, sum_960, mul_2570, sum_961, index_put_213], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_50.run(arg208_1, buf3050, buf3053, buf3054, arg211_1, arg213_1, arg1_1, buf3058, arg215_1, 1024, 2048, stream=stream0)
        del arg211_1
        del arg213_1
        # Topologically Sorted Source Nodes: [convert_element_type_3420, pow_430, mean_429, mul_2573, cat_213, mul_2574, add_972, index_put_212], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_47.run(buf3058, arg212_1, arg1_1, buf2385, arg214_1, 8, 128, stream=stream0)
        del arg212_1
        buf3055 = buf3022; del buf3022  # reuse
        # Topologically Sorted Source Nodes: [mul_2564, sum_959], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_28.run(arg208_1, buf3050, buf3053, buf3054, arg209_1, buf3055, 2048, 2048, stream=stream0)
        del arg208_1
        del arg209_1
        buf3066 = reinterpret_tensor(buf3018, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf3018  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_3415, pow_429, mean_428, mul_2571, cat_212, mul_2572, add_971, convert_element_type_3425, mul_2575], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf3055, arg210_1, buf2385, buf3066, 16, 128, stream=stream0)
        del arg210_1
        buf3067 = buf2208; del buf2208  # reuse
        # Topologically Sorted Source Nodes: [mul_2577, sum_962], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf3066, arg214_1, buf3067, 15360, 128, stream=stream0)
        del arg214_1
        buf3076 = reinterpret_tensor(buf3067, (1, 16, 1, 960), (15360, 960, 15360, 1), 0); del buf3067  # reuse
        # Topologically Sorted Source Nodes: [full_default_555, full_default_553, where_212, add_973, eq_106, logical_not_212, any_107, logical_not_213, full_default_557, , sub, exp, div_106, where_213], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_48.run(buf3076, arg1_1, 16, 960, stream=stream0)
        buf3077 = buf2218; del buf2218  # reuse
        # Topologically Sorted Source Nodes: [mul_2578, sum_964], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf3076, arg215_1, buf3077, 16384, 120, stream=stream0)
        del arg215_1
        del buf3076
        buf3078 = reinterpret_tensor(buf3066, (16, 1, 128), (128, 128, 1), 0); del buf3066  # reuse
        # Topologically Sorted Source Nodes: [mul_2578, sum_964], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf3077, buf3078, 2048, 8, stream=stream0)
        del buf3077
        buf3079 = buf3055; del buf3055  # reuse
        # Topologically Sorted Source Nodes: [mul_2579, sum_965], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_17.run(buf3078, arg216_1, buf3079, 2048, 2048, stream=stream0)
        del arg216_1
        buf3081 = reinterpret_tensor(buf3078, (1, 2048), (2048, 1), 0); del buf3078  # reuse
        # Topologically Sorted Source Nodes: [add_967, add_974, convert_element_type_3432, pow_431, mean_430, convert_element_type_3434], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_30.run(buf3050, buf3053, buf3079, arg217_1, buf3081, 1, 2048, stream=stream0)
        del arg217_1
        buf3082 = buf3052; del buf3052  # reuse
        # Topologically Sorted Source Nodes: [mul_2582, sum_966], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_19.run(buf3081, arg218_1, buf3082, 12288, 2048, stream=stream0)
        del arg218_1
        buf3083 = buf3081; del buf3081  # reuse
        # Topologically Sorted Source Nodes: [mul_2585, sum_967], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf3082, arg219_1, buf3083, 2048, 6144, stream=stream0)
        del arg219_1
        buf3085 = buf3024; del buf3024  # reuse
        # Topologically Sorted Source Nodes: [add_967, add_974, add_976, convert_element_type_3442, pow_432, mean_431, add_977, rsqrt_431, mul_2586, convert_element_type_3443, mul_2587], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_mul_pow_rsqrt_31.run(buf3050, buf3053, buf3079, buf3083, arg220_1, buf3085, 1, 2048, stream=stream0)
        del arg220_1
        buf3089 = buf3058; del buf3058  # reuse
        # Topologically Sorted Source Nodes: [mul_2591, sum_969, mul_2594, sum_970, index_put_215], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_49.run(buf3085, arg223_1, arg225_1, arg1_1, buf3089, arg227_1, 1024, 2048, stream=stream0)
        del arg223_1
        del arg225_1
        # Topologically Sorted Source Nodes: [convert_element_type_3452, pow_434, mean_433, mul_2597, cat_215, mul_2598, add_981, index_put_214], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_47.run(buf3089, arg224_1, arg1_1, buf2385, arg226_1, 8, 128, stream=stream0)
        del arg224_1
        buf3086 = buf2992; del buf2992  # reuse
        # Topologically Sorted Source Nodes: [mul_2588, sum_968], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_22.run(buf3085, arg221_1, buf3086, 2048, 2048, stream=stream0)
        del arg221_1
        buf3097 = reinterpret_tensor(buf3025, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf3025  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_3447, pow_433, mean_432, mul_2595, cat_214, mul_2596, add_980, convert_element_type_3457, mul_2599], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf3086, arg222_1, buf2385, buf3097, 16, 128, stream=stream0)
        del arg222_1
        buf3098 = buf2239; del buf2239  # reuse
        # Topologically Sorted Source Nodes: [mul_2601, sum_971], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf3097, arg226_1, buf3098, 15360, 128, stream=stream0)
        del arg226_1
        buf3107 = reinterpret_tensor(buf3098, (1, 16, 1, 960), (15360, 960, 15360, 1), 0); del buf3098  # reuse
        # Topologically Sorted Source Nodes: [full_default_561, full_default_559, where_214, add_982, eq_107, logical_not_214, any_108, logical_not_215, full_default_563, , sub, exp, div_107, where_215], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_48.run(buf3107, arg1_1, 16, 960, stream=stream0)
        buf3108 = buf2249; del buf2249  # reuse
        # Topologically Sorted Source Nodes: [mul_2602, sum_973], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf3107, arg227_1, buf3108, 16384, 120, stream=stream0)
        del arg227_1
        del buf3107
        buf3109 = reinterpret_tensor(buf3097, (16, 1, 128), (128, 128, 1), 0); del buf3097  # reuse
        # Topologically Sorted Source Nodes: [mul_2602, sum_973], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf3108, buf3109, 2048, 8, stream=stream0)
        del buf3108
        buf3111 = buf3050; del buf3050  # reuse
        # Topologically Sorted Source Nodes: [add_967, add_974, add_976, mul_2603, sum_974, add_983], Original ATen: [aten.add, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_sum_32.run(buf3111, buf3109, arg228_1, buf3053, buf3079, buf3083, 2048, 2048, stream=stream0)
        del arg228_1
        buf3112 = buf3054; del buf3054  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_3464, pow_435, mean_434], Original ATen: [prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_mean_pow_25.run(buf3111, buf3112, 1, 2048, stream=stream0)
        buf3113 = buf3082; del buf3082  # reuse
        # Topologically Sorted Source Nodes: [mul_2606, sum_975], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_26.run(arg229_1, buf3111, buf3112, arg230_1, buf3113, 12288, 2048, stream=stream0)
        del arg229_1
        del arg230_1
        buf3114 = reinterpret_tensor(buf3109, (1, 2048), (2048, 1), 0); del buf3109  # reuse
        # Topologically Sorted Source Nodes: [mul_2609, sum_976], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf3113, arg231_1, buf3114, 2048, 6144, stream=stream0)
        del arg231_1
        buf3115 = buf3112; del buf3112  # reuse
        # Topologically Sorted Source Nodes: [add_985, convert_element_type_3474, pow_436, mean_435], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_27.run(buf3111, buf3114, buf3115, 1, 2048, stream=stream0)
        buf3119 = buf3089; del buf3089  # reuse
        # Topologically Sorted Source Nodes: [mul_2615, sum_978, mul_2618, sum_979, index_put_217], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_50.run(arg232_1, buf3111, buf3114, buf3115, arg235_1, arg237_1, arg1_1, buf3119, arg239_1, 1024, 2048, stream=stream0)
        del arg235_1
        del arg237_1
        # Topologically Sorted Source Nodes: [convert_element_type_3484, pow_438, mean_437, mul_2621, cat_217, mul_2622, add_990, index_put_216], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_47.run(buf3119, arg236_1, arg1_1, buf2385, arg238_1, 8, 128, stream=stream0)
        del arg236_1
        buf3116 = buf3083; del buf3083  # reuse
        # Topologically Sorted Source Nodes: [mul_2612, sum_977], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_28.run(arg232_1, buf3111, buf3114, buf3115, arg233_1, buf3116, 2048, 2048, stream=stream0)
        del arg232_1
        del arg233_1
        buf3127 = reinterpret_tensor(buf3079, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf3079  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_3479, pow_437, mean_436, mul_2619, cat_216, mul_2620, add_989, convert_element_type_3489, mul_2623], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf3116, arg234_1, buf2385, buf3127, 16, 128, stream=stream0)
        del arg234_1
        buf3128 = buf2269; del buf2269  # reuse
        # Topologically Sorted Source Nodes: [mul_2625, sum_980], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf3127, arg238_1, buf3128, 15360, 128, stream=stream0)
        del arg238_1
        buf3137 = reinterpret_tensor(buf3128, (1, 16, 1, 960), (15360, 960, 15360, 1), 0); del buf3128  # reuse
        # Topologically Sorted Source Nodes: [full_default_567, full_default_565, where_216, add_991, eq_108, logical_not_216, any_109, logical_not_217, full_default_569, , sub, exp, div_108, where_217], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_48.run(buf3137, arg1_1, 16, 960, stream=stream0)
        buf3138 = buf2279; del buf2279  # reuse
        # Topologically Sorted Source Nodes: [mul_2626, sum_982], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf3137, arg239_1, buf3138, 16384, 120, stream=stream0)
        del arg239_1
        del buf3137
        buf3139 = reinterpret_tensor(buf3127, (16, 1, 128), (128, 128, 1), 0); del buf3127  # reuse
        # Topologically Sorted Source Nodes: [mul_2626, sum_982], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf3138, buf3139, 2048, 8, stream=stream0)
        del buf3138
        buf3140 = buf3116; del buf3116  # reuse
        # Topologically Sorted Source Nodes: [mul_2627, sum_983], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_17.run(buf3139, arg240_1, buf3140, 2048, 2048, stream=stream0)
        del arg240_1
        buf3142 = reinterpret_tensor(buf3139, (1, 2048), (2048, 1), 0); del buf3139  # reuse
        # Topologically Sorted Source Nodes: [add_985, add_992, convert_element_type_3496, pow_439, mean_438, convert_element_type_3498], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_30.run(buf3111, buf3114, buf3140, arg241_1, buf3142, 1, 2048, stream=stream0)
        del arg241_1
        buf3143 = buf3113; del buf3113  # reuse
        # Topologically Sorted Source Nodes: [mul_2630, sum_984], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_19.run(buf3142, arg242_1, buf3143, 12288, 2048, stream=stream0)
        del arg242_1
        buf3144 = buf3142; del buf3142  # reuse
        # Topologically Sorted Source Nodes: [mul_2633, sum_985], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf3143, arg243_1, buf3144, 2048, 6144, stream=stream0)
        del arg243_1
        buf3146 = buf3085; del buf3085  # reuse
        # Topologically Sorted Source Nodes: [add_985, add_992, add_994, convert_element_type_3506, pow_440, mean_439, add_995, rsqrt_439, mul_2634, convert_element_type_3507, mul_2635], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_mul_pow_rsqrt_31.run(buf3111, buf3114, buf3140, buf3144, arg244_1, buf3146, 1, 2048, stream=stream0)
        del arg244_1
        buf3150 = buf3119; del buf3119  # reuse
        # Topologically Sorted Source Nodes: [mul_2639, sum_987, mul_2642, sum_988, index_put_219], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_49.run(buf3146, arg247_1, arg249_1, arg1_1, buf3150, arg251_1, 1024, 2048, stream=stream0)
        del arg247_1
        del arg249_1
        # Topologically Sorted Source Nodes: [convert_element_type_3516, pow_442, mean_441, mul_2645, cat_219, mul_2646, add_999, index_put_218], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_47.run(buf3150, arg248_1, arg1_1, buf2385, arg250_1, 8, 128, stream=stream0)
        del arg248_1
        buf3147 = buf3053; del buf3053  # reuse
        # Topologically Sorted Source Nodes: [mul_2636, sum_986], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_22.run(buf3146, arg245_1, buf3147, 2048, 2048, stream=stream0)
        del arg245_1
        buf3158 = reinterpret_tensor(buf3086, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf3086  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_3511, pow_441, mean_440, mul_2643, cat_218, mul_2644, add_998, convert_element_type_3521, mul_2647], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf3147, arg246_1, buf2385, buf3158, 16, 128, stream=stream0)
        del arg246_1
        buf3159 = buf2300; del buf2300  # reuse
        # Topologically Sorted Source Nodes: [mul_2649, sum_989], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf3158, arg250_1, buf3159, 15360, 128, stream=stream0)
        del arg250_1
        buf3168 = reinterpret_tensor(buf3159, (1, 16, 1, 960), (15360, 960, 15360, 1), 0); del buf3159  # reuse
        # Topologically Sorted Source Nodes: [full_default_573, full_default_571, where_218, add_1000, eq_109, logical_not_218, any_110, logical_not_219, full_default_575, , sub, exp, div_109, where_219], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_48.run(buf3168, arg1_1, 16, 960, stream=stream0)
        buf3169 = buf2310; del buf2310  # reuse
        # Topologically Sorted Source Nodes: [mul_2650, sum_991], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf3168, arg251_1, buf3169, 16384, 120, stream=stream0)
        del arg251_1
        del buf3168
        buf3170 = reinterpret_tensor(buf3158, (16, 1, 128), (128, 128, 1), 0); del buf3158  # reuse
        # Topologically Sorted Source Nodes: [mul_2650, sum_991], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf3169, buf3170, 2048, 8, stream=stream0)
        del buf3169
        buf3172 = buf3111; del buf3111  # reuse
        # Topologically Sorted Source Nodes: [add_985, add_992, add_994, mul_2651, sum_992, add_1001], Original ATen: [aten.add, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_sum_32.run(buf3172, buf3170, arg252_1, buf3114, buf3140, buf3144, 2048, 2048, stream=stream0)
        del arg252_1
        buf3173 = buf3115; del buf3115  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_3528, pow_443, mean_442], Original ATen: [prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_mean_pow_25.run(buf3172, buf3173, 1, 2048, stream=stream0)
        buf3174 = buf3143; del buf3143  # reuse
        # Topologically Sorted Source Nodes: [mul_2654, sum_993], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_26.run(arg253_1, buf3172, buf3173, arg254_1, buf3174, 12288, 2048, stream=stream0)
        del arg253_1
        del arg254_1
        buf3175 = reinterpret_tensor(buf3170, (1, 2048), (2048, 1), 0); del buf3170  # reuse
        # Topologically Sorted Source Nodes: [mul_2657, sum_994], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf3174, arg255_1, buf3175, 2048, 6144, stream=stream0)
        del arg255_1
        buf3176 = buf3173; del buf3173  # reuse
        # Topologically Sorted Source Nodes: [add_1003, convert_element_type_3538, pow_444, mean_443], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_27.run(buf3172, buf3175, buf3176, 1, 2048, stream=stream0)
        buf3180 = buf3150; del buf3150  # reuse
        # Topologically Sorted Source Nodes: [mul_2663, sum_996, mul_2666, sum_997, index_put_221], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_50.run(arg256_1, buf3172, buf3175, buf3176, arg259_1, arg261_1, arg1_1, buf3180, arg263_1, 1024, 2048, stream=stream0)
        del arg259_1
        del arg261_1
        # Topologically Sorted Source Nodes: [convert_element_type_3548, pow_446, mean_445, mul_2669, cat_221, mul_2670, add_1008, index_put_220], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_47.run(buf3180, arg260_1, arg1_1, buf2385, arg262_1, 8, 128, stream=stream0)
        del arg260_1
        buf3177 = buf3144; del buf3144  # reuse
        # Topologically Sorted Source Nodes: [mul_2660, sum_995], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_28.run(arg256_1, buf3172, buf3175, buf3176, arg257_1, buf3177, 2048, 2048, stream=stream0)
        del arg256_1
        del arg257_1
        buf3188 = reinterpret_tensor(buf3140, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf3140  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_3543, pow_445, mean_444, mul_2667, cat_220, mul_2668, add_1007, convert_element_type_3553, mul_2671], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf3177, arg258_1, buf2385, buf3188, 16, 128, stream=stream0)
        del arg258_1
        buf3189 = buf2330; del buf2330  # reuse
        # Topologically Sorted Source Nodes: [mul_2673, sum_998], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf3188, arg262_1, buf3189, 15360, 128, stream=stream0)
        del arg262_1
        buf3198 = reinterpret_tensor(buf3189, (1, 16, 1, 960), (15360, 960, 15360, 1), 0); del buf3189  # reuse
        # Topologically Sorted Source Nodes: [full_default_579, full_default_577, where_220, add_1009, eq_110, logical_not_220, any_111, logical_not_221, full_default_581, , sub, exp, div_110, where_221], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_48.run(buf3198, arg1_1, 16, 960, stream=stream0)
        buf3199 = buf2340; del buf2340  # reuse
        # Topologically Sorted Source Nodes: [mul_2674, sum_1000], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf3198, arg263_1, buf3199, 16384, 120, stream=stream0)
        del arg263_1
        del buf3198
        buf3200 = reinterpret_tensor(buf3188, (16, 1, 128), (128, 128, 1), 0); del buf3188  # reuse
        # Topologically Sorted Source Nodes: [mul_2674, sum_1000], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf3199, buf3200, 2048, 8, stream=stream0)
        del buf3199
        buf3201 = buf3177; del buf3177  # reuse
        # Topologically Sorted Source Nodes: [mul_2675, sum_1001], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_17.run(buf3200, arg264_1, buf3201, 2048, 2048, stream=stream0)
        del arg264_1
        buf3203 = reinterpret_tensor(buf3200, (1, 2048), (2048, 1), 0); del buf3200  # reuse
        # Topologically Sorted Source Nodes: [add_1003, add_1010, convert_element_type_3560, pow_447, mean_446, convert_element_type_3562], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_30.run(buf3172, buf3175, buf3201, arg265_1, buf3203, 1, 2048, stream=stream0)
        del arg265_1
        buf3204 = buf3174; del buf3174  # reuse
        # Topologically Sorted Source Nodes: [mul_2678, sum_1002], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_19.run(buf3203, arg266_1, buf3204, 12288, 2048, stream=stream0)
        del arg266_1
        buf3205 = buf3203; del buf3203  # reuse
        # Topologically Sorted Source Nodes: [mul_2681, sum_1003], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf3204, arg267_1, buf3205, 2048, 6144, stream=stream0)
        del arg267_1
        buf3207 = buf3146; del buf3146  # reuse
        # Topologically Sorted Source Nodes: [add_1003, add_1010, add_1012, convert_element_type_3570, pow_448, mean_447, add_1013, rsqrt_447, mul_2682, convert_element_type_3571, mul_2683], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_mul_pow_rsqrt_31.run(buf3172, buf3175, buf3201, buf3205, arg268_1, buf3207, 1, 2048, stream=stream0)
        del arg268_1
        buf3211 = buf3180; del buf3180  # reuse
        # Topologically Sorted Source Nodes: [mul_2687, sum_1005, mul_2690, sum_1006, index_put_223], Original ATen: [aten.mul, aten.sum, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_red_fused_index_put_mul_sum_49.run(buf3207, arg271_1, arg273_1, arg1_1, buf3211, arg275_1, 1024, 2048, stream=stream0)
        del arg271_1
        del arg273_1
        # Topologically Sorted Source Nodes: [convert_element_type_3580, pow_450, mean_449, mul_2693, cat_223, mul_2694, add_1017, index_put_222], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add, aten.index_put]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_index_put_mean_mul_pow_47.run(buf3211, arg272_1, arg1_1, buf2385, arg274_1, 8, 128, stream=stream0)
        del arg272_1
        del buf3211
        buf3208 = buf3114; del buf3114  # reuse
        # Topologically Sorted Source Nodes: [mul_2684, sum_1004], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_22.run(buf3207, arg269_1, buf3208, 2048, 2048, stream=stream0)
        del arg269_1
        del buf3207
        buf3219 = reinterpret_tensor(buf3147, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf3147  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_3575, pow_449, mean_448, mul_2691, cat_222, mul_2692, add_1016, convert_element_type_3585, mul_2695], Original ATen: [prims.convert_element_type, aten.pow, aten.mean, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_pow_11.run(buf3208, arg270_1, buf2385, buf3219, 16, 128, stream=stream0)
        del arg270_1
        del buf2385
        del buf3208
        buf3220 = buf2361; del buf2361  # reuse
        # Topologically Sorted Source Nodes: [mul_2697, sum_1007], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_13.run(buf3219, arg274_1, buf3220, 15360, 128, stream=stream0)
        del arg274_1
        buf3229 = reinterpret_tensor(buf3220, (1, 16, 1, 960), (15360, 960, 15360, 1), 0); del buf3220  # reuse
        # Topologically Sorted Source Nodes: [full_default_585, full_default_583, where_222, add_1018, eq_111, logical_not_222, any_112, logical_not_223, full_default_587, , sub, exp, div_111, where_223], Original ATen: [aten.full, aten.where, aten.add, aten.eq, aten.logical_not, aten.any, prims.prepare_softmax_online, aten.sub, aten.exp, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_any_div_eq_exp_full_logical_not_prepare_softmax_online_sub_where_48.run(buf3229, arg1_1, 16, 960, stream=stream0)
        del arg1_1
        buf3230 = buf2371; del buf2371  # reuse
        # Topologically Sorted Source Nodes: [mul_2698, sum_1009], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_15.run(buf3229, arg275_1, buf3230, 16384, 120, stream=stream0)
        del arg275_1
        del buf3229
        buf3231 = reinterpret_tensor(buf3219, (16, 1, 128), (128, 128, 1), 0); del buf3219  # reuse
        # Topologically Sorted Source Nodes: [mul_2698, sum_1009], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(buf3230, buf3231, 2048, 8, stream=stream0)
        del buf3230
        buf3233 = buf3172; del buf3172  # reuse
        # Topologically Sorted Source Nodes: [add_1003, add_1010, add_1012, mul_2699, sum_1010, add_1019], Original ATen: [aten.add, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_sum_32.run(buf3233, buf3231, arg276_1, buf3175, buf3201, buf3205, 2048, 2048, stream=stream0)
        del arg276_1
        del buf3175
        del buf3201
        del buf3205
        buf3234 = buf3176; del buf3176  # reuse
        # Topologically Sorted Source Nodes: [convert_element_type_3592, pow_451, mean_450], Original ATen: [prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_mean_pow_25.run(buf3233, buf3234, 1, 2048, stream=stream0)
        buf3235 = buf3204; del buf3204  # reuse
        # Topologically Sorted Source Nodes: [mul_2702, sum_1011], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_26.run(arg277_1, buf3233, buf3234, arg278_1, buf3235, 12288, 2048, stream=stream0)
        del arg277_1
        del arg278_1
        buf3236 = reinterpret_tensor(buf3231, (1, 2048), (2048, 1), 0); del buf3231  # reuse
        # Topologically Sorted Source Nodes: [mul_2705, sum_1012], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_20.run(buf3235, arg279_1, buf3236, 2048, 6144, stream=stream0)
        del arg279_1
        del buf3235
        buf3237 = buf3234; del buf3234  # reuse
        # Topologically Sorted Source Nodes: [add_1021, convert_element_type_3602, pow_452, mean_451], Original ATen: [aten.add, prims.convert_element_type, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convert_element_type_mean_pow_27.run(buf3233, buf3236, buf3237, 1, 2048, stream=stream0)
        buf3238 = buf2379; del buf2379  # reuse
        # Topologically Sorted Source Nodes: [logits_3], Original ATen: [aten.mm]
        stream0 = get_raw_stream(0)
        triton_red_fused_mm_33.run(arg280_1, buf3233, buf3236, buf3237, arg282_1, buf3238, 151936, 2048, stream=stream0)
        del arg280_1
        del arg282_1
        del buf3233
        del buf3236
        del buf3237
        buf3243 = reinterpret_tensor(buf3244, (1, 1), (4, 1), 3)  # alias
        # Topologically Sorted Source Nodes: [logits_3, argmax_3, cat_224], Original ATen: [aten.mm, aten.argmax, aten.cat]
        stream0 = get_raw_stream(0)
        triton_red_fused_argmax_cat_mm_51.run(buf3238, buf3243, 1, 151936, stream=stream0)
        del buf3238
    return (buf3244, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 1), (1, 1), device='cuda:0', dtype=torch.int64)
    arg1_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg2_1 = rand_strided((12288, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg3_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.float16)
    arg4_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg5_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg6_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg7_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg8_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg9_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg10_1 = rand_strided((1, 8, 960, 128), (983040, 122880, 128, 1), device='cuda:0', dtype=torch.float16)
    arg11_1 = rand_strided((1, 8, 960, 128), (983040, 122880, 128, 1), device='cuda:0', dtype=torch.float16)
    arg12_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg13_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg14_1 = rand_strided((12288, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg15_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.float16)
    arg16_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg17_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg18_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg19_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg20_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg21_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg22_1 = rand_strided((1, 8, 960, 128), (983040, 122880, 128, 1), device='cuda:0', dtype=torch.float16)
    arg23_1 = rand_strided((1, 8, 960, 128), (983040, 122880, 128, 1), device='cuda:0', dtype=torch.float16)
    arg24_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg25_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg26_1 = rand_strided((12288, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg27_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.float16)
    arg28_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg29_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg30_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg31_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg32_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg33_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg34_1 = rand_strided((1, 8, 960, 128), (983040, 122880, 128, 1), device='cuda:0', dtype=torch.float16)
    arg35_1 = rand_strided((1, 8, 960, 128), (983040, 122880, 128, 1), device='cuda:0', dtype=torch.float16)
    arg36_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg37_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg38_1 = rand_strided((12288, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg39_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.float16)
    arg40_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg41_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg42_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg43_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg44_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg45_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg46_1 = rand_strided((1, 8, 960, 128), (983040, 122880, 128, 1), device='cuda:0', dtype=torch.float16)
    arg47_1 = rand_strided((1, 8, 960, 128), (983040, 122880, 128, 1), device='cuda:0', dtype=torch.float16)
    arg48_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg49_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg50_1 = rand_strided((12288, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg51_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.float16)
    arg52_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg53_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg54_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg55_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg56_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg57_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg58_1 = rand_strided((1, 8, 960, 128), (983040, 122880, 128, 1), device='cuda:0', dtype=torch.float16)
    arg59_1 = rand_strided((1, 8, 960, 128), (983040, 122880, 128, 1), device='cuda:0', dtype=torch.float16)
    arg60_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg61_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg62_1 = rand_strided((12288, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg63_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.float16)
    arg64_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg65_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg66_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg67_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg68_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg69_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg70_1 = rand_strided((1, 8, 960, 128), (983040, 122880, 128, 1), device='cuda:0', dtype=torch.float16)
    arg71_1 = rand_strided((1, 8, 960, 128), (983040, 122880, 128, 1), device='cuda:0', dtype=torch.float16)
    arg72_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg73_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg74_1 = rand_strided((12288, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg75_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.float16)
    arg76_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg77_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg78_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg79_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg80_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg81_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg82_1 = rand_strided((1, 8, 960, 128), (983040, 122880, 128, 1), device='cuda:0', dtype=torch.float16)
    arg83_1 = rand_strided((1, 8, 960, 128), (983040, 122880, 128, 1), device='cuda:0', dtype=torch.float16)
    arg84_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg85_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg86_1 = rand_strided((12288, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg87_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.float16)
    arg88_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg89_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg90_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg91_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg92_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg93_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg94_1 = rand_strided((1, 8, 960, 128), (983040, 122880, 128, 1), device='cuda:0', dtype=torch.float16)
    arg95_1 = rand_strided((1, 8, 960, 128), (983040, 122880, 128, 1), device='cuda:0', dtype=torch.float16)
    arg96_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg97_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg98_1 = rand_strided((12288, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg99_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.float16)
    arg100_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg101_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg102_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg103_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg104_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg105_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg106_1 = rand_strided((1, 8, 960, 128), (983040, 122880, 128, 1), device='cuda:0', dtype=torch.float16)
    arg107_1 = rand_strided((1, 8, 960, 128), (983040, 122880, 128, 1), device='cuda:0', dtype=torch.float16)
    arg108_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg109_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg110_1 = rand_strided((12288, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg111_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.float16)
    arg112_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg113_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg114_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg115_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg116_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg117_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg118_1 = rand_strided((1, 8, 960, 128), (983040, 122880, 128, 1), device='cuda:0', dtype=torch.float16)
    arg119_1 = rand_strided((1, 8, 960, 128), (983040, 122880, 128, 1), device='cuda:0', dtype=torch.float16)
    arg120_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg121_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg122_1 = rand_strided((12288, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg123_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.float16)
    arg124_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg125_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg126_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg127_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg128_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg129_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg130_1 = rand_strided((1, 8, 960, 128), (983040, 122880, 128, 1), device='cuda:0', dtype=torch.float16)
    arg131_1 = rand_strided((1, 8, 960, 128), (983040, 122880, 128, 1), device='cuda:0', dtype=torch.float16)
    arg132_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg133_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg134_1 = rand_strided((12288, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg135_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.float16)
    arg136_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg137_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg138_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg139_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg140_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg141_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg142_1 = rand_strided((1, 8, 960, 128), (983040, 122880, 128, 1), device='cuda:0', dtype=torch.float16)
    arg143_1 = rand_strided((1, 8, 960, 128), (983040, 122880, 128, 1), device='cuda:0', dtype=torch.float16)
    arg144_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg145_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg146_1 = rand_strided((12288, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg147_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.float16)
    arg148_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg149_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg150_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg151_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg152_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg153_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg154_1 = rand_strided((1, 8, 960, 128), (983040, 122880, 128, 1), device='cuda:0', dtype=torch.float16)
    arg155_1 = rand_strided((1, 8, 960, 128), (983040, 122880, 128, 1), device='cuda:0', dtype=torch.float16)
    arg156_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg157_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg158_1 = rand_strided((12288, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg159_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.float16)
    arg160_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg161_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg162_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg163_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg164_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg165_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg166_1 = rand_strided((1, 8, 960, 128), (983040, 122880, 128, 1), device='cuda:0', dtype=torch.float16)
    arg167_1 = rand_strided((1, 8, 960, 128), (983040, 122880, 128, 1), device='cuda:0', dtype=torch.float16)
    arg168_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg169_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg170_1 = rand_strided((12288, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg171_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.float16)
    arg172_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg173_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg174_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg175_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg176_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg177_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg178_1 = rand_strided((1, 8, 960, 128), (983040, 122880, 128, 1), device='cuda:0', dtype=torch.float16)
    arg179_1 = rand_strided((1, 8, 960, 128), (983040, 122880, 128, 1), device='cuda:0', dtype=torch.float16)
    arg180_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg181_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg182_1 = rand_strided((12288, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg183_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.float16)
    arg184_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg185_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg186_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg187_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg188_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg189_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg190_1 = rand_strided((1, 8, 960, 128), (983040, 122880, 128, 1), device='cuda:0', dtype=torch.float16)
    arg191_1 = rand_strided((1, 8, 960, 128), (983040, 122880, 128, 1), device='cuda:0', dtype=torch.float16)
    arg192_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg193_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg194_1 = rand_strided((12288, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg195_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.float16)
    arg196_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg197_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg198_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg199_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg200_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg201_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg202_1 = rand_strided((1, 8, 960, 128), (983040, 122880, 128, 1), device='cuda:0', dtype=torch.float16)
    arg203_1 = rand_strided((1, 8, 960, 128), (983040, 122880, 128, 1), device='cuda:0', dtype=torch.float16)
    arg204_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg205_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg206_1 = rand_strided((12288, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg207_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.float16)
    arg208_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg209_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg210_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg211_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg212_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg213_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg214_1 = rand_strided((1, 8, 960, 128), (983040, 122880, 128, 1), device='cuda:0', dtype=torch.float16)
    arg215_1 = rand_strided((1, 8, 960, 128), (983040, 122880, 128, 1), device='cuda:0', dtype=torch.float16)
    arg216_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg217_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg218_1 = rand_strided((12288, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg219_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.float16)
    arg220_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg221_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg222_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg223_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg224_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg225_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg226_1 = rand_strided((1, 8, 960, 128), (983040, 122880, 128, 1), device='cuda:0', dtype=torch.float16)
    arg227_1 = rand_strided((1, 8, 960, 128), (983040, 122880, 128, 1), device='cuda:0', dtype=torch.float16)
    arg228_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg229_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg230_1 = rand_strided((12288, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg231_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.float16)
    arg232_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg233_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg234_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg235_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg236_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg237_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg238_1 = rand_strided((1, 8, 960, 128), (983040, 122880, 128, 1), device='cuda:0', dtype=torch.float16)
    arg239_1 = rand_strided((1, 8, 960, 128), (983040, 122880, 128, 1), device='cuda:0', dtype=torch.float16)
    arg240_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg241_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg242_1 = rand_strided((12288, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg243_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.float16)
    arg244_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg245_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg246_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg247_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg248_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg249_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg250_1 = rand_strided((1, 8, 960, 128), (983040, 122880, 128, 1), device='cuda:0', dtype=torch.float16)
    arg251_1 = rand_strided((1, 8, 960, 128), (983040, 122880, 128, 1), device='cuda:0', dtype=torch.float16)
    arg252_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg253_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg254_1 = rand_strided((12288, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg255_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.float16)
    arg256_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg257_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg258_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg259_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg260_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg261_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg262_1 = rand_strided((1, 8, 960, 128), (983040, 122880, 128, 1), device='cuda:0', dtype=torch.float16)
    arg263_1 = rand_strided((1, 8, 960, 128), (983040, 122880, 128, 1), device='cuda:0', dtype=torch.float16)
    arg264_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg265_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg266_1 = rand_strided((12288, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg267_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.float16)
    arg268_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg269_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg270_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg271_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg272_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg273_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg274_1 = rand_strided((1, 8, 960, 128), (983040, 122880, 128, 1), device='cuda:0', dtype=torch.float16)
    arg275_1 = rand_strided((1, 8, 960, 128), (983040, 122880, 128, 1), device='cuda:0', dtype=torch.float16)
    arg276_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg277_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg278_1 = rand_strided((12288, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg279_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.float16)
    arg280_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg281_1 = rand_strided((1, 1), (1, 1), device='cuda:0', dtype=torch.int64)
    arg282_1 = rand_strided((151936, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg283_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg285_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg286_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg287_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg288_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg289_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg290_1 = rand_strided((1, 8, 960, 128), (983040, 122880, 128, 1), device='cuda:0', dtype=torch.float16)
    arg291_1 = rand_strided((1, 8, 960, 128), (983040, 122880, 128, 1), device='cuda:0', dtype=torch.float16)
    arg292_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg293_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg294_1 = rand_strided((12288, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg295_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.float16)
    arg296_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg297_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg298_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg299_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg300_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg301_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg302_1 = rand_strided((1, 8, 960, 128), (983040, 122880, 128, 1), device='cuda:0', dtype=torch.float16)
    arg303_1 = rand_strided((1, 8, 960, 128), (983040, 122880, 128, 1), device='cuda:0', dtype=torch.float16)
    arg304_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg305_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg306_1 = rand_strided((12288, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg307_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.float16)
    arg308_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg309_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg310_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg311_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg312_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg313_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg314_1 = rand_strided((1, 8, 960, 128), (983040, 122880, 128, 1), device='cuda:0', dtype=torch.float16)
    arg315_1 = rand_strided((1, 8, 960, 128), (983040, 122880, 128, 1), device='cuda:0', dtype=torch.float16)
    arg316_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg317_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg318_1 = rand_strided((12288, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg319_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.float16)
    arg320_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg321_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg322_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg323_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg324_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg325_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg326_1 = rand_strided((1, 8, 960, 128), (983040, 122880, 128, 1), device='cuda:0', dtype=torch.float16)
    arg327_1 = rand_strided((1, 8, 960, 128), (983040, 122880, 128, 1), device='cuda:0', dtype=torch.float16)
    arg328_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg329_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg330_1 = rand_strided((12288, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg331_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.float16)
    arg332_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg333_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg334_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg335_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg336_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg337_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg338_1 = rand_strided((1, 8, 960, 128), (983040, 122880, 128, 1), device='cuda:0', dtype=torch.float16)
    arg339_1 = rand_strided((1, 8, 960, 128), (983040, 122880, 128, 1), device='cuda:0', dtype=torch.float16)
    arg340_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    arg341_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
