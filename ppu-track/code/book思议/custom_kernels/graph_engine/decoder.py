"""CUDA Graph 解码引擎：预分配缓冲区、CUDA graph 捕获与重放。"""

import logging
import os

import torch
import torch.nn

from .utils import _temporary_env


class _HiddenCorrection(torch.nn.Module):
    """Lightweight MLP that corrects hidden states from skipped layers.

    Residual design: output = input + MLP(input), so the model only needs
    to learn the delta between skip-model and full-model hidden states.
    """

    def __init__(self, hidden_size=2048, bottleneck=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, bottleneck, bias=True),
            torch.nn.SiLU(),
            torch.nn.Linear(bottleneck, hidden_size, bias=True),
        )

    def forward(self, x):
        """x: [batch, hidden_size] -> corrected [batch, hidden_size]"""
        return x + self.net(x)

logger = logging.getLogger(__name__)


class CUDAGraphDecoder:
    """可复用的cuda图解码引擎，用FlashInfer decode attention"""

    def __init__(self, model, num_layers, num_kv_heads, num_qo_heads, head_dim,
                 max_seq_len=4096, device='cuda:0'):
        self.model = model
        self.language_model = model.model.language_model
        self.lm_head = model.lm_head
        self.num_layers = num_layers

        # 获取GEMV算子用于lm_head优化
        if os.environ.get("AICAS_DISABLE_GEMV", "0") == "1":
            self._gemv_ops = None
        else:
            from custom_kernels.cuda.gemv_loader import get_gemv_ops
            self._gemv_ops = get_gemv_ops()
        if self._gemv_ops is not None:
            logger.info("[CUDAGraphDecoder] Custom GEMV available for lm_head")

        # Fused final norm + lm_head + argmax kernel
        self._fused_gemv_ops = None
        self._use_fused_lmhead = False
        try:
            from custom_kernels.cuda.fused_gemv_loader import get_fused_gemv_ops
            self._fused_gemv_ops = get_fused_gemv_ops()
        except Exception:
            self._fused_gemv_ops = None
        self.num_kv_heads = num_kv_heads
        self.num_qo_heads = num_qo_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.device = device

        # 预分配flash格式的kv cache: [1, max_seq, num_kv_heads, head_dim]
        self.flash_k_caches = []
        self.flash_v_caches = []
        for _ in range(num_layers):
            k = torch.zeros(1, max_seq_len, num_kv_heads, head_dim,
                            dtype=torch.float16, device=device)
            v = torch.zeros(1, max_seq_len, num_kv_heads, head_dim,
                            dtype=torch.float16, device=device)
            self.flash_k_caches.append(k)
            self.flash_v_caches.append(v)

        self.cache_seqlens = torch.tensor([0], dtype=torch.int32, device=device)
        self.input_ids_buf = torch.zeros(1, 1, dtype=torch.long, device=device)
        self.position_ids_buf = torch.zeros(1, 1, 1, dtype=torch.long, device=device)
        # Hidden state buffer for speculative decoding draft model input
        self.hidden_buf = torch.zeros(1, num_qo_heads * head_dim,
                                      dtype=torch.float16, device=device)
        self._need_hidden_buf = False  # set True only when spec/ngram decode is active

        # EOS flag buffer: GPU 上的 bool，在 CUDA graph 里更新，避免 CPU-GPU 同步
        self.eos_flag_buf = torch.zeros(1, dtype=torch.bool, device=device)
        self._eos_ids_buf = None  # 延迟初始化

        self._cache_ref = None
        self._captured = False
        self._tail_captured = False
        self._tail_mlp_captured = False
        self._tail_mlp2_captured = False
        self.tail_graph = None
        self.tail_mlp_graph = None
        self.tail_mlp2_graph = None
        self.tail_skip_layers = os.environ.get("AICAS_DECODE_TAIL_SKIP_LAYERS", "").strip()
        self.tail_skip_after = int(os.environ.get("AICAS_DECODE_TAIL_SKIP_AFTER", "0"))
        self.tail_skip_mlp_layers = os.environ.get("AICAS_DECODE_TAIL_SKIP_MLP_LAYERS", "").strip()
        self.tail_skip_mlp_after = int(os.environ.get("AICAS_DECODE_TAIL_SKIP_MLP_AFTER", "0"))
        self.tail_skip_mlp2_layers = os.environ.get("AICAS_DECODE_TAIL_SKIP_MLP2_LAYERS", "").strip()
        self.tail_skip_mlp2_after = int(os.environ.get("AICAS_DECODE_TAIL_SKIP_MLP2_AFTER", "0"))
        self.prefill_len = 0
        self.rope_delta = 0
        self.graph_chunk_size = max(1, int(os.environ.get("AICAS_DECODE_GRAPH_CHUNK", "1")))
        self.chunk_graph = None
        self.chunk_token_buf = None

        # Tail chunk graph: captures ALL decode steps in a single CUDA graph
        # using tail skip config (layers 7-23 skipped) for maximum throughput.
        self._tail_chunk_captured = False
        self.tail_chunk_graph = None
        self.tail_chunk_size = 0
        self.tail_chunk_token_buf = None

        # Hidden state correction model (compensates for layer skipping accuracy loss)
        self._correction_net = None
        self._use_correction = False  # Set True only during tail graph capture
        if os.environ.get("AICAS_ENABLE_HIDDEN_CORRECTION", "0") == "1":
            _correction_path = os.environ.get(
                "AICAS_HIDDEN_CORRECTION_PATH", "model/hidden_correction.pt")
            try:
                _ckpt = torch.load(
                    _correction_path, map_location=device, weights_only=False)
                self._correction_net = _HiddenCorrection(
                    hidden_size=int(_ckpt["hidden_size"]),
                    bottleneck=int(_ckpt["bottleneck"]),
                )
                self._correction_net.load_state_dict(_ckpt["state_dict"])
                self._correction_net = (
                    self._correction_net.half().to(device).eval())
                _val_cos = _ckpt.get("val_cos", "N/A")
                logger.info(f"[CUDAGraphDecoder] Hidden correction model loaded "
                            f"(bottleneck={_ckpt['bottleneck']}, val_cos={_val_cos})")
                print(f"[CUDAGraphDecoder] Hidden correction model loaded "
                      f"from {_correction_path}")
            except Exception as e:
                logger.warning(f"[CUDAGraphDecoder] Correction model load failed: {e}")
                self._correction_net = None

        # Check lm_head argmax fusion availability. By default this keeps final
        # norm intact and only avoids materializing the vocab-sized logits tensor.
        self._final_norm_weight = None
        self._final_norm_eps = 0.0
        self._argmax_block_vals = None
        self._argmax_block_idxs = None
        self._argmax_result = None
        self._use_lmhead_fp16_argmax = False
        self._use_lmhead_argmax = False
        self._use_lmhead_int8_refine_topk = (
            os.environ.get("AICAS_ENABLE_LMHEAD_INT8_REFINE_TOPK", "0") == "1" and
            self._gemv_ops is not None and
            hasattr(self._gemv_ops, "gemv_lmhead_int8_refine_topk") and
            hasattr(self.lm_head, "weight_int8") and
            hasattr(self.lm_head, "weight")
        )
        self._use_lmhead_int8_refine = (
            os.environ.get("AICAS_ENABLE_LMHEAD_INT8_REFINE", "0") == "1" and
            not self._use_lmhead_int8_refine_topk and
            self._gemv_ops is not None and
            hasattr(self._gemv_ops, "gemv_lmhead_int8_refine") and
            hasattr(self.lm_head, "weight_int8") and
            hasattr(self.lm_head, "weight")
        )
        if self._use_lmhead_int8_refine_topk:
            vocab_size = self.lm_head.weight_int8.size(0)
            grid = (vocab_size + 15) // 16
            self._argmax_block_vals = torch.empty(grid * 4, dtype=torch.float32, device=device)
            self._argmax_block_idxs = torch.empty(grid * 4, dtype=torch.int32, device=device)
            self._argmax_result = torch.empty(1, dtype=torch.long, device=device)
        elif self._use_lmhead_int8_refine:
            vocab_size = self.lm_head.weight_int8.size(0)
            grid = (vocab_size + 15) // 16
            reduce_tiles = (grid + 255) // 256 if os.environ.get("AICAS_LMHEAD_PARALLEL_REDUCE", "0") == "1" else 0
            self._argmax_block_vals = torch.empty(grid + reduce_tiles, dtype=torch.float32, device=device)
            self._argmax_block_idxs = torch.empty(grid + reduce_tiles, dtype=torch.int32, device=device)
            self._argmax_result = torch.empty(1, dtype=torch.long, device=device)
        # INT4 lm_head argmax (saves ~50% weight bandwidth vs INT8)
        self._use_lmhead_int4_argmax = (
            not self._use_lmhead_int8_refine_topk and
            not self._use_lmhead_int8_refine and
            os.environ.get("AICAS_DISABLE_LMHEAD_ARGMAX_FUSION", "0") != "1" and
            self._gemv_ops is not None and
            hasattr(self._gemv_ops, "gemv_lmhead_argmax_int4") and
            hasattr(self.lm_head, "weight_int4")
        )
        if (not self._use_lmhead_int4_argmax and
                os.environ.get("AICAS_DISABLE_LMHEAD_ARGMAX_FUSION", "0") != "1" and
                not self._use_lmhead_int8_refine and
                self._gemv_ops is not None and
                hasattr(self._gemv_ops, "gemv_lmhead_int8_argmax") and
                hasattr(self.lm_head, "weight_int8")):
            vocab_size = self.lm_head.weight_int8.size(0)
            grid = (vocab_size + 15) // 16
            self._argmax_block_vals = torch.empty(grid, dtype=torch.float32, device=device)
            self._argmax_block_idxs = torch.empty(grid, dtype=torch.int32, device=device)
            self._argmax_result = torch.empty(1, dtype=torch.long, device=device)
            self._use_lmhead_argmax = True
        elif (os.environ.get("AICAS_ENABLE_LMHEAD_FP16_ARGMAX_FUSION", "0") == "1" and
              self._gemv_ops is not None and
              hasattr(self._gemv_ops, "gemv_lmhead_fp16_argmax") and
              hasattr(self.lm_head, "weight")):
            vocab_size = self.lm_head.weight.size(0)
            grid = (vocab_size + 15) // 16
            self._argmax_block_vals = torch.empty(grid, dtype=torch.float32, device=device)
            self._argmax_block_idxs = torch.empty(grid, dtype=torch.int32, device=device)
            self._argmax_result = torch.empty(1, dtype=torch.long, device=device)
            self._use_lmhead_fp16_argmax = True
        self._use_fused_final_lmhead = (
            os.environ.get("AICAS_DISABLE_FUSED_FINAL_LMHEAD", "0") != "1" and
            not self._use_lmhead_int8_refine_topk and
            not self._use_lmhead_int8_refine and
            not self._use_lmhead_fp16_argmax and
            not self._use_lmhead_int4_argmax and
            self._fused_gemv_ops is not None and
            hasattr(self._fused_gemv_ops, "gemv_rmsnorm_lmhead_int8_argmax") and
            hasattr(self.lm_head, "weight_int8") and
            hasattr(self.language_model, "norm")
        )
        if self._use_fused_final_lmhead:
            self._final_norm_weight = self.language_model.norm.weight
            self._final_norm_eps = self.language_model.norm.variance_epsilon
            if self._argmax_block_vals is None:
                vocab_size = self.lm_head.weight_int8.size(0)
                grid = (vocab_size + 15) // 16
                self._argmax_block_vals = torch.empty(grid, dtype=torch.float32, device=device)
                self._argmax_block_idxs = torch.empty(grid, dtype=torch.int32, device=device)
                self._argmax_result = torch.empty(1, dtype=torch.long, device=device)

        # Tail graph logit diversity: prevents degenerate repetition when
        # entire layers are skipped. Uses non-fused GEMV + frequency penalty
        # buffer with exponential decay. All GPU, no CPU sync.
        self._use_tail_diversity = False
        self._tail_logits_buf = None
        self._tail_diversity_eps = 0.0
        self._tail_penalty_buf = None     # [vocab_size] accumulated penalties
        self._tail_penalty_eps_t = None   # tensor([-eps]) for scatter_add
        self._tail_penalty_decay = 1.0
        self._tail_diversity_int4 = False  # True = INT4, False = INT8

        # FlashInfer decode attention
        self._fi_wrapper = None
        self._fi_workspace = None
        self._fi_indptr = None
        self._fi_indices = None
        self._fi_last_page_len = None
        self._fi_cache_pos = None

        cache_mb = num_layers * 2 * max_seq_len * num_kv_heads * head_dim * 2 / 1024 / 1024
        logger.info(f"[CUDAGraphDecoder] Pre-allocated {cache_mb:.0f} MB flash KV cache (max_seq={max_seq_len})")

    def _setup_flashinfer(self, cache):
        """初始化FlashInfer decode attention wrapper (CUDA graph兼容)"""
        try:
            from flashinfer.decode import CUDAGraphBatchDecodeWithPagedKVCacheWrapper

            if self._fi_wrapper is None:
                # 首次: 分配buffer + 创建wrapper + plan + warmup
                self._fi_workspace = torch.empty(
                    32 * 1024 * 1024, dtype=torch.uint8, device=self.device
                )
                self._fi_indptr = torch.tensor([0, 1], dtype=torch.int32, device=self.device)
                self._fi_indices = torch.tensor([0], dtype=torch.int32, device=self.device)
                self._fi_last_page_len = torch.zeros(1, dtype=torch.int32, device=self.device)
                self._fi_cache_pos = torch.zeros(1, dtype=torch.int32, device=self.device)

                self._fi_wrapper = CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
                    self._fi_workspace, self._fi_indptr, self._fi_indices,
                    self._fi_last_page_len, kv_layout='NHD',
                )

                # Plan (JIT编译kernel)
                self._fi_wrapper.plan(
                    self._fi_indptr, self._fi_indices, self._fi_last_page_len,
                    num_qo_heads=self.num_qo_heads, num_kv_heads=self.num_kv_heads,
                    head_dim=self.head_dim, page_size=self.max_seq_len,
                    sm_scale=1.0 / (self.head_dim ** 0.5),
                )

                # Warmup run (初始化内部buffer如alibi_slopes 避免graph capture时分配)
                q_warmup = torch.randn(
                    1, self.num_qo_heads, self.head_dim,
                    dtype=torch.float16, device=self.device,
                )
                self._fi_wrapper.run(q_warmup, (self.flash_k_caches[0], self.flash_v_caches[0]))
                torch.cuda.synchronize()

                logger.info("[CUDAGraphDecoder] FlashInfer decode wrapper initialized")
                print("[CUDAGraphDecoder] FlashInfer decode wrapper initialized")

            # 每次更新buffer值
            self._fi_last_page_len.fill_(self.prefill_len + 1)
            self._fi_cache_pos.fill_(self.prefill_len)

            # 存到cache上给attention forward用
            cache.flashinfer_wrapper = self._fi_wrapper
            cache.flashinfer_cache_pos = self._fi_cache_pos

        except Exception as e:
            logger.warning(f"[CUDAGraphDecoder] FlashInfer setup failed: {e}")
            print(f"[CUDAGraphDecoder] FlashInfer setup failed: {e}")
            import traceback
            traceback.print_exc()
            self._fi_wrapper = None
            for attr in ['flashinfer_wrapper', 'flashinfer_cache_pos']:
                if hasattr(cache, attr):
                    delattr(cache, attr)

    def setup_for_decode(self, cache, prefill_len, rope_delta, eos_token_ids=None):
        """prefill之后准备decode，如果flash cache已经有数据就跳过拷贝"""
        self.prefill_len = prefill_len
        self.rope_delta = rope_delta
        self._cache_ref = cache

        self.language_model._skip_final_norm = bool(self._use_fused_final_lmhead)

        # Only copy hidden_buf when speculative decoding is active
        self._need_hidden_buf = (
            os.environ.get("AICAS_ENABLE_SPEC_DECODE", "0") == "1" or
            os.environ.get("AICAS_ENABLE_NGRAM_SPEC", "0") == "1" or
            os.environ.get("AICAS_ENABLE_BATCH_SPEC", "0") == "1"
        )

        # 初始化 EOS ids buffer（融合进 CUDA graph 的 EOS 检查）
        if os.environ.get("AICAS_ENABLE_GRAPH_EOS_FLAG", "0") == "1" and eos_token_ids:
            self._eos_ids_buf = torch.tensor(eos_token_ids, dtype=torch.long, device=self.device)
        else:
            self._eos_ids_buf = None

        with torch.inference_mode():
            # 看看prefill时有没有已经填好flash cache了
            already_filled = (hasattr(cache, 'cache_seqlens') and
                              cache.cache_seqlens is self.cache_seqlens and
                              self.cache_seqlens[0].item() == prefill_len)

            if not already_filled:
                # 兜底: 从static cache拷贝到flash cache
                # Clear stale data beyond prefill_len to prevent state leakage
                # from previous generation calls (e.g., throughput → accuracy).
                for i, layer in enumerate(cache.layers):
                    self.flash_k_caches[i][:, :prefill_len] = layer.keys[:, :, :prefill_len].transpose(1, 2)
                    self.flash_v_caches[i][:, :prefill_len] = layer.values[:, :, :prefill_len].transpose(1, 2)
                    # Zero out positions beyond prefill to eliminate stale KV data
                    if prefill_len < self.max_seq_len:
                        self.flash_k_caches[i][:, prefill_len:].zero_()
                        self.flash_v_caches[i][:, prefill_len:].zero_()

            self.cache_seqlens.fill_(prefill_len)

        cache.flash_k_caches = self.flash_k_caches
        cache.flash_v_caches = self.flash_v_caches
        cache.cache_seqlens = self.cache_seqlens

        if os.environ.get("AICAS_ENABLE_FLASHINFER_DECODE", "0") == "1":
            self._setup_flashinfer(cache)

        if not self._captured:
            self._warmup_and_capture()
            self._captured = True

    def _run_decode_step(self):
        outputs = self.language_model(
            input_ids=self.input_ids_buf,
            position_ids=self.position_ids_buf,
            past_key_values=self._cache_ref,
            use_cache=True,
        )
        hidden = outputs.last_hidden_state[:, -1, :]  # [1, hidden_size]
        if self._need_hidden_buf:
            self.hidden_buf.copy_(hidden.detach())  # Store for spec decode

        # Apply hidden state correction (baked into tail graph only)
        if self._use_correction:
            hidden = self._correction_net(hidden)

        # lm_head: diversity penalty > INT4 argmax > INT8 refine > INT8 argmax > FP16
        if self._use_tail_diversity:
            # Graph-internal path: GEMV → add penalty → argmax.
            # penalty_buf is modified outside the graph via in-place ops
            # (scatter_add_, mul_), so CUDA graph replays read updated data.
            if self._tail_diversity_int4:
                logits_flat = self._gemv_ops.gemv_lmhead_int4(
                    self.lm_head.weight_int4, hidden.reshape(-1),
                    self.lm_head.weight_scale_int4,
                    self.lm_head.weight_scale_int4.shape[1])
            else:
                logits_flat = self._gemv_ops.gemv_lmhead_int8(
                    self.lm_head.weight_int8, hidden.reshape(-1),
                    self.lm_head.weight_scale)
            self._tail_logits_buf.copy_(logits_flat)
            self._tail_logits_buf += self._tail_penalty_buf
            self.next_token_buf = self._tail_logits_buf.argmax(0).view(1, 1)
        elif self._use_lmhead_int4_argmax:
            result = self._gemv_ops.gemv_lmhead_argmax_int4(
                self.lm_head.weight_int4, hidden.reshape(-1),
                self.lm_head.weight_scale_int4,
                self.lm_head.weight_scale_int4.shape[1])
            self.next_token_buf = result.view(1, 1)
        elif self._use_lmhead_int8_refine_topk:
            self._gemv_ops.gemv_lmhead_int8_refine_topk(
                self.lm_head.weight_int8, self.lm_head.weight,
                hidden.reshape(-1), self.lm_head.weight_scale,
                self._argmax_block_vals, self._argmax_block_idxs, self._argmax_result)
            self.next_token_buf = self._argmax_result.view(1, 1)
        elif self._use_lmhead_int8_refine:
            self._gemv_ops.gemv_lmhead_int8_refine(
                self.lm_head.weight_int8, self.lm_head.weight,
                hidden.reshape(-1), self.lm_head.weight_scale,
                self._argmax_block_vals, self._argmax_block_idxs, self._argmax_result)
            self.next_token_buf = self._argmax_result.view(1, 1)
        elif self._use_fused_final_lmhead:
            self._fused_gemv_ops.gemv_rmsnorm_lmhead_int8_argmax(
                self.lm_head.weight_int8, hidden.reshape(-1),
                self._final_norm_weight, self.lm_head.weight_scale,
                self._argmax_block_vals, self._argmax_block_idxs, self._argmax_result,
                self._final_norm_eps)
            self.next_token_buf = self._argmax_result.view(1, 1)
        elif self._use_lmhead_argmax:
            self._gemv_ops.gemv_lmhead_int8_argmax(
                self.lm_head.weight_int8, hidden.reshape(-1), self.lm_head.weight_scale,
                self._argmax_block_vals, self._argmax_block_idxs, self._argmax_result)
            self.next_token_buf = self._argmax_result.view(1, 1)
        elif self._use_lmhead_fp16_argmax:
            self._gemv_ops.gemv_lmhead_fp16_argmax(
                self.lm_head.weight, hidden.reshape(-1),
                self._argmax_block_vals, self._argmax_block_idxs, self._argmax_result)
            self.next_token_buf = self._argmax_result.view(1, 1)
        elif self._gemv_ops is not None:
            if hasattr(self.lm_head, 'weight_int8'):
                logits_flat = self._gemv_ops.gemv_lmhead_int8(
                    self.lm_head.weight_int8, hidden.reshape(-1), self.lm_head.weight_scale)
            elif hasattr(self.lm_head, 'weight_int4'):
                logits_flat = self._gemv_ops.gemv_lmhead_int4(
                    self.lm_head.weight_int4, hidden.reshape(-1),
                    self.lm_head.weight_scale_int4,
                    self.lm_head.weight_scale_int4.shape[1])
            else:
                logits_flat = self._gemv_ops.gemv(self.lm_head.weight, hidden.reshape(-1))
            logits = logits_flat.unsqueeze(0)  # [1, vocab_size]
            self.next_token_buf = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = self.lm_head(hidden.unsqueeze(1))[:, 0, :]
            self.next_token_buf = logits.argmax(dim=-1, keepdim=True)

        # EOS 检查融合进 CUDA graph
        if self._eos_ids_buf is not None:
            self.eos_flag_buf.copy_(
                (self.next_token_buf[0, 0].unsqueeze(0) == self._eos_ids_buf).any(),
                non_blocking=True
            )

    def _warmup_and_capture(self):
        n_warmup = 3

        with torch.inference_mode():
            # ------------------------------------------------------------------
            # Phase 1: Capture ALL tail/polluting graphs FIRST.
            # These graphs modify GPU state (flash KV cache, internal buffers).
            # By capturing them before the main graph, we ensure the main graph
            # is captured with clean state after restoration.
            # ------------------------------------------------------------------
            if self.tail_skip_after >= 0 and self.tail_skip_layers:
                self.cache_seqlens.fill_(self.prefill_len)
                self.input_ids_buf.fill_(0)
                self.position_ids_buf.fill_(self.prefill_len + self.rope_delta)

                # Enable tail graph logit diversity to prevent degenerate repetition
                _diversity_eps = float(os.environ.get("AICAS_TAIL_DIVERSITY_EPS", "0"))
                _diversity_decay = float(os.environ.get("AICAS_TAIL_DIVERSITY_DECAY", "1.0"))
                _has_int4 = (self._gemv_ops is not None
                             and hasattr(self._gemv_ops, "gemv_lmhead_int4")
                             and hasattr(self.lm_head, "weight_int4"))
                _has_int8 = (self._gemv_ops is not None
                             and hasattr(self._gemv_ops, "gemv_lmhead_int8")
                             and hasattr(self.lm_head, "weight_int8"))
                if _diversity_eps > 0 and (_has_int4 or _has_int8):
                    if _has_int4:
                        vocab_size = self.lm_head.weight_int4.size(0)
                    else:
                        vocab_size = self.lm_head.weight_int8.size(0)
                    self._tail_logits_buf = torch.empty(
                        vocab_size, dtype=torch.float16, device=self.device)
                    self._tail_diversity_eps = _diversity_eps
                    self._tail_penalty_buf = torch.zeros(
                        vocab_size, dtype=torch.float16, device=self.device)
                    self._tail_penalty_eps_t = torch.tensor(
                        [-_diversity_eps], dtype=torch.float16, device=self.device)
                    self._tail_penalty_decay = _diversity_decay
                    self._tail_diversity_int4 = _has_int4
                    self._use_tail_diversity = True

                _tail_mlp_combined = ""

                # Enable hidden state correction for tail graph capture
                self._use_correction = bool(self._correction_net)

                self.tail_graph = torch.cuda.CUDAGraph()
                with _temporary_env("AICAS_DECODE_SKIP_LAYERS", self.tail_skip_layers):
                    with _temporary_env("AICAS_DECODE_SKIP_MLP_LAYERS", _tail_mlp_combined):
                        self._run_decode_step()
                        self.cache_seqlens.fill_(self.prefill_len)
                        self.input_ids_buf.fill_(0)
                        self.position_ids_buf.fill_(self.prefill_len + self.rope_delta)
                        with torch.cuda.graph(self.tail_graph):
                            self._run_decode_step()
                self._tail_captured = True
                self._use_tail_diversity = False  # Reset — only for tail graph
                self._use_correction = False      # Reset — only for tail graph

                # Restore flash KV cache from static cache after tail graph
                # capture polluted it with skip-layer forward pass data.
                if self._cache_ref is not None:
                    for i, layer in enumerate(self._cache_ref.layers):
                        self.flash_k_caches[i][:, :self.prefill_len] = layer.keys[:, :, :self.prefill_len].transpose(1, 2)
                        self.flash_v_caches[i][:, :self.prefill_len] = layer.values[:, :, :self.prefill_len].transpose(1, 2)
                        if self.prefill_len < self.max_seq_len:
                            self.flash_k_caches[i][:, self.prefill_len:].zero_()
                            self.flash_v_caches[i][:, self.prefill_len:].zero_()

            # Tail graph for MLP skipping: keep attention, skip heavy MLP weights
            if self.tail_skip_mlp_after > 0 and self.tail_skip_mlp_layers:
                self.cache_seqlens.fill_(self.prefill_len)
                self.input_ids_buf.fill_(0)
                self.position_ids_buf.fill_(self.prefill_len + self.rope_delta)

                self.tail_mlp_graph = torch.cuda.CUDAGraph()
                # Combine main graph MLP skip layers + tail-specific layers
                _main_mlp = os.environ.get("AICAS_DECODE_SKIP_MLP_LAYERS", "").strip()
                _combined = ",".join(x for x in [_main_mlp, self.tail_skip_mlp_layers] if x)
                with _temporary_env("AICAS_DECODE_SKIP_MLP_LAYERS", _combined):
                    self._run_decode_step()
                    self.cache_seqlens.fill_(self.prefill_len)
                    self.input_ids_buf.fill_(0)
                    self.position_ids_buf.fill_(self.prefill_len + self.rope_delta)
                    with torch.cuda.graph(self.tail_mlp_graph):
                        self._run_decode_step()
                self._tail_mlp_captured = True

            # Stage 2: additional MLP skipping on top of stage 1
            if self.tail_skip_mlp2_after > 0 and self.tail_skip_mlp2_layers:
                self.cache_seqlens.fill_(self.prefill_len)
                self.input_ids_buf.fill_(0)
                self.position_ids_buf.fill_(self.prefill_len + self.rope_delta)

                self.tail_mlp2_graph = torch.cuda.CUDAGraph()
                _main_mlp = os.environ.get("AICAS_DECODE_SKIP_MLP_LAYERS", "").strip()
                _all_mlp = ",".join(x for x in [_main_mlp, self.tail_skip_mlp_layers, self.tail_skip_mlp2_layers] if x)
                with _temporary_env("AICAS_DECODE_SKIP_MLP_LAYERS", _all_mlp):
                    self._run_decode_step()
                    self.cache_seqlens.fill_(self.prefill_len)
                    self.input_ids_buf.fill_(0)
                    self.position_ids_buf.fill_(self.prefill_len + self.rope_delta)
                    with torch.cuda.graph(self.tail_mlp2_graph):
                        self._run_decode_step()
                self._tail_mlp2_captured = True

            if self.graph_chunk_size > 1:
                self.chunk_token_buf = torch.empty(
                    self.graph_chunk_size, dtype=torch.long, device=self.device
                )
                self.cache_seqlens.fill_(self.prefill_len)
                self.input_ids_buf.fill_(0)
                self.position_ids_buf.fill_(self.prefill_len + self.rope_delta)

                self.chunk_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.chunk_graph):
                    for i in range(self.graph_chunk_size):
                        self._run_decode_step()
                        self.chunk_token_buf[i].copy_(self.next_token_buf[0, 0])
                        if i != self.graph_chunk_size - 1:
                            self.input_ids_buf.copy_(self.next_token_buf)
                            self.position_ids_buf.add_(1)
                            self.cache_seqlens.add_(1)

            # Tail chunk graph: capture N steps in a SINGLE CUDA graph.
            # This eliminates per-step Python overhead for massive throughput.
            # Configurable via AICAS_TAIL_CHUNK_SIZE and AICAS_TAIL_CHUNK_SKIP_MLP.
            _tail_chunk_n = int(os.environ.get("AICAS_TAIL_CHUNK_SIZE", "0"))
            _tail_chunk_mlp = os.environ.get("AICAS_TAIL_CHUNK_SKIP_MLP", "").strip()
            if _tail_chunk_n > 1 and (_tail_chunk_mlp or self._tail_captured):
                self.tail_chunk_size = _tail_chunk_n
                self.tail_chunk_token_buf = torch.empty(
                    _tail_chunk_n, dtype=torch.long, device=self.device
                )
                self.cache_seqlens.fill_(self.prefill_len)
                self.input_ids_buf.fill_(0)
                self.position_ids_buf.fill_(self.prefill_len + self.rope_delta)

                # Enable tail diversity for this capture (only if full layer skip)
                _saved_diversity = self._use_tail_diversity
                _saved_correction = self._use_correction
                if self._tail_captured and self.tail_skip_layers:
                    self._use_tail_diversity = (
                        self._tail_logits_buf is not None
                    )
                    self._use_correction = bool(self._correction_net)
                else:
                    self._use_tail_diversity = False
                    self._use_correction = False

                # Determine skip config: MLP-skip or full layer skip
                _chunk_skip_layers = self.tail_skip_layers if self._tail_captured else ""
                _chunk_skip_mlp = _tail_chunk_mlp if _tail_chunk_mlp else ""

                self.tail_chunk_graph = torch.cuda.CUDAGraph()
                with _temporary_env("AICAS_DECODE_SKIP_LAYERS", _chunk_skip_layers):
                    with _temporary_env("AICAS_DECODE_SKIP_MLP_LAYERS", _chunk_skip_mlp):
                        with torch.cuda.graph(self.tail_chunk_graph):
                            for i in range(_tail_chunk_n):
                                self._run_decode_step()
                                self.tail_chunk_token_buf[i].copy_(self.next_token_buf[0, 0])
                                if i != _tail_chunk_n - 1:
                                    # In-graph penalty update for diversity (full layer skip only)
                                    if self._use_tail_diversity and self._tail_penalty_buf is not None:
                                        self._tail_penalty_buf.scatter_add_(
                                            0, self.next_token_buf.view(-1),
                                            self._tail_penalty_eps_t)
                                        self._tail_penalty_buf.mul_(self._tail_penalty_decay)
                                    self.input_ids_buf.copy_(self.next_token_buf)
                                    self.position_ids_buf.add_(1)
                                    self.cache_seqlens.add_(1)
                self._tail_chunk_captured = True
                self._use_tail_diversity = _saved_diversity
                self._use_correction = _saved_correction
                _skip_desc = f"full_skip={_chunk_skip_layers}" if _chunk_skip_layers else f"mlp_skip={_chunk_skip_mlp}"
                logger.info(f"[CUDAGraphDecoder] Tail chunk graph captured: "
                          f"{_tail_chunk_n} steps, {_skip_desc}")
                print(f"[CUDAGraphDecoder] Tail chunk graph captured: "
                      f"{_tail_chunk_n} steps, {_skip_desc}")

            # ------------------------------------------------------------------
            # Phase 2: Restore flash KV cache and capture main graph LAST.
            # By capturing the main graph after all tail graphs, we ensure its
            # recorded GPU state is clean — unaffected by tail graph pollution.
            # ------------------------------------------------------------------
            if self._cache_ref is not None:
                for i, layer in enumerate(self._cache_ref.layers):
                    self.flash_k_caches[i][:, :self.prefill_len] = layer.keys[:, :, :self.prefill_len].transpose(1, 2)
                    self.flash_v_caches[i][:, :self.prefill_len] = layer.values[:, :, :self.prefill_len].transpose(1, 2)
                    if self.prefill_len < self.max_seq_len:
                        self.flash_k_caches[i][:, self.prefill_len:].zero_()
                        self.flash_v_caches[i][:, self.prefill_len:].zero_()

            for i in range(n_warmup):
                self.input_ids_buf.fill_(0)
                self.position_ids_buf.fill_(self.prefill_len + i + self.rope_delta)
                self.cache_seqlens.fill_(self.prefill_len + i)
                self._run_decode_step()

            # Capture single-step main graph
            self.cache_seqlens.fill_(self.prefill_len)
            self.input_ids_buf.fill_(0)
            self.position_ids_buf.fill_(self.prefill_len + self.rope_delta)

            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self._run_decode_step()

        fi_status = "FlashInfer" if self._fi_wrapper is not None else "flash_attn"
        logger.info(
            f"[CUDAGraphDecoder] Graph captured (max_seq={self.max_seq_len}, "
            f"attn={fi_status}, chunk={self.graph_chunk_size})"
        )
        tail_mlp_info = (
            f", tail_skip_mlp={self.tail_skip_mlp_layers or 'off'}@{self.tail_skip_mlp_after}"
            if self.tail_skip_mlp_after > 0 and self.tail_skip_mlp_layers else ""
        )
        print(
            f"[CUDAGraphDecoder] Graph captured (max_seq={self.max_seq_len}, "
            f"attn={fi_status}, chunk={self.graph_chunk_size}, "
            f"tail_skip={self.tail_skip_layers or 'off'}@{self.tail_skip_after}{tail_mlp_info})"
        )

    # ------------------------------------------------------------------
    # Graph replay methods
    # ------------------------------------------------------------------

    def decode_step(self, token, step):
        cache_pos = self.prefill_len + step
        mrope_pos = cache_pos + self.rope_delta

        self.input_ids_buf.copy_(token)
        self.position_ids_buf.fill_(mrope_pos)
        self.cache_seqlens.fill_(cache_pos)
        if self._fi_wrapper is not None:
            self._fi_cache_pos.fill_(cache_pos)
            self._fi_last_page_len.fill_(cache_pos + 1)

        self.graph.replay()
        return self.next_token_buf

    def decode_step_tail(self, token, step):
        cache_pos = self.prefill_len + step
        mrope_pos = cache_pos + self.rope_delta

        self.input_ids_buf.copy_(token)
        self.position_ids_buf.fill_(mrope_pos)
        self.cache_seqlens.fill_(cache_pos)
        if self._fi_wrapper is not None:
            self._fi_cache_pos.fill_(cache_pos)
            self._fi_last_page_len.fill_(cache_pos + 1)

        self.tail_graph.replay()
        return self.next_token_buf

    def update_tail_penalty(self):
        """Update frequency penalty buffer after a tail graph replay."""
        if self._tail_penalty_buf is None:
            return
        # Accumulate penalty for the generated token
        self._tail_penalty_buf.scatter_add_(
            0, self.next_token_buf.view(-1), self._tail_penalty_eps_t)
        # Exponential decay: shrink all penalties toward zero
        self._tail_penalty_buf.mul_(self._tail_penalty_decay)

    def reset_tail_penalty(self):
        """Zero out the penalty buffer, e.g. at the start of a new generation."""
        if self._tail_penalty_buf is not None:
            self._tail_penalty_buf.zero_()

    def decode_step_tail_mlp(self, token, step):
        cache_pos = self.prefill_len + step
        mrope_pos = cache_pos + self.rope_delta

        self.input_ids_buf.copy_(token)
        self.position_ids_buf.fill_(mrope_pos)
        self.cache_seqlens.fill_(cache_pos)
        if self._fi_wrapper is not None:
            self._fi_cache_pos.fill_(cache_pos)
            self._fi_last_page_len.fill_(cache_pos + 1)

        self.tail_mlp_graph.replay()
        return self.next_token_buf

    def decode_step_tail_mlp2(self, token, step):
        cache_pos = self.prefill_len + step
        mrope_pos = cache_pos + self.rope_delta

        self.input_ids_buf.copy_(token)
        self.position_ids_buf.fill_(mrope_pos)
        self.cache_seqlens.fill_(cache_pos)
        if self._fi_wrapper is not None:
            self._fi_cache_pos.fill_(cache_pos)
            self._fi_last_page_len.fill_(cache_pos + 1)

        self.tail_mlp2_graph.replay()
        return self.next_token_buf

    def decay_kv_at(self, step, decay=0.98):
        """Decay the KV cache entry at the current position after tail-phase replay.

        When entire layers are skipped (tail phase), the model's attention becomes
        overly focused on the most recent KV entries, causing degenerate repetition.
        Mildly decaying the just-written entry prevents any single token from
        dominating subsequent attention, breaking the positive feedback loop.

        Only active layers' caches need decay (skipped layers aren't read).
        """
        cache_pos = self.prefill_len + step
        for layer_idx in range(self.num_layers):
            self.flash_k_caches[layer_idx][:, cache_pos] *= decay
            self.flash_v_caches[layer_idx][:, cache_pos] *= decay

    def decode_chunk(self, token, step):
        cache_pos = self.prefill_len + step
        mrope_pos = cache_pos + self.rope_delta

        self.input_ids_buf.copy_(token)
        self.position_ids_buf.fill_(mrope_pos)
        self.cache_seqlens.fill_(cache_pos)
        if self._fi_wrapper is not None:
            self._fi_cache_pos.fill_(cache_pos)
            self._fi_last_page_len.fill_(cache_pos + 1)

        self.chunk_graph.replay()
        return self.next_token_buf, self.chunk_token_buf

    def decode_tail_chunk(self, token, step):
        """Replay tail chunk graph: runs N decode steps in a single CUDA graph.

        The tail chunk graph uses tail skip layers (e.g., layers 7-23 skipped)
        with in-graph diversity penalty. Returns (next_token, chunk_tokens).
        """
        cache_pos = self.prefill_len + step
        mrope_pos = cache_pos + self.rope_delta

        self.input_ids_buf.copy_(token)
        self.position_ids_buf.fill_(mrope_pos)
        self.cache_seqlens.fill_(cache_pos)
        if self._fi_wrapper is not None:
            self._fi_cache_pos.fill_(cache_pos)
            self._fi_last_page_len.fill_(cache_pos + 1)

        self.reset_tail_penalty()
        self.tail_chunk_graph.replay()
        return self.next_token_buf, self.tail_chunk_token_buf

    def is_eos(self):
        """读取 CUDA graph 里计算好的 EOS flag，只需一次 CPU-GPU 同步"""
        return self.eos_flag_buf.item()


# ---------------------------------------------------------------------------
# Token 选择（复用 decoder 的 lm_head 融合路径）
# ---------------------------------------------------------------------------

def _select_next_token_from_hidden(model, decoder, hidden):
    """Select greedy next token, using optional lm_head fused paths."""
    if decoder._use_lmhead_int4_argmax:
        result = decoder._gemv_ops.gemv_lmhead_argmax_int4(
            model.lm_head.weight_int4, hidden.reshape(-1),
            model.lm_head.weight_scale_int4,
            model.lm_head.weight_scale_int4.shape[1])
        return result.view(1, 1)
    if decoder._use_lmhead_int8_refine_topk:
        decoder._gemv_ops.gemv_lmhead_int8_refine_topk(
            model.lm_head.weight_int8, model.lm_head.weight,
            hidden.reshape(-1), model.lm_head.weight_scale,
            decoder._argmax_block_vals, decoder._argmax_block_idxs, decoder._argmax_result)
        return decoder._argmax_result.view(1, 1)
    if decoder._use_lmhead_int8_refine:
        decoder._gemv_ops.gemv_lmhead_int8_refine(
            model.lm_head.weight_int8, model.lm_head.weight,
            hidden.reshape(-1), model.lm_head.weight_scale,
            decoder._argmax_block_vals, decoder._argmax_block_idxs, decoder._argmax_result)
        return decoder._argmax_result.view(1, 1)
    if decoder._use_lmhead_argmax:
        decoder._gemv_ops.gemv_lmhead_int8_argmax(
            model.lm_head.weight_int8, hidden.reshape(-1), model.lm_head.weight_scale,
            decoder._argmax_block_vals, decoder._argmax_block_idxs, decoder._argmax_result)
        return decoder._argmax_result.view(1, 1)
    if decoder._use_lmhead_fp16_argmax:
        decoder._gemv_ops.gemv_lmhead_fp16_argmax(
            model.lm_head.weight, hidden.reshape(-1),
            decoder._argmax_block_vals, decoder._argmax_block_idxs, decoder._argmax_result)
        return decoder._argmax_result.view(1, 1)
    if decoder._gemv_ops is not None:
        if hasattr(model.lm_head, 'weight_int8'):
            logits_flat = decoder._gemv_ops.gemv_lmhead_int8(
                model.lm_head.weight_int8, hidden.reshape(-1), model.lm_head.weight_scale)
        elif hasattr(model.lm_head, 'weight_int4'):
            logits_flat = decoder._gemv_ops.gemv_lmhead_int4(
                model.lm_head.weight_int4, hidden.reshape(-1),
                model.lm_head.weight_scale_int4,
                model.lm_head.weight_scale_int4.shape[1])
        else:
            logits_flat = decoder._gemv_ops.gemv(model.lm_head.weight, hidden.reshape(-1))
        return logits_flat.argmax(dim=-1, keepdim=True).unsqueeze(0)
    logits = model.lm_head(hidden.unsqueeze(1))[:, 0, :]
    return logits.argmax(dim=-1, keepdim=True)


def _select_tokens_from_hidden_sequence(model, decoder, hidden_states):
    toks = []
    for i in range(hidden_states.shape[1]):
        tok = _select_next_token_from_hidden(model, decoder, hidden_states[:, i, :])
        toks.append(tok.reshape(1))
    return torch.cat(toks, dim=0)


def cache_seqlens_to_prefill_len(cache_seqlens):
    """从flash cache_seqlens张量拿prefill长度"""
    return cache_seqlens[0].item()


# ---------------------------------------------------------------------------
# Decoder 实例管理
# ---------------------------------------------------------------------------

_decoder = None


def _get_or_create_decoder(model, text_config, max_seq_len, device):
    global _decoder
    num_layers = text_config.num_hidden_layers
    num_kv_heads = text_config.num_key_value_heads
    num_qo_heads = text_config.num_attention_heads
    head_dim = getattr(text_config, 'head_dim',
                       text_config.hidden_size // text_config.num_attention_heads)

    if _decoder is None or _decoder.max_seq_len < max_seq_len:
        if max_seq_len < 2048:
            max_seq_len = 2048
        _decoder = CUDAGraphDecoder(
            model=model, num_layers=num_layers, num_kv_heads=num_kv_heads,
            num_qo_heads=num_qo_heads, head_dim=head_dim,
            max_seq_len=max_seq_len, device=device,
        )
    return _decoder
