#include <torch/extension.h>

#include <algorithm>
#include <ATen/ops/conv3d.h>
#include <ATen/ops/embedding.h>
#include <ATen/ops/gelu.h>
#include <ATen/ops/layer_norm.h>
#include <ATen/ops/linear.h>
#include <ATen/ops/mean.h>
#include <ATen/ops/pow.h>
#include <ATen/ops/rsqrt.h>
#include <ATen/ops/scaled_dot_product_attention.h>
#include <ATen/ops/silu.h>
#include <pybind11/pybind11.h>

#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

std::atomic<int64_t>& cpp_cache_update_counter() {
  static std::atomic<int64_t> counter{0};
  return counter;
}

std::atomic<int64_t>& prefill_kv_cache_write_counter() {
  static std::atomic<int64_t> counter{0};
  return counter;
}

std::atomic<int64_t>& python_cache_update_fallback_counter() {
  static std::atomic<int64_t> counter{0};
  return counter;
}

std::atomic<int64_t>& prefill_qk_rope_counter() {
  static std::atomic<int64_t> counter{0};
  return counter;
}

std::atomic<int64_t>& prefill_rope_counter() {
  static std::atomic<int64_t> counter{0};
  return counter;
}

std::atomic<int64_t>& add_rmsnorm_counter() {
  static std::atomic<int64_t> counter{0};
  return counter;
}

std::atomic<int64_t>& rmsnorm_counter() {
  static std::atomic<int64_t> counter{0};
  return counter;
}

std::atomic<int64_t>& swiglu_counter() {
  static std::atomic<int64_t> counter{0};
  return counter;
}

std::atomic<int64_t>& vision_runtime_forward_counter() {
  static std::atomic<int64_t> counter{0};
  return counter;
}

std::atomic<int64_t>& vision_get_image_features_counter() {
  static std::atomic<int64_t> counter{0};
  return counter;
}

std::atomic<int64_t>& vision_setup_cache_hit_counter() {
  static std::atomic<int64_t> counter{0};
  return counter;
}

std::atomic<int64_t>& vision_setup_cache_miss_counter() {
  static std::atomic<int64_t> counter{0};
  return counter;
}

py::tuple launch_prefill_rope_cached_triton(
    py::object launcher_obj,
    py::object function_obj,
    py::object packed_metadata_obj,
    py::object cuda_current_stream_obj,
    int64_t block_d,
    const torch::Tensor& position_ids,
    const torch::Tensor& inv_freq,
    const torch::Tensor& out_cos,
    const torch::Tensor& out_sin,
    int64_t seq_len,
    double attention_scaling,
    int64_t len_h,
    int64_t len_w);

bool cpp_runtime_validate_enabled() {
  static bool enabled = []() {
    const char* env = std::getenv("AICAS_CPP_RUNTIME_VALIDATE");
    return env != nullptr && std::string(env) == "1";
  }();
  return enabled;
}

bool cpp_vision_setup_cache_enabled() {
  static bool enabled = []() {
    const char* env = std::getenv("AICAS_ENABLE_CPP_VISION_SETUP_CACHE");
    return env == nullptr || std::string(env) == "1";
  }();
  return enabled;
}

bool cpp_vision_patch_linear_enabled() {
  static bool enabled = []() {
    const char* env = std::getenv("AICAS_ENABLE_CPP_VISION_PATCH_LINEAR");
    return env == nullptr || std::string(env) == "1";
  }();
  return enabled;
}

bool cpp_text_interlayer_add_rmsnorm_enabled() {
  static bool enabled = []() {
    const char* env = std::getenv("AICAS_ENABLE_CPP_TEXT_INTERLAYER_ADD_RMSNORM");
    return env == nullptr || std::string(env) == "1";
  }();
  return enabled;
}

bool cpp_text_final_add_rmsnorm_enabled() {
  static bool enabled = []() {
    const char* env = std::getenv("AICAS_ENABLE_CPP_TEXT_FINAL_ADD_RMSNORM");
    return env == nullptr || std::string(env) == "1";
  }();
  return enabled;
}

py::object call_with_kwargs(py::object callable, py::tuple args, py::dict kwargs) {
  PyObject* result_ptr = PyObject_Call(callable.ptr(), args.ptr(), kwargs.ptr());
  if (result_ptr == nullptr) {
    throw py::error_already_set();
  }
  return py::reinterpret_steal<py::object>(result_ptr);
}

py::object call_vectorcall(
    PyObject* callable_ptr,
    PyObject* const* args,
    size_t positional_count,
    PyObject* kwnames_ptr) {
#if PY_VERSION_HEX >= 0x03080000
  PyObject* result_ptr = PyObject_Vectorcall(callable_ptr, args, positional_count, kwnames_ptr);
  if (result_ptr == nullptr) {
    throw py::error_already_set();
  }
  return py::reinterpret_steal<py::object>(result_ptr);
#else
  py::tuple py_args(positional_count);
  for (size_t i = 0; i < positional_count; ++i) {
    PyObject* arg = args[i];
    Py_INCREF(arg);
    PyTuple_SET_ITEM(py_args.ptr(), i, arg);
  }
  py::dict py_kwargs;
  if (kwnames_ptr != nullptr) {
    auto kwnames = py::reinterpret_borrow<py::tuple>(kwnames_ptr);
    auto kw_count = static_cast<size_t>(py::len(kwnames));
    for (size_t i = 0; i < kw_count; ++i) {
      auto key = py::reinterpret_borrow<py::object>(PyTuple_GET_ITEM(kwnames.ptr(), i));
      auto value = py::reinterpret_borrow<py::object>(args[positional_count + i]);
      py_kwargs[key] = value;
    }
  }
  return call_with_kwargs(py::reinterpret_borrow<py::object>(callable_ptr), py_args, py_kwargs);
#endif
}

bool looks_like_triton_launcher_abi_type_error(const py::error_already_set& err) {
  if (!err.matches(PyExc_TypeError)) {
    return false;
  }
  std::string message = err.what();
  return message.find("function takes exactly") != std::string::npos ||
         message.find("argument") != std::string::npos;
}

py::object call_triton_launcher_compat(
    PyObject* callable_ptr,
    PyObject* const* args,
    size_t positional_count,
    bool retry_without_last_arg) {
  try {
    return call_vectorcall(callable_ptr, args, positional_count, nullptr);
  } catch (py::error_already_set& err) {
    if (!retry_without_last_arg || positional_count == 0 ||
        !looks_like_triton_launcher_abi_type_error(err)) {
      throw;
    }
  }
  return call_vectorcall(callable_ptr, args, positional_count - 1, nullptr);
}

py::object resolve_cuda_stream_handle(const py::object& cuda_current_stream_obj, int64_t device_index) {
  py::object stream_obj = cuda_current_stream_obj(py::int_(device_index));
  return stream_obj.attr("cuda_stream");
}

struct AtenLayerRefs {
  py::object self_attn;
  py::object mlp;
  torch::Tensor q_proj_weight;
  torch::Tensor q_proj_bias;
  bool q_proj_has_bias;
  torch::Tensor k_proj_weight;
  torch::Tensor k_proj_bias;
  bool k_proj_has_bias;
  torch::Tensor v_proj_weight;
  torch::Tensor v_proj_bias;
  bool v_proj_has_bias;
  torch::Tensor o_proj_weight;
  torch::Tensor o_proj_bias;
  bool o_proj_has_bias;
  torch::Tensor mlp_gate_proj_weight;
  torch::Tensor mlp_gate_proj_bias;
  bool mlp_gate_proj_has_bias;
  torch::Tensor mlp_up_proj_weight;
  torch::Tensor mlp_up_proj_bias;
  bool mlp_up_proj_has_bias;
  torch::Tensor mlp_down_proj_weight;
  torch::Tensor mlp_down_proj_bias;
  bool mlp_down_proj_has_bias;
  torch::Tensor qkv_fused_weight;
  torch::Tensor qkv_fused_bias;
  std::vector<int64_t> qkv_split_sizes;
  bool qkv_has_bias;
  bool qkv_has_pack;
  torch::Tensor gate_up_fused_weight;
  torch::Tensor gate_up_fused_bias;
  std::vector<int64_t> gate_up_split_sizes;
  bool gate_up_has_bias;
  bool gate_up_has_pack;
  torch::Tensor input_layernorm_weight;
  double input_layernorm_eps;
  torch::Tensor q_norm_weight;
  torch::Tensor k_norm_weight;
  double q_norm_eps;
  double k_norm_eps;
  double attn_scaling;
  int64_t head_dim;
  torch::Tensor post_attention_layernorm_weight;
  double post_attention_layernorm_eps;
};

struct AtenLayerCacheEntry {
  int scalar_type_key;
  int64_t layer_count;
  std::vector<AtenLayerRefs> refs;
};

struct VisionLayerRefs {
  py::object attn_module;
  torch::Tensor norm1_weight;
  torch::Tensor norm1_bias;
  bool norm1_has_bias;
  double norm1_eps;
  torch::Tensor norm2_weight;
  torch::Tensor norm2_bias;
  bool norm2_has_bias;
  double norm2_eps;
  torch::Tensor qkv_weight;
  torch::Tensor qkv_bias;
  bool qkv_has_bias;
  torch::Tensor proj_weight;
  torch::Tensor proj_bias;
  bool proj_has_bias;
  int64_t num_heads;
  int64_t head_dim;
  double attn_scaling;
  std::string attn_impl;
  torch::Tensor mlp_fc1_weight;
  torch::Tensor mlp_fc1_bias;
  bool mlp_fc1_has_bias;
  torch::Tensor mlp_fc2_weight;
  torch::Tensor mlp_fc2_bias;
  bool mlp_fc2_has_bias;
  std::string mlp_act;
};

struct VisionMergerRefs {
  torch::Tensor norm_weight;
  torch::Tensor norm_bias;
  bool norm_has_bias;
  double norm_eps;
  torch::Tensor fc1_weight;
  torch::Tensor fc1_bias;
  bool fc1_has_bias;
  torch::Tensor fc2_weight;
  torch::Tensor fc2_bias;
  bool fc2_has_bias;
  int64_t hidden_size;
  bool use_postshuffle_norm;
};

struct VisionRefs {
  int scalar_type_key;
  int64_t layer_count;
  torch::Tensor patch_proj_weight;
  torch::Tensor patch_proj_weight_flat;
  torch::Tensor patch_proj_bias;
  bool patch_proj_has_bias;
  int64_t patch_in_channels;
  int64_t patch_temporal_size;
  int64_t patch_size;
  int64_t patch_embed_dim;
  torch::Tensor pos_embed_weight;
  int64_t num_grid_per_side;
  int64_t spatial_merge_size;
  torch::Tensor rotary_inv_freq;
  std::vector<VisionLayerRefs> layers;
  VisionMergerRefs final_merger;
  std::vector<VisionMergerRefs> deepstack_mergers;
};

struct VisionSetupCacheEntry {
  uintptr_t vision_key;
  int64_t t;
  int64_t h;
  int64_t w;
  int64_t merge_size;
  int device_type_key;
  int16_t device_index;
  int scalar_type_key;
  torch::Tensor pos_embeds;
  torch::Tensor rotary_cos;
  torch::Tensor rotary_sin;
  std::vector<int64_t> chunk_lengths;
};

std::unordered_map<uintptr_t, AtenLayerCacheEntry>* aten_layer_cache_store() {
  static auto* cache = new std::unordered_map<uintptr_t, AtenLayerCacheEntry>();
  return cache;
}

std::unordered_map<uintptr_t, VisionRefs>* vision_refs_cache_store() {
  static auto* cache = new std::unordered_map<uintptr_t, VisionRefs>();
  return cache;
}

std::unordered_map<std::string, VisionSetupCacheEntry>* vision_setup_cache_store() {
  static auto* cache = new std::unordered_map<std::string, VisionSetupCacheEntry>();
  return cache;
}

int16_t normalize_device_index(const torch::Device& device) {
  return device.has_index() ? static_cast<int16_t>(device.index()) : static_cast<int16_t>(-1);
}

const char* dtype_cache_key(torch::ScalarType scalar_type) {
  switch (scalar_type) {
    case torch::kHalf:
      return "torch.float16";
    case torch::kFloat:
      return "torch.float32";
    default:
      return nullptr;
  }
}

bool resolve_cached_linear_pack(
    py::object owner_module,
    const char* cache_attr,
    torch::ScalarType scalar_type,
    torch::Tensor* fused_weight,
    torch::Tensor* fused_bias,
    std::vector<int64_t>* split_sizes,
    bool* has_bias) {
  try {
    if (fused_weight == nullptr || fused_bias == nullptr || split_sizes == nullptr || has_bias == nullptr) {
      return false;
    }
    const char* key_cstr = dtype_cache_key(scalar_type);
    if (key_cstr == nullptr || !py::hasattr(owner_module, cache_attr)) {
      return false;
    }
    py::object cache_obj = owner_module.attr(cache_attr);
    if (!py::isinstance<py::dict>(cache_obj)) {
      return false;
    }
    py::dict cache = cache_obj.cast<py::dict>();
    py::str key(key_cstr);
    if (!cache.contains(key)) {
      return false;
    }
    py::object pack_obj = cache[key];
    if (pack_obj.is_none() || !py::isinstance<py::tuple>(pack_obj)) {
      return false;
    }
    py::tuple pack = pack_obj.cast<py::tuple>();
    if (py::len(pack) != 3) {
      return false;
    }
    py::object split_sizes_obj = py::reinterpret_borrow<py::object>(pack[2]);
    if (!(py::isinstance<py::tuple>(split_sizes_obj) || py::isinstance<py::list>(split_sizes_obj))) {
      return false;
    }
    py::sequence split_sizes_seq = split_sizes_obj.cast<py::sequence>();
    split_sizes->clear();
    split_sizes->reserve(static_cast<size_t>(py::len(split_sizes_seq)));
    for (py::handle split_obj : split_sizes_seq) {
      split_sizes->emplace_back(py::cast<int64_t>(split_obj));
    }
    if (split_sizes->empty()) {
      return false;
    }
    *fused_weight = pack[0].cast<torch::Tensor>();
    py::object fused_bias_obj = py::reinterpret_borrow<py::object>(pack[1]);
    if (fused_bias_obj.is_none()) {
      *fused_bias = torch::Tensor();
      *has_bias = false;
    } else {
      *fused_bias = fused_bias_obj.cast<torch::Tensor>();
      *has_bias = true;
    }
    return true;
  } catch (py::error_already_set& e) {
    e.restore();
    PyErr_Clear();
    return false;
  } catch (...) {
    return false;
  }
}

bool linear_from_resolved_pack(
    const torch::Tensor& fused_weight,
    const torch::Tensor& fused_bias,
    bool has_bias,
    const std::vector<int64_t>& split_sizes,
    const torch::Tensor& input,
    std::vector<torch::Tensor>* outputs) {
  if (outputs == nullptr || !fused_weight.defined() || split_sizes.empty()) {
    return false;
  }
  auto fused_out = has_bias ? at::linear(input, fused_weight, fused_bias) : at::linear(input, fused_weight, std::nullopt);
  auto parts = fused_out.split_with_sizes(split_sizes, -1);
  outputs->assign(parts.begin(), parts.end());
  return true;
}

torch::Tensor linear_from_module(py::object linear_module, const torch::Tensor& input) {
  auto weight = linear_module.attr("weight").cast<torch::Tensor>();
  py::object bias_obj = linear_module.attr("bias");
  if (bias_obj.is_none()) {
    return at::linear(input, weight, std::nullopt);
  }
  return at::linear(input, weight, bias_obj.cast<torch::Tensor>());
}

torch::Tensor linear_from_weight_bias(
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    bool has_bias,
    const torch::Tensor& input) {
  return has_bias ? at::linear(input, weight, bias) : at::linear(input, weight, std::nullopt);
}

torch::Tensor layernorm_last_dim(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    bool has_bias,
    double eps) {
  std::optional<torch::Tensor> weight_opt = weight;
  std::optional<torch::Tensor> bias_opt = has_bias ? std::optional<torch::Tensor>(bias) : std::nullopt;
  return at::layer_norm(input, {input.size(-1)}, weight_opt, bias_opt, eps, true);
}

bool linear_from_cached_pack(
    py::object owner_module,
    const char* cache_attr,
    const torch::Tensor& input,
    std::vector<torch::Tensor>* outputs) {
  try {
    if (outputs == nullptr || !py::hasattr(owner_module, cache_attr)) {
      return false;
    }
    auto cache_obj = owner_module.attr(cache_attr);
    if (!py::isinstance<py::dict>(cache_obj)) {
      return false;
    }
    auto cache = cache_obj.cast<py::dict>();
    if (py::len(cache) == 0) {
      return false;
    }

    py::object pack_obj = py::none();
    const char* dtype_key_cstr = nullptr;
    switch (input.scalar_type()) {
      case torch::kHalf:
        dtype_key_cstr = "torch.float16";
        break;
      case torch::kFloat:
        dtype_key_cstr = "torch.float32";
        break;
      default:
        return false;
    }
    auto dtype_key = py::str(dtype_key_cstr);
    if (!cache.contains(dtype_key)) {
      return false;
    }
    pack_obj = cache[dtype_key];
    if (pack_obj.is_none() || !py::isinstance<py::tuple>(pack_obj)) {
      return false;
    }
    auto pack = pack_obj.cast<py::tuple>();
    if (py::len(pack) != 3) {
      return false;
    }

    auto fused_weight = pack[0].cast<torch::Tensor>();
    auto fused_bias_obj = py::reinterpret_borrow<py::object>(pack[1]);
    auto split_sizes_obj = py::reinterpret_borrow<py::object>(pack[2]);
    if (!(py::isinstance<py::tuple>(split_sizes_obj) || py::isinstance<py::list>(split_sizes_obj))) {
      return false;
    }
    auto split_sizes_seq = split_sizes_obj.cast<py::sequence>();
    std::vector<int64_t> split_sizes;
    split_sizes.reserve(static_cast<size_t>(py::len(split_sizes_seq)));
    for (py::handle split_obj : split_sizes_seq) {
      split_sizes.emplace_back(py::cast<int64_t>(split_obj));
    }
    if (split_sizes.empty()) {
      return false;
    }

    auto fused_out = fused_bias_obj.is_none()
                         ? at::linear(input, fused_weight, std::nullopt)
                         : at::linear(input, fused_weight, fused_bias_obj.cast<torch::Tensor>());
    auto parts = fused_out.split_with_sizes(split_sizes, -1);
    outputs->assign(parts.begin(), parts.end());
    return true;
  } catch (py::error_already_set& e) {
    e.restore();
    PyErr_Clear();
    return false;
  } catch (...) {
    return false;
  }
}

bool get_optional_bias(py::object module, const char* attr, torch::Tensor* bias, bool* has_bias) {
  py::object bias_obj = module.attr(attr);
  *has_bias = !bias_obj.is_none();
  if (*has_bias) {
    *bias = bias_obj.cast<torch::Tensor>();
  } else {
    *bias = torch::Tensor();
  }
  return true;
}

std::string module_class_name(py::object module) {
  try {
    return py::cast<std::string>(module.attr("__class__").attr("__name__"));
  } catch (...) {
    return "";
  }
}

std::string resolve_vision_mlp_act(py::object act_fn) {
  auto name = module_class_name(act_fn);
  if (name == "GELUTanh") {
    return "gelu_tanh";
  }
  if (name == "GELU" || name == "GELUActivation") {
    return "gelu";
  }
  return name;
}

torch::Tensor apply_vision_activation(const torch::Tensor& input, const std::string& act) {
  if (act == "gelu_tanh") {
    return at::gelu(input, "tanh");
  }
  if (act == "GELU" || act == "gelu" || act == "GELUActivation") {
    return at::gelu(input, "none");
  }
  TORCH_CHECK(false, "unsupported C++ vision activation: ", act);
}

std::string resolve_vision_attn_impl(py::object attn_module) {
  if (!py::hasattr(attn_module, "config")) {
    return "sdpa";
  }
  auto config = attn_module.attr("config");
  if (!py::hasattr(config, "_attn_implementation")) {
    return "sdpa";
  }
  auto impl_obj = config.attr("_attn_implementation");
  if (impl_obj.is_none()) {
    return "sdpa";
  }
  auto impl = py::cast<std::string>(impl_obj);
  return impl.empty() ? "sdpa" : impl;
}

VisionMergerRefs resolve_vision_merger_refs(py::object merger) {
  VisionMergerRefs refs{
      torch::Tensor(),
      torch::Tensor(),
      false,
      0.0,
      torch::Tensor(),
      torch::Tensor(),
      false,
      torch::Tensor(),
      torch::Tensor(),
      false,
      0,
      false,
  };
  auto norm = merger.attr("norm");
  refs.norm_weight = norm.attr("weight").cast<torch::Tensor>();
  get_optional_bias(norm, "bias", &refs.norm_bias, &refs.norm_has_bias);
  refs.norm_eps = py::float_(norm.attr("eps"));
  auto fc1 = merger.attr("linear_fc1");
  refs.fc1_weight = fc1.attr("weight").cast<torch::Tensor>();
  get_optional_bias(fc1, "bias", &refs.fc1_bias, &refs.fc1_has_bias);
  auto fc2 = merger.attr("linear_fc2");
  refs.fc2_weight = fc2.attr("weight").cast<torch::Tensor>();
  get_optional_bias(fc2, "bias", &refs.fc2_bias, &refs.fc2_has_bias);
  refs.hidden_size = py::int_(merger.attr("hidden_size"));
  refs.use_postshuffle_norm = py::bool_(merger.attr("use_postshuffle_norm"));
  return refs;
}

VisionRefs build_vision_refs(
    py::sequence blocks,
    py::sequence deepstack_mergers,
    py::object merger_obj,
    torch::ScalarType scalar_type) {
  VisionRefs out;
  out.scalar_type_key = static_cast<int>(scalar_type);
  out.layer_count = static_cast<int64_t>(py::len(blocks));
  out.layers.reserve(static_cast<size_t>(out.layer_count));
  for (int64_t layer_idx = 0; layer_idx < out.layer_count; ++layer_idx) {
    py::object block = py::reinterpret_borrow<py::object>(blocks[static_cast<py::ssize_t>(layer_idx)]);
    VisionLayerRefs refs{
        py::none(),
        torch::Tensor(),
        torch::Tensor(),
        false,
        0.0,
        torch::Tensor(),
        torch::Tensor(),
        false,
        0.0,
        torch::Tensor(),
        torch::Tensor(),
        false,
        torch::Tensor(),
        torch::Tensor(),
        false,
        0,
        0,
        0.0,
        "",
        torch::Tensor(),
        torch::Tensor(),
        false,
        torch::Tensor(),
        torch::Tensor(),
        false,
        "",
    };
    auto norm1 = block.attr("norm1");
    refs.norm1_weight = norm1.attr("weight").cast<torch::Tensor>();
    get_optional_bias(norm1, "bias", &refs.norm1_bias, &refs.norm1_has_bias);
    refs.norm1_eps = py::float_(norm1.attr("eps"));
    auto norm2 = block.attr("norm2");
    refs.norm2_weight = norm2.attr("weight").cast<torch::Tensor>();
    get_optional_bias(norm2, "bias", &refs.norm2_bias, &refs.norm2_has_bias);
    refs.norm2_eps = py::float_(norm2.attr("eps"));

    refs.attn_module = block.attr("attn");
    refs.attn_impl = resolve_vision_attn_impl(refs.attn_module);
    TORCH_CHECK(
        refs.attn_impl == "sdpa" || refs.attn_impl == "eager",
        "unsupported C++ vision attention implementation: ",
        refs.attn_impl);
    auto qkv = refs.attn_module.attr("qkv");
    refs.qkv_weight = qkv.attr("weight").cast<torch::Tensor>();
    get_optional_bias(qkv, "bias", &refs.qkv_bias, &refs.qkv_has_bias);
    auto proj = refs.attn_module.attr("proj");
    refs.proj_weight = proj.attr("weight").cast<torch::Tensor>();
    get_optional_bias(proj, "bias", &refs.proj_bias, &refs.proj_has_bias);
    refs.num_heads = py::int_(refs.attn_module.attr("num_heads"));
    refs.head_dim = py::int_(refs.attn_module.attr("head_dim"));
    refs.attn_scaling = py::float_(refs.attn_module.attr("scaling"));

    auto mlp = block.attr("mlp");
    auto fc1 = mlp.attr("linear_fc1");
    refs.mlp_fc1_weight = fc1.attr("weight").cast<torch::Tensor>();
    get_optional_bias(fc1, "bias", &refs.mlp_fc1_bias, &refs.mlp_fc1_has_bias);
    auto fc2 = mlp.attr("linear_fc2");
    refs.mlp_fc2_weight = fc2.attr("weight").cast<torch::Tensor>();
    get_optional_bias(fc2, "bias", &refs.mlp_fc2_bias, &refs.mlp_fc2_has_bias);
    refs.mlp_act = resolve_vision_mlp_act(mlp.attr("act_fn"));
    out.layers.emplace_back(std::move(refs));
  }
  out.final_merger = resolve_vision_merger_refs(merger_obj);
  int64_t deepstack_count = static_cast<int64_t>(py::len(deepstack_mergers));
  out.deepstack_mergers.reserve(static_cast<size_t>(deepstack_count));
  for (int64_t idx = 0; idx < deepstack_count; ++idx) {
    py::object merger = py::reinterpret_borrow<py::object>(deepstack_mergers[static_cast<py::ssize_t>(idx)]);
    out.deepstack_mergers.emplace_back(resolve_vision_merger_refs(merger));
  }
  return out;
}

VisionRefs build_vision_refs_from_model(py::object vision_model, torch::ScalarType scalar_type) {
  auto blocks = vision_model.attr("blocks").cast<py::sequence>();
  auto deepstack_mergers = vision_model.attr("deepstack_merger_list").cast<py::sequence>();
  auto merger_obj = vision_model.attr("merger");
  auto out = build_vision_refs(blocks, deepstack_mergers, merger_obj, scalar_type);

  auto patch_embed = vision_model.attr("patch_embed");
  auto patch_proj = patch_embed.attr("proj");
  out.patch_proj_weight = patch_proj.attr("weight").cast<torch::Tensor>();
  get_optional_bias(patch_proj, "bias", &out.patch_proj_bias, &out.patch_proj_has_bias);
  out.patch_in_channels = py::int_(patch_embed.attr("in_channels"));
  out.patch_temporal_size = py::int_(patch_embed.attr("temporal_patch_size"));
  out.patch_size = py::int_(patch_embed.attr("patch_size"));
  out.patch_embed_dim = py::int_(patch_embed.attr("embed_dim"));
  out.patch_proj_weight_flat = out.patch_proj_weight.view({out.patch_embed_dim, -1});

  auto pos_embed = vision_model.attr("pos_embed");
  out.pos_embed_weight = pos_embed.attr("weight").cast<torch::Tensor>();
  out.num_grid_per_side = py::int_(vision_model.attr("num_grid_per_side"));
  out.spatial_merge_size = py::int_(vision_model.attr("spatial_merge_size"));

  auto rotary_pos_emb = vision_model.attr("rotary_pos_emb");
  out.rotary_inv_freq = rotary_pos_emb.attr("inv_freq").cast<torch::Tensor>();
  return out;
}

torch::Tensor rmsnorm_last_dim(
    const torch::Tensor& hidden_states,
    const torch::Tensor& weight,
    double eps) {
  auto hidden_states_fp32 = hidden_states.to(torch::kFloat32);
  auto variance =
      at::mean(at::pow(hidden_states_fp32, 2.0), std::vector<int64_t>{-1}, true, std::nullopt);
  auto normalized = hidden_states_fp32 * at::rsqrt(variance + eps);
  return normalized.to(hidden_states.scalar_type()) * weight;
}

std::pair<torch::Tensor, torch::Tensor> add_rmsnorm_last_dim(
    const torch::Tensor& residual,
    const torch::Tensor& hidden_states,
    const torch::Tensor& weight,
    double eps) {
  auto summed = residual + hidden_states;
  auto normalized = rmsnorm_last_dim(summed, weight, eps);
  return std::make_pair(summed, normalized);
}

torch::Tensor rotate_half(const torch::Tensor& x) {
  auto half_dim = x.size(-1) / 2;
  auto x1 = x.slice(-1, 0, half_dim);
  auto x2 = x.slice(-1, half_dim);
  return at::cat({-x2, x1}, -1);
}

std::pair<torch::Tensor, torch::Tensor> apply_rotary_pos_emb_aten(
    const torch::Tensor& query_states,
    const torch::Tensor& key_states,
    const torch::Tensor& cos,
    const torch::Tensor& sin) {
  auto cos_broadcast = cos.unsqueeze(1);
  auto sin_broadcast = sin.unsqueeze(1);
  auto query_embed = (query_states * cos_broadcast) + (rotate_half(query_states) * sin_broadcast);
  auto key_embed = (key_states * cos_broadcast) + (rotate_half(key_states) * sin_broadcast);
  return std::make_pair(query_embed, key_embed);
}

py::object ensure_dynamic_cache(
    py::object past_key_values_obj,
    bool use_cache,
    bool is_tracing,
    py::object dynamic_cache_cls,
    py::object module_config_obj) {
  if (!use_cache || !past_key_values_obj.is_none() || is_tracing) {
    return past_key_values_obj;
  }
  TORCH_CHECK(!dynamic_cache_cls.is_none(), "DynamicCache class must be provided when use_cache=True");
  py::dict kwargs;
  kwargs["config"] = module_config_obj;
  return call_with_kwargs(dynamic_cache_cls, py::tuple(), kwargs);
}

torch::Tensor embed_tokens_cpp(py::object text_model_obj, py::object input_ids_obj) {
  TORCH_CHECK(!text_model_obj.is_none(), "text_model must be provided");
  TORCH_CHECK(!input_ids_obj.is_none(), "input_ids must be provided");
  auto input_ids = input_ids_obj.cast<torch::Tensor>();
  TORCH_CHECK(py::hasattr(text_model_obj, "embed_tokens"), "text_model must expose embed_tokens");
  py::object embed_tokens_obj = text_model_obj.attr("embed_tokens");
  TORCH_CHECK(py::hasattr(embed_tokens_obj, "weight"), "embed_tokens module must expose weight");
  auto weight = embed_tokens_obj.attr("weight").cast<torch::Tensor>();
  int64_t padding_idx = -1;
  if (py::hasattr(embed_tokens_obj, "padding_idx")) {
    py::object padding_idx_obj = embed_tokens_obj.attr("padding_idx");
    if (!padding_idx_obj.is_none()) {
      padding_idx = py::cast<int64_t>(padding_idx_obj);
    }
  }
  return at::embedding(weight, input_ids, padding_idx, false, false);
}

torch::Tensor build_cache_position(
    py::object past_key_values_obj,
    py::object cache_position_obj,
    const torch::Tensor& inputs_embeds) {
  if (!cache_position_obj.is_none()) {
    return cache_position_obj.cast<torch::Tensor>();
  }
  int64_t past_seen_tokens = 0;
  if (!past_key_values_obj.is_none()) {
    past_seen_tokens = py::int_(past_key_values_obj.attr("get_seq_length")());
  }
  return torch::arange(
      past_seen_tokens,
      past_seen_tokens + inputs_embeds.size(1),
      inputs_embeds.options().dtype(torch::kLong));
}

py::tuple normalize_position_ids(
    py::object position_ids_obj,
    const torch::Tensor& cache_position,
    const torch::Tensor& inputs_embeds) {
  torch::Tensor position_ids;
  if (position_ids_obj.is_none()) {
    position_ids = cache_position.view({1, 1, -1}).expand({3, inputs_embeds.size(0), -1});
  } else {
    position_ids = position_ids_obj.cast<torch::Tensor>();
    if (position_ids.dim() == 2) {
      position_ids = position_ids.unsqueeze(0).expand({3, position_ids.size(0), -1});
    }
  }

  torch::Tensor text_position_ids;
  if (position_ids.dim() == 3 && position_ids.size(0) == 4) {
    text_position_ids = position_ids.select(0, 0);
    position_ids = position_ids.slice(0, 1, 4);
  } else {
    text_position_ids = position_ids.select(0, 0);
  }
  return py::make_tuple(position_ids, text_position_ids);
}

bool can_skip_sdpa_causal_mask_fastpath(
    py::object module_config_obj,
    const torch::Tensor& inputs_embeds,
    py::object attention_mask_obj,
    const torch::Tensor& cache_position,
    py::object past_key_values_obj,
    bool is_tracing) {
  if (is_tracing) {
    return false;
  }
  if (module_config_obj.is_none() || !py::hasattr(module_config_obj, "_attn_implementation")) {
    return false;
  }
  py::object attn_impl_obj = module_config_obj.attr("_attn_implementation");
  if (attn_impl_obj.is_none()) {
    return false;
  }
  std::string attn_impl = py::cast<std::string>(attn_impl_obj);
  if (attn_impl != "sdpa") {
    return false;
  }

  const int64_t query_length = inputs_embeds.size(1);
  int64_t kv_length = query_length;
  int64_t kv_offset = 0;
  if (!past_key_values_obj.is_none() && py::hasattr(past_key_values_obj, "get_mask_sizes")) {
    auto sizes = past_key_values_obj.attr("get_mask_sizes")(cache_position, py::int_(0)).cast<py::tuple>();
    if (py::len(sizes) == 2) {
      kv_length = py::cast<int64_t>(sizes[0]);
      kv_offset = py::cast<int64_t>(sizes[1]);
    }
  }

  if (!(query_length == 1 || kv_length == query_length)) {
    return false;
  }
  if (kv_offset != 0) {
    return false;
  }

  if (attention_mask_obj.is_none()) {
    return true;
  }
  auto attention_mask = attention_mask_obj.cast<torch::Tensor>();
  if (attention_mask.dim() == 4) {
    return false;
  }
  if (attention_mask.dim() != 2) {
    return false;
  }
  auto bool_mask = attention_mask.to(inputs_embeds.device(), torch::kBool);
  return bool_mask.numel() == 0 || bool_mask.all().item<bool>();
}

py::object build_attention_mask(
    py::object create_causal_mask_obj,
    py::object module_config_obj,
    const torch::Tensor& inputs_embeds,
    py::object attention_mask_obj,
    const torch::Tensor& cache_position,
    py::object past_key_values_obj,
    const torch::Tensor& text_position_ids,
    bool is_tracing) {
  TORCH_CHECK(!create_causal_mask_obj.is_none(), "create_causal_mask callable must be provided");
  if (can_skip_sdpa_causal_mask_fastpath(
          module_config_obj,
          inputs_embeds,
          attention_mask_obj,
          cache_position,
          past_key_values_obj,
          is_tracing)) {
    return py::none();
  }
  py::dict kwargs;
  kwargs["config"] = module_config_obj;
  kwargs["input_embeds"] = inputs_embeds;
  kwargs["attention_mask"] = attention_mask_obj;
  kwargs["cache_position"] = cache_position;
  kwargs["past_key_values"] = past_key_values_obj;
  kwargs["position_ids"] = text_position_ids;
  return call_with_kwargs(create_causal_mask_obj, py::tuple(), kwargs);
}

std::pair<torch::Tensor, torch::Tensor> build_position_embeddings_cpp(
    py::object text_model_obj,
    const torch::Tensor& hidden_states,
    const torch::Tensor& position_ids,
    py::object prefill_rope_launcher_obj,
    py::object prefill_rope_function_obj,
    py::object prefill_rope_packed_metadata_obj,
    int64_t prefill_rope_block_d,
    py::object cuda_current_stream_obj) {
  TORCH_CHECK(!text_model_obj.is_none(), "text_model must be provided");
  py::object rotary_obj = text_model_obj.attr("rotary_emb");

  bool use_cached_prefill_rope =
      !prefill_rope_launcher_obj.is_none() && !prefill_rope_function_obj.is_none() &&
      !prefill_rope_packed_metadata_obj.is_none() && prefill_rope_block_d > 0 && !cuda_current_stream_obj.is_none();
  if (use_cached_prefill_rope) {
    TORCH_CHECK(py::hasattr(rotary_obj, "inv_freq"), "rotary_emb must expose inv_freq");
    auto inv_freq = rotary_obj.attr("inv_freq").cast<torch::Tensor>().to(torch::kFloat32);
    if (!inv_freq.is_contiguous()) {
      inv_freq = inv_freq.contiguous();
    }
    auto batch_size = position_ids.size(1);
    auto seq_len = position_ids.size(2);
    auto half_dim = inv_freq.size(0);
    auto head_dim = half_dim * 2;
    auto out_cos = torch::empty({batch_size, seq_len, head_dim}, hidden_states.options());
    auto out_sin = torch::empty_like(out_cos);
    py::object stream_handle = resolve_cuda_stream_handle(cuda_current_stream_obj, hidden_states.device().index());

    int64_t len_h = 0;
    int64_t len_w = 0;
    if (py::hasattr(rotary_obj, "mrope_section")) {
      auto mrope_section = rotary_obj.attr("mrope_section");
      if (!mrope_section.is_none()) {
        auto seq = mrope_section.cast<py::sequence>();
        if (py::len(seq) >= 3) {
          len_h = py::cast<int64_t>(seq[1]) * 3;
          len_w = py::cast<int64_t>(seq[2]) * 3;
        }
      }
    }
    len_h = std::max<int64_t>(0, std::min<int64_t>(len_h, half_dim));
    len_w = std::max<int64_t>(0, std::min<int64_t>(len_w, half_dim));
    double attention_scaling = py::float_(rotary_obj.attr("attention_scaling"));
    auto rope_pair = launch_prefill_rope_cached_triton(
        prefill_rope_launcher_obj,
        prefill_rope_function_obj,
        prefill_rope_packed_metadata_obj,
        cuda_current_stream_obj,
        prefill_rope_block_d,
        position_ids,
        inv_freq,
        out_cos,
        out_sin,
        seq_len,
        attention_scaling,
        len_h,
        len_w);
    return {
        rope_pair[0].cast<torch::Tensor>(),
        rope_pair[1].cast<torch::Tensor>(),
    };
  }

  auto rope_out = rotary_obj(hidden_states, position_ids).cast<py::tuple>();
  TORCH_CHECK(py::len(rope_out) == 2, "rotary_emb must return (cos, sin)");
  return {
      rope_out[0].cast<torch::Tensor>(),
      rope_out[1].cast<torch::Tensor>(),
  };
}

bool launch_prefill_kv_cache_write_cached_triton(
    py::object launcher_obj,
    py::object function_obj,
    py::object packed_metadata_obj,
    py::object stream_handle,
    int64_t block_t,
    int64_t block_d,
    const torch::Tensor& key_states,
    const torch::Tensor& value_states,
    const torch::Tensor& cached_keys,
    const torch::Tensor& cached_values,
    const torch::Tensor& cache_position) {
  if (launcher_obj.is_none() || function_obj.is_none() || packed_metadata_obj.is_none() ||
      stream_handle.is_none() || block_t <= 0 || block_d <= 0) {
    return false;
  }
  if (!key_states.defined() || !value_states.defined() || !cached_keys.defined() ||
      !cached_values.defined() || !cache_position.defined()) {
    return false;
  }
  if (!key_states.is_cuda() || !value_states.is_cuda() || !cached_keys.is_cuda() ||
      !cached_values.is_cuda() || !cache_position.is_cuda()) {
    return false;
  }
  if (key_states.scalar_type() != torch::kFloat16 || value_states.scalar_type() != torch::kFloat16 ||
      cached_keys.scalar_type() != torch::kFloat16 || cached_values.scalar_type() != torch::kFloat16) {
    return false;
  }
  if (cache_position.scalar_type() != torch::kInt64) {
    return false;
  }
  if (key_states.dim() != 4 || value_states.dim() != 4 || cached_keys.dim() != 4 ||
      cached_values.dim() != 4 || cache_position.dim() != 1) {
    return false;
  }
  if (key_states.sizes() != value_states.sizes()) {
    return false;
  }
  if (cached_keys.sizes() != cached_values.sizes()) {
    return false;
  }
  if (key_states.size(0) != cached_keys.size(0) ||
      key_states.size(1) != cached_keys.size(1) ||
      key_states.size(3) != cached_keys.size(3) ||
      key_states.size(2) != cache_position.numel() ||
      cached_keys.size(2) < key_states.size(2)) {
    return false;
  }

  int64_t seq_len = key_states.size(2);
  int64_t kv_heads = key_states.size(1);
  int64_t head_dim = key_states.size(3);
  int64_t grid_t = (seq_len + block_t - 1) / block_t;
  int64_t grid_bh = key_states.size(0) * kv_heads;
  py::object args_objs[] = {
      py::int_(grid_t),
      py::int_(grid_bh),
      py::int_(1),
      stream_handle,
      function_obj,
      packed_metadata_obj,
      py::none(),
      py::none(),
      py::none(),
      py::cast(key_states),
      py::cast(value_states),
      py::cast(cached_keys),
      py::cast(cached_values),
      py::cast(cache_position),
      py::int_(key_states.stride(0)),
      py::int_(key_states.stride(1)),
      py::int_(key_states.stride(2)),
      py::int_(key_states.stride(3)),
      py::int_(value_states.stride(0)),
      py::int_(value_states.stride(1)),
      py::int_(value_states.stride(2)),
      py::int_(value_states.stride(3)),
      py::int_(cached_keys.stride(0)),
      py::int_(cached_keys.stride(1)),
      py::int_(cached_keys.stride(2)),
      py::int_(cached_keys.stride(3)),
      py::int_(cached_values.stride(0)),
      py::int_(cached_values.stride(1)),
      py::int_(cached_values.stride(2)),
      py::int_(cached_values.stride(3)),
      py::int_(kv_heads),
      py::int_(seq_len),
      py::int_(head_dim),
      py::int_(block_t),
      py::int_(block_d),
  };
  PyObject* args[std::size(args_objs)];
  for (size_t i = 0; i < std::size(args_objs); ++i) {
    args[i] = args_objs[i].ptr();
  }
  auto _ = call_triton_launcher_compat(launcher_obj.ptr(), args, std::size(args), true);
  (void)_;
  prefill_kv_cache_write_counter().fetch_add(1, std::memory_order_relaxed);
  return true;
}

bool try_update_cache_layer_in_cpp(
    py::object cache_layer_obj,
    const torch::Tensor& key_states,
    const torch::Tensor& value_states,
    const torch::Tensor& cache_position,
    py::object kv_cache_write_launcher_obj,
    py::object kv_cache_write_function_obj,
    py::object kv_cache_write_packed_metadata_obj,
    py::object stream_handle,
    int64_t kv_cache_write_block_t,
    int64_t kv_cache_write_block_d,
    torch::Tensor* key_states_out,
    torch::Tensor* value_states_out) {
  if (cache_layer_obj.is_none()) {
    return false;
  }
  bool has_dynamic_slots =
      py::hasattr(cache_layer_obj, "is_initialized") &&
      py::hasattr(cache_layer_obj, "lazy_initialization") &&
      py::hasattr(cache_layer_obj, "keys") &&
      py::hasattr(cache_layer_obj, "values");
  if (!has_dynamic_slots) {
    return false;
  }

  bool is_initialized = py::bool_(cache_layer_obj.attr("is_initialized"));
  if (!is_initialized) {
    cache_layer_obj.attr("lazy_initialization")(key_states);
  }

  auto cached_keys = cache_layer_obj.attr("keys").cast<torch::Tensor>();
  auto cached_values = cache_layer_obj.attr("values").cast<torch::Tensor>();
  bool has_sliding_window =
      py::hasattr(cache_layer_obj, "sliding_window") && py::hasattr(cache_layer_obj, "cumulative_length");
  bool has_static_storage = py::hasattr(cache_layer_obj, "max_cache_len") &&
                            cached_keys.defined() &&
                            cached_values.defined() &&
                            cached_keys.dim() == 4 &&
                            cached_values.dim() == 4 &&
                            cached_keys.size(-2) >= key_states.size(-2) &&
                            cached_values.size(-2) >= value_states.size(-2) &&
                            cache_position.defined() &&
                            cache_position.dim() == 1 &&
                            cache_position.numel() == key_states.size(-2);

  if (has_static_storage) {
    auto index = cache_position;
    bool can_use_fused_cache_write = index.scalar_type() == torch::kInt64;
    if (index.scalar_type() != torch::kInt64) {
      index = index.to(torch::kInt64);
    }
    bool fused_cache_write = false;
    if (can_use_fused_cache_write) {
      fused_cache_write = launch_prefill_kv_cache_write_cached_triton(
          kv_cache_write_launcher_obj,
          kv_cache_write_function_obj,
          kv_cache_write_packed_metadata_obj,
          stream_handle,
          kv_cache_write_block_t,
          kv_cache_write_block_d,
          key_states,
          value_states,
          cached_keys,
          cached_values,
          index);
    }
    if (!fused_cache_write) {
      cached_keys.index_copy_(2, index, key_states);
      cached_values.index_copy_(2, index, value_states);
    }
    *key_states_out = cached_keys;
    *value_states_out = cached_values;
    cpp_cache_update_counter().fetch_add(1, std::memory_order_relaxed);
    return true;
  }

  if (!has_sliding_window) {
    auto full_keys = cached_keys.numel() == 0 ? key_states : torch::cat({cached_keys, key_states}, -2);
    auto full_values = cached_values.numel() == 0 ? value_states : torch::cat({cached_values, value_states}, -2);
    cache_layer_obj.attr("keys") = full_keys;
    cache_layer_obj.attr("values") = full_values;
    *key_states_out = full_keys;
    *value_states_out = full_values;
    cpp_cache_update_counter().fetch_add(1, std::memory_order_relaxed);
    return true;
  }

  int64_t sliding_window = py::cast<int64_t>(cache_layer_obj.attr("sliding_window"));
  if (sliding_window <= 0) {
    return false;
  }

  int64_t cumulative_length = py::cast<int64_t>(cache_layer_obj.attr("cumulative_length"));
  cumulative_length += key_states.size(-2);
  cache_layer_obj.attr("cumulative_length") = py::int_(cumulative_length);

  auto full_keys = cached_keys.numel() == 0 ? key_states : torch::cat({cached_keys, key_states}, -2);
  auto full_values = cached_values.numel() == 0 ? value_states : torch::cat({cached_values, value_states}, -2);
  int64_t keep_len = std::max<int64_t>(sliding_window - 1, 0);
  if (keep_len == 0) {
    cache_layer_obj.attr("keys") = full_keys.slice(-2, full_keys.size(-2), full_keys.size(-2));
    cache_layer_obj.attr("values") = full_values.slice(-2, full_values.size(-2), full_values.size(-2));
  } else {
    int64_t start = std::max<int64_t>(full_keys.size(-2) - keep_len, 0);
    cache_layer_obj.attr("keys") = full_keys.slice(-2, start, full_keys.size(-2));
    cache_layer_obj.attr("values") = full_values.slice(-2, start, full_values.size(-2));
  }
  *key_states_out = full_keys;
  *value_states_out = full_values;
  cpp_cache_update_counter().fetch_add(1, std::memory_order_relaxed);
  return true;
}

bool try_apply_deepstack_contiguous_add(
    torch::Tensor* hidden_states,
    const torch::Tensor& deepstack_embed_in,
    int64_t token_offset,
    int64_t token_count) {
  if (hidden_states == nullptr || !deepstack_embed_in.defined()) {
    return false;
  }
  auto& hs = *hidden_states;
  torch::Tensor deepstack_embed = deepstack_embed_in;
  if (token_offset < 0 || token_count <= 0) {
    return false;
  }
  if (hs.dim() != 3 || hs.size(0) != 1) {
    return false;
  }
  if (deepstack_embed.device() != hs.device() || deepstack_embed.scalar_type() != hs.scalar_type()) {
    deepstack_embed = deepstack_embed.to(hs.device(), hs.scalar_type());
  }
  if (deepstack_embed.dim() != 2 || deepstack_embed.size(0) != token_count) {
    return false;
  }
  if (deepstack_embed.size(1) != hs.size(2)) {
    return false;
  }
  if (token_offset + token_count > hs.size(1)) {
    return false;
  }
  hs.select(0, 0).narrow(0, token_offset, token_count).add_(deepstack_embed);
  return true;
}

void validate_deepstack_contiguous_mask(
    py::object visual_pos_masks_obj,
    int64_t token_offset,
    int64_t token_count) {
  TORCH_CHECK(!visual_pos_masks_obj.is_none(), "visual_pos_masks must be provided for deepstack fastpath");
  auto mask = visual_pos_masks_obj.cast<torch::Tensor>();
  TORCH_CHECK(mask.defined(), "visual_pos_masks object must be a tensor");
  TORCH_CHECK(mask.dim() == 2 && mask.size(0) == 1, "deepstack fastpath requires [1, seq_len] visual_pos_masks");
  TORCH_CHECK(mask.scalar_type() == torch::kBool, "deepstack fastpath requires bool visual_pos_masks");
  TORCH_CHECK(token_offset >= 0 && token_count > 0, "invalid deepstack token span");
  TORCH_CHECK(token_offset + token_count <= mask.size(1), "deepstack token span exceeds sequence length");

  auto true_count = at::sum(mask.to(torch::kLong)).item<int64_t>();
  TORCH_CHECK(true_count == token_count, "deepstack visual mask count does not match visual embeds");
  auto span = mask.slice(1, token_offset, token_offset + token_count);
  auto span_count = at::sum(span.to(torch::kLong)).item<int64_t>();
  TORCH_CHECK(span_count == token_count, "deepstack visual mask is not contiguous at recorded token span");
}

torch::Tensor mlp_forward_aten(py::object mlp_module, const torch::Tensor& hidden_states) {
  torch::Tensor gate;
  torch::Tensor up;
  std::vector<torch::Tensor> gate_up_parts;
  bool used_fused_gate_up =
      linear_from_cached_pack(mlp_module, "_aicas_gate_up_fused_linear_cache", hidden_states, &gate_up_parts) &&
      gate_up_parts.size() == 2;
  if (used_fused_gate_up) {
    gate = gate_up_parts[0];
    up = gate_up_parts[1];
  } else {
    gate = linear_from_module(mlp_module.attr("gate_proj"), hidden_states);
    up = linear_from_module(mlp_module.attr("up_proj"), hidden_states);
  }
  auto activated = at::silu(gate) * up;
  return linear_from_module(mlp_module.attr("down_proj"), activated);
}

torch::Tensor launch_swiglu_cached_triton(
    py::object launcher_obj,
    py::object function_obj,
    py::object packed_metadata_obj,
    py::object stream_handle,
    int64_t block_size,
    const torch::Tensor& gate,
    const torch::Tensor& up) {
  swiglu_counter().fetch_add(1, std::memory_order_relaxed);
  auto gate2d = gate.view({-1, gate.size(-1)});
  auto up2d = up.view({-1, up.size(-1)});
  auto out2d = torch::empty({gate2d.size(0), gate2d.size(1)}, gate2d.options());
  py::object gate2d_obj = py::cast(gate2d);
  py::object up2d_obj = py::cast(up2d);
  py::object out2d_obj = py::cast(out2d);
  py::object rows_obj = py::int_(gate2d.size(0));
  py::object grid_cols_obj = py::int_((gate2d.size(1) + block_size - 1) / block_size);
  py::object one_obj = py::int_(1);
  py::object none_obj = py::none();
  py::object args_objs[] = {
      rows_obj,
      grid_cols_obj,
      one_obj,
      stream_handle,
      function_obj,
      packed_metadata_obj,
      none_obj,
      none_obj,
      none_obj,
      gate2d_obj,
      up2d_obj,
      out2d_obj,
      py::int_(gate2d.stride(0)),
      py::int_(gate2d.stride(1)),
      py::int_(up2d.stride(0)),
      py::int_(up2d.stride(1)),
      py::int_(out2d.stride(0)),
      py::int_(out2d.stride(1)),
      py::int_(gate2d.size(1)),
      py::int_(block_size),
  };
  PyObject* args[std::size(args_objs)];
  for (size_t i = 0; i < std::size(args_objs); ++i) {
    args[i] = args_objs[i].ptr();
  }
  auto _ = call_triton_launcher_compat(launcher_obj.ptr(), args, std::size(args), true);
  (void)_;
  return out2d.view_as(gate);
}

torch::Tensor mlp_forward_aten_refs(
    const AtenLayerRefs& layer_refs,
    const torch::Tensor& hidden_states,
    bool use_cached_swiglu,
    py::object swiglu_launcher_obj,
    py::object swiglu_function_obj,
    py::object swiglu_packed_metadata_obj,
    py::object stream_handle,
    int64_t swiglu_block_size) {
  torch::Tensor gate;
  torch::Tensor up;
  std::vector<torch::Tensor> gate_up_parts;
  bool used_fused_gate_up = layer_refs.gate_up_has_pack &&
                            linear_from_resolved_pack(
                                layer_refs.gate_up_fused_weight,
                                layer_refs.gate_up_fused_bias,
                                layer_refs.gate_up_has_bias,
                                layer_refs.gate_up_split_sizes,
                                hidden_states,
                                &gate_up_parts) &&
                            gate_up_parts.size() == 2;
  if (used_fused_gate_up) {
    gate = gate_up_parts[0];
    up = gate_up_parts[1];
  } else {
    gate = linear_from_weight_bias(
        layer_refs.mlp_gate_proj_weight,
        layer_refs.mlp_gate_proj_bias,
        layer_refs.mlp_gate_proj_has_bias,
        hidden_states);
    up = linear_from_weight_bias(
        layer_refs.mlp_up_proj_weight,
        layer_refs.mlp_up_proj_bias,
        layer_refs.mlp_up_proj_has_bias,
        hidden_states);
  }
  torch::Tensor activated;
  if (use_cached_swiglu) {
    activated = launch_swiglu_cached_triton(
        swiglu_launcher_obj,
        swiglu_function_obj,
        swiglu_packed_metadata_obj,
        stream_handle,
        swiglu_block_size,
        gate,
        up);
  } else {
    activated = at::silu(gate) * up;
  }
  return linear_from_weight_bias(
      layer_refs.mlp_down_proj_weight,
      layer_refs.mlp_down_proj_bias,
      layer_refs.mlp_down_proj_has_bias,
      activated);
}

py::tuple launch_prefill_qk_rope_cached_triton(
    py::object launcher_obj,
    py::object function_obj,
    py::object packed_metadata_obj,
    py::object stream_handle,
    int64_t block_d,
    const torch::Tensor& query_states,
    const torch::Tensor& key_states,
    const torch::Tensor& q_norm_weight,
    const torch::Tensor& k_norm_weight,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    double q_norm_eps,
    double k_norm_eps) {
  prefill_qk_rope_counter().fetch_add(1, std::memory_order_relaxed);
  auto out_query = torch::empty_like(query_states);
  auto out_key = torch::empty_like(key_states);
  auto q_rows = query_states.size(0) * query_states.size(1) * query_states.size(2);
  auto total_rows = q_rows + key_states.size(0) * key_states.size(1) * key_states.size(2);
  py::object query_states_obj = py::cast(query_states);
  py::object key_states_obj = py::cast(key_states);
  py::object q_norm_weight_obj = py::cast(q_norm_weight);
  py::object k_norm_weight_obj = py::cast(k_norm_weight);
  py::object cos_obj = py::cast(cos);
  py::object sin_obj = py::cast(sin);
  py::object out_query_obj = py::cast(out_query);
  py::object out_key_obj = py::cast(out_key);
  py::object total_rows_obj = py::int_(total_rows);
  py::object one_obj = py::int_(1);
  py::object none_obj = py::none();
  py::object args_objs[] = {
      total_rows_obj,
      one_obj,
      one_obj,
      stream_handle,
      function_obj,
      packed_metadata_obj,
      none_obj,
      none_obj,
      none_obj,
      query_states_obj,
      key_states_obj,
      q_norm_weight_obj,
      k_norm_weight_obj,
      cos_obj,
      sin_obj,
      out_query_obj,
      out_key_obj,
      py::int_(query_states.stride(0)),
      py::int_(query_states.stride(1)),
      py::int_(query_states.stride(2)),
      py::int_(query_states.stride(3)),
      py::int_(key_states.stride(0)),
      py::int_(key_states.stride(1)),
      py::int_(key_states.stride(2)),
      py::int_(key_states.stride(3)),
      py::int_(cos.stride(0)),
      py::int_(cos.stride(1)),
      py::int_(cos.stride(2)),
      py::int_(sin.stride(0)),
      py::int_(sin.stride(1)),
      py::int_(sin.stride(2)),
      py::int_(out_query.stride(0)),
      py::int_(out_query.stride(1)),
      py::int_(out_query.stride(2)),
      py::int_(out_query.stride(3)),
      py::int_(out_key.stride(0)),
      py::int_(out_key.stride(1)),
      py::int_(out_key.stride(2)),
      py::int_(out_key.stride(3)),
      py::int_(query_states.size(1)),
      py::int_(key_states.size(1)),
      py::int_(query_states.size(2)),
      py::int_(query_states.size(3)),
      py::int_(cos.size(0)),
      py::float_(q_norm_eps),
      py::float_(k_norm_eps),
      py::int_(q_rows),
      py::int_(block_d),
  };
  PyObject* args[std::size(args_objs)];
  for (size_t i = 0; i < std::size(args_objs); ++i) {
    args[i] = args_objs[i].ptr();
  }
  auto _ = call_triton_launcher_compat(launcher_obj.ptr(), args, std::size(args), true);
  (void)_;
  return py::make_tuple(out_query, out_key);
}

py::tuple launch_add_rmsnorm_cached_triton(
    py::object launcher_obj,
    py::object function_obj,
    py::object packed_metadata_obj,
    py::object stream_handle,
    int64_t block_size,
    const torch::Tensor& x,
    const torch::Tensor& y,
    const torch::Tensor& weight,
    double eps) {
  add_rmsnorm_counter().fetch_add(1, std::memory_order_relaxed);
  auto x2d = x.contiguous().view({-1, x.size(-1)});
  auto y2d = y.contiguous().view({-1, y.size(-1)});
  auto sum_out = torch::empty_like(x2d);
  auto norm_out = torch::empty_like(x2d);
  py::object x2d_obj = py::cast(x2d);
  py::object y2d_obj = py::cast(y2d);
  py::object weight_obj = py::cast(weight);
  py::object sum_out_obj = py::cast(sum_out);
  py::object norm_out_obj = py::cast(norm_out);
  py::object rows_obj = py::int_(x2d.size(0));
  py::object one_obj = py::int_(1);
  py::object none_obj = py::none();
  py::object args_objs[] = {
      rows_obj,
      one_obj,
      one_obj,
      stream_handle,
      function_obj,
      packed_metadata_obj,
      none_obj,
      none_obj,
      none_obj,
      x2d_obj,
      y2d_obj,
      weight_obj,
      sum_out_obj,
      norm_out_obj,
      py::int_(x2d.stride(0)),
      py::int_(x2d.stride(1)),
      py::int_(y2d.stride(0)),
      py::int_(y2d.stride(1)),
      py::int_(sum_out.stride(0)),
      py::int_(sum_out.stride(1)),
      py::int_(norm_out.stride(0)),
      py::int_(norm_out.stride(1)),
      py::int_(x2d.size(1)),
      py::float_(eps),
      py::int_(block_size),
  };
  PyObject* args[std::size(args_objs)];
  for (size_t i = 0; i < std::size(args_objs); ++i) {
    args[i] = args_objs[i].ptr();
  }
  auto _ = call_triton_launcher_compat(launcher_obj.ptr(), args, std::size(args), true);
  (void)_;
  return py::make_tuple(sum_out.view_as(x), norm_out.view_as(x));
}

torch::Tensor launch_rmsnorm_cached_triton(
    py::object launcher_obj,
    py::object function_obj,
    py::object packed_metadata_obj,
    py::object stream_handle,
    int64_t block_size,
    const torch::Tensor& x,
    const torch::Tensor& weight,
    double eps) {
  rmsnorm_counter().fetch_add(1, std::memory_order_relaxed);
  auto x2d = x.contiguous().view({-1, x.size(-1)});
  auto y2d = torch::empty_like(x2d);
  py::object x2d_obj = py::cast(x2d);
  py::object weight_obj = py::cast(weight);
  py::object y2d_obj = py::cast(y2d);
  py::object rows_obj = py::int_(x2d.size(0));
  py::object one_obj = py::int_(1);
  py::object none_obj = py::none();
  py::object args_objs[] = {
      rows_obj,
      one_obj,
      one_obj,
      stream_handle,
      function_obj,
      packed_metadata_obj,
      none_obj,
      none_obj,
      none_obj,
      x2d_obj,
      weight_obj,
      y2d_obj,
      py::int_(x2d.stride(0)),
      py::int_(x2d.stride(1)),
      py::int_(y2d.stride(0)),
      py::int_(y2d.stride(1)),
      py::int_(x2d.size(1)),
      py::float_(eps),
      py::int_(block_size),
  };
  PyObject* args[std::size(args_objs)];
  for (size_t i = 0; i < std::size(args_objs); ++i) {
    args[i] = args_objs[i].ptr();
  }
  auto _ = call_triton_launcher_compat(launcher_obj.ptr(), args, std::size(args), true);
  (void)_;
  return y2d.view_as(x);
}

py::tuple launch_add_layernorm_cached_triton(
    py::object launcher_obj,
    py::object function_obj,
    py::object packed_metadata_obj,
    py::object cuda_current_stream_obj,
    int64_t block_size,
    const torch::Tensor& x,
    const torch::Tensor& y,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    double eps) {
  auto x2d = x.contiguous().view({-1, x.size(-1)});
  auto y2d = y.contiguous().view({-1, y.size(-1)});
  auto sum_out = torch::empty_like(x2d);
  auto norm_out = torch::empty_like(x2d);
  py::object stream_obj = cuda_current_stream_obj(py::int_(x.device().index()));
  py::object stream_handle = stream_obj.attr("cuda_stream");
  py::object x2d_obj = py::cast(x2d);
  py::object y2d_obj = py::cast(y2d);
  py::object weight_obj = py::cast(weight);
  py::object bias_obj = py::cast(bias);
  py::object sum_out_obj = py::cast(sum_out);
  py::object norm_out_obj = py::cast(norm_out);
  py::object rows_obj = py::int_(x2d.size(0));
  py::object one_obj = py::int_(1);
  py::object none_obj = py::none();
  py::object args_objs[] = {
      rows_obj,
      one_obj,
      one_obj,
      stream_handle,
      function_obj,
      packed_metadata_obj,
      none_obj,
      none_obj,
      none_obj,
      x2d_obj,
      y2d_obj,
      weight_obj,
      bias_obj,
      sum_out_obj,
      norm_out_obj,
      py::int_(x2d.stride(0)),
      py::int_(x2d.stride(1)),
      py::int_(y2d.stride(0)),
      py::int_(y2d.stride(1)),
      py::int_(sum_out.stride(0)),
      py::int_(sum_out.stride(1)),
      py::int_(norm_out.stride(0)),
      py::int_(norm_out.stride(1)),
      py::int_(x2d.size(1)),
      py::float_(eps),
      py::int_(block_size),
  };
  PyObject* args[std::size(args_objs)];
  for (size_t i = 0; i < std::size(args_objs); ++i) {
    args[i] = args_objs[i].ptr();
  }
  auto _ = call_triton_launcher_compat(launcher_obj.ptr(), args, std::size(args), true);
  (void)_;
  return py::make_tuple(sum_out.view_as(x), norm_out.view_as(x));
}

py::tuple launch_vision_rotary_cached_triton(
    py::object launcher_obj,
    py::object function_obj,
    py::object packed_metadata_obj,
    py::object cuda_current_stream_obj,
    int64_t block_d,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    py::object out_q_obj = py::none(),
    py::object out_k_obj = py::none()) {
  auto out_q = out_q_obj.is_none() ? torch::empty_like(q) : out_q_obj.cast<torch::Tensor>();
  auto out_k = out_k_obj.is_none() ? torch::empty_like(k) : out_k_obj.cast<torch::Tensor>();
  auto grid_x = q.size(0) * q.size(1);
  auto grid_y = (q.size(2) + block_d - 1) / block_d;
  py::object stream_obj = cuda_current_stream_obj(py::int_(q.device().index()));
  py::object stream_handle = stream_obj.attr("cuda_stream");
  py::object q_obj = py::cast(q);
  py::object k_obj = py::cast(k);
  py::object cos_obj = py::cast(cos);
  py::object sin_obj = py::cast(sin);
  py::object out_q_tensor_obj = py::cast(out_q);
  py::object out_k_tensor_obj = py::cast(out_k);
  py::object grid_x_obj = py::int_(grid_x);
  py::object grid_y_obj = py::int_(grid_y);
  py::object one_obj = py::int_(1);
  py::object none_obj = py::none();
  py::object args_objs[] = {
      grid_x_obj,
      grid_y_obj,
      one_obj,
      stream_handle,
      function_obj,
      packed_metadata_obj,
      none_obj,
      none_obj,
      none_obj,
      q_obj,
      k_obj,
      cos_obj,
      sin_obj,
      out_q_tensor_obj,
      out_k_tensor_obj,
      py::int_(q.stride(0)),
      py::int_(q.stride(1)),
      py::int_(q.stride(2)),
      py::int_(k.stride(0)),
      py::int_(k.stride(1)),
      py::int_(k.stride(2)),
      py::int_(cos.stride(0)),
      py::int_(cos.stride(1)),
      py::int_(sin.stride(0)),
      py::int_(sin.stride(1)),
      py::int_(out_q.stride(0)),
      py::int_(out_q.stride(1)),
      py::int_(out_q.stride(2)),
      py::int_(out_k.stride(0)),
      py::int_(out_k.stride(1)),
      py::int_(out_k.stride(2)),
      py::int_(q.size(1)),
      py::int_(q.size(2)),
      py::int_(q.size(2) / 2),
      py::int_(block_d),
  };
  PyObject* args[std::size(args_objs)];
  for (size_t i = 0; i < std::size(args_objs); ++i) {
    args[i] = args_objs[i].ptr();
  }
  auto _ = call_triton_launcher_compat(launcher_obj.ptr(), args, std::size(args), true);
  (void)_;
  return py::make_tuple(out_q, out_k);
}

py::tuple launch_prefill_rope_cached_triton(
    py::object launcher_obj,
    py::object function_obj,
    py::object packed_metadata_obj,
    py::object cuda_current_stream_obj,
    int64_t block_d,
    const torch::Tensor& position_ids,
    const torch::Tensor& inv_freq,
    const torch::Tensor& out_cos,
    const torch::Tensor& out_sin,
    int64_t seq_len,
    double attention_scaling,
    int64_t len_h,
    int64_t len_w) {
  prefill_rope_counter().fetch_add(1, std::memory_order_relaxed);
  auto half_dim = inv_freq.size(0);
  auto grid_x = out_cos.size(0) * out_cos.size(1);
  auto grid_y = (half_dim + block_d - 1) / block_d;
  py::object stream_obj = cuda_current_stream_obj(py::int_(position_ids.device().index()));
  py::object stream_handle = stream_obj.attr("cuda_stream");
  py::object position_ids_obj = py::cast(position_ids);
  py::object inv_freq_obj = py::cast(inv_freq);
  py::object out_cos_obj = py::cast(out_cos);
  py::object out_sin_obj = py::cast(out_sin);
  py::object grid_x_obj = py::int_(grid_x);
  py::object grid_y_obj = py::int_(grid_y);
  py::object one_obj = py::int_(1);
  py::object none_obj = py::none();
  py::object args_objs[] = {
      grid_x_obj,
      grid_y_obj,
      one_obj,
      stream_handle,
      function_obj,
      packed_metadata_obj,
      none_obj,
      none_obj,
      none_obj,
      position_ids_obj,
      inv_freq_obj,
      out_cos_obj,
      out_sin_obj,
      py::int_(position_ids.stride(0)),
      py::int_(position_ids.stride(1)),
      py::int_(position_ids.stride(2)),
      py::int_(out_cos.stride(0)),
      py::int_(out_cos.stride(1)),
      py::int_(out_cos.stride(2)),
      py::int_(out_sin.stride(0)),
      py::int_(out_sin.stride(1)),
      py::int_(out_sin.stride(2)),
      py::int_(seq_len),
      py::int_(half_dim),
      py::int_(len_h),
      py::int_(len_w),
      py::float_(attention_scaling),
      py::int_(block_d),
  };
  PyObject* args[std::size(args_objs)];
  for (size_t i = 0; i < std::size(args_objs); ++i) {
    args[i] = args_objs[i].ptr();
  }
  auto _ = call_triton_launcher_compat(launcher_obj.ptr(), args, std::size(args), true);
  (void)_;
  return py::make_tuple(out_cos, out_sin);
}

bool add_layernorm_launcher_ready(
    const py::object& launcher_obj,
    const py::object& function_obj,
    const py::object& packed_metadata_obj,
    const py::object& cuda_current_stream_obj,
    int64_t block_size) {
  return !launcher_obj.is_none() && !function_obj.is_none() && !packed_metadata_obj.is_none() &&
         !cuda_current_stream_obj.is_none() && block_size > 0;
}

std::pair<torch::Tensor, torch::Tensor> vision_add_layernorm_or_module(
    const torch::Tensor& x,
    const torch::Tensor& y,
    py::object norm_module,
    py::object add_layernorm_launcher_obj,
    py::object add_layernorm_function_obj,
    py::object add_layernorm_packed_metadata_obj,
    py::object cuda_current_stream_obj,
    int64_t add_layernorm_block_size) {
  if (add_layernorm_launcher_ready(
          add_layernorm_launcher_obj,
          add_layernorm_function_obj,
          add_layernorm_packed_metadata_obj,
          cuda_current_stream_obj,
          add_layernorm_block_size)) {
    auto fused_out = launch_add_layernorm_cached_triton(
        add_layernorm_launcher_obj,
        add_layernorm_function_obj,
        add_layernorm_packed_metadata_obj,
        cuda_current_stream_obj,
        add_layernorm_block_size,
        x,
        y,
        norm_module.attr("weight").cast<torch::Tensor>(),
        norm_module.attr("bias").cast<torch::Tensor>(),
        norm_module.attr("eps").cast<double>());
    return std::make_pair(
        fused_out[0].cast<torch::Tensor>(),
        fused_out[1].cast<torch::Tensor>());
  }

  auto residual = x + y;
  auto normed = norm_module(residual).cast<torch::Tensor>();
  return std::make_pair(residual, normed);
}

std::pair<torch::Tensor, torch::Tensor> vision_add_layernorm_or_refs(
    const torch::Tensor& x,
    const torch::Tensor& y,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    bool has_bias,
    double eps,
    py::object add_layernorm_launcher_obj,
    py::object add_layernorm_function_obj,
    py::object add_layernorm_packed_metadata_obj,
    py::object cuda_current_stream_obj,
    int64_t add_layernorm_block_size) {
  if (has_bias &&
      add_layernorm_launcher_ready(
          add_layernorm_launcher_obj,
          add_layernorm_function_obj,
          add_layernorm_packed_metadata_obj,
          cuda_current_stream_obj,
          add_layernorm_block_size)) {
    auto fused_out = launch_add_layernorm_cached_triton(
        add_layernorm_launcher_obj,
        add_layernorm_function_obj,
        add_layernorm_packed_metadata_obj,
        cuda_current_stream_obj,
        add_layernorm_block_size,
        x,
        y,
        weight,
        bias,
        eps);
    return std::make_pair(
        fused_out[0].cast<torch::Tensor>(),
        fused_out[1].cast<torch::Tensor>());
  }
  auto residual = x + y;
  auto normed = layernorm_last_dim(residual, weight, bias, has_bias, eps);
  return std::make_pair(residual, normed);
}

std::pair<torch::Tensor, torch::Tensor> apply_vision_rotary_pos_emb_aten(
    const torch::Tensor& query_states,
    const torch::Tensor& key_states,
    const torch::Tensor& cos,
    const torch::Tensor& sin) {
  auto q_dtype = query_states.scalar_type();
  auto k_dtype = key_states.scalar_type();
  auto q = query_states.to(torch::kFloat32);
  auto k = key_states.to(torch::kFloat32);
  auto cos_broadcast = cos.unsqueeze(1).to(torch::kFloat32);
  auto sin_broadcast = sin.unsqueeze(1).to(torch::kFloat32);
  auto query_embed = (q * cos_broadcast) + (rotate_half(q) * sin_broadcast);
  auto key_embed = (k * cos_broadcast) + (rotate_half(k) * sin_broadcast);
  return std::make_pair(query_embed.to(q_dtype), key_embed.to(k_dtype));
}

std::vector<int64_t> parse_vision_chunk_lengths(py::object chunk_lengths_obj, int64_t seq_len) {
  std::vector<int64_t> lengths;
  if (chunk_lengths_obj.is_none()) {
    lengths.push_back(seq_len);
    return lengths;
  }
  auto seq = chunk_lengths_obj.cast<py::sequence>();
  lengths.reserve(static_cast<size_t>(py::len(seq)));
  int64_t total = 0;
  for (py::handle item : seq) {
    int64_t value = py::cast<int64_t>(item);
    TORCH_CHECK(value > 0, "vision chunk length must be positive");
    lengths.push_back(value);
    total += value;
  }
  TORCH_CHECK(total == seq_len, "vision chunk lengths do not sum to sequence length");
  return lengths;
}

torch::Tensor vision_attention_forward_refs(
    const VisionLayerRefs& refs,
    const torch::Tensor& hidden_states,
    const std::vector<int64_t>& chunk_lengths,
    const torch::Tensor& rotary_cos,
    const torch::Tensor& rotary_sin,
    py::object vision_rotary_launcher_obj,
    py::object vision_rotary_function_obj,
    py::object vision_rotary_packed_metadata_obj,
    py::object cuda_current_stream_obj,
    int64_t vision_rotary_block_d) {
  TORCH_CHECK(
      refs.attn_impl == "sdpa" || refs.attn_impl == "eager",
      "C++ vision runtime only supports sdpa/eager attention, got ",
      refs.attn_impl);
  auto seq_len = hidden_states.size(0);
  auto qkv_out = linear_from_weight_bias(
      refs.qkv_weight,
      refs.qkv_bias,
      refs.qkv_has_bias,
      hidden_states);
  auto qkv = qkv_out.view({seq_len, 3, refs.num_heads, refs.head_dim});
  auto query_states = qkv.select(1, 0);
  auto key_states = qkv.select(1, 1);
  auto value_states = qkv.select(1, 2);

  bool use_cached_rotary =
      !vision_rotary_launcher_obj.is_none() && !vision_rotary_function_obj.is_none() &&
      !vision_rotary_packed_metadata_obj.is_none() && !cuda_current_stream_obj.is_none() &&
      vision_rotary_block_d > 0;
  if (use_cached_rotary) {
    auto rotary_out = launch_vision_rotary_cached_triton(
        vision_rotary_launcher_obj,
        vision_rotary_function_obj,
        vision_rotary_packed_metadata_obj,
        cuda_current_stream_obj,
        vision_rotary_block_d,
        query_states,
        key_states,
        rotary_cos,
        rotary_sin);
    query_states = rotary_out[0].cast<torch::Tensor>();
    key_states = rotary_out[1].cast<torch::Tensor>();
  } else {
    std::tie(query_states, key_states) =
        apply_vision_rotary_pos_emb_aten(query_states, key_states, rotary_cos, rotary_sin);
  }

  auto query_bhsd = query_states.transpose(0, 1).unsqueeze(0);
  auto key_bhsd = key_states.transpose(0, 1).unsqueeze(0);
  auto value_bhsd = value_states.transpose(0, 1).unsqueeze(0);
  std::vector<torch::Tensor> outputs;
  outputs.reserve(chunk_lengths.size());
  int64_t offset = 0;
  for (int64_t chunk_len : chunk_lengths) {
    auto q = query_bhsd.slice(2, offset, offset + chunk_len);
    auto k = key_bhsd.slice(2, offset, offset + chunk_len);
    auto v = value_bhsd.slice(2, offset, offset + chunk_len);
    torch::Tensor chunk_out;
    if (refs.attn_impl == "eager") {
      auto attn_weights = at::matmul(q, k.transpose(2, 3)) * refs.attn_scaling;
      attn_weights = at::softmax(attn_weights, -1, torch::kFloat32).to(q.scalar_type());
      chunk_out = at::matmul(attn_weights, v).transpose(1, 2).contiguous();
    } else {
      chunk_out = at::scaled_dot_product_attention(
                      q,
                      k,
                      v,
                      std::nullopt,
                      0.0,
                      false,
                      refs.attn_scaling,
                      false)
                      .transpose(1, 2)
                      .contiguous();
    }
    outputs.emplace_back(chunk_out);
    offset += chunk_len;
  }
  auto attn_output = at::cat(outputs, 1).reshape({seq_len, refs.num_heads * refs.head_dim}).contiguous();
  return linear_from_weight_bias(
      refs.proj_weight,
      refs.proj_bias,
      refs.proj_has_bias,
      attn_output);
}

torch::Tensor vision_mlp_forward_refs(
    const VisionLayerRefs& refs,
    const torch::Tensor& hidden_states) {
  auto fc1 = linear_from_weight_bias(
      refs.mlp_fc1_weight,
      refs.mlp_fc1_bias,
      refs.mlp_fc1_has_bias,
      hidden_states);
  auto activated = apply_vision_activation(fc1, refs.mlp_act);
  return linear_from_weight_bias(
      refs.mlp_fc2_weight,
      refs.mlp_fc2_bias,
      refs.mlp_fc2_has_bias,
      activated);
}

torch::Tensor vision_merger_forward_refs(
    const VisionMergerRefs& refs,
    const torch::Tensor& hidden_states,
    py::object cached_norm_obj = py::none()) {
  torch::Tensor x;
  if (!cached_norm_obj.is_none()) {
    x = cached_norm_obj.cast<torch::Tensor>();
  } else {
    x = refs.use_postshuffle_norm ? hidden_states.view({-1, refs.hidden_size}) : hidden_states;
    x = layernorm_last_dim(x, refs.norm_weight, refs.norm_bias, refs.norm_has_bias, refs.norm_eps);
  }
  x = x.view({-1, refs.hidden_size});
  x = linear_from_weight_bias(refs.fc1_weight, refs.fc1_bias, refs.fc1_has_bias, x);
  x = at::gelu(x, "none");
  return linear_from_weight_bias(refs.fc2_weight, refs.fc2_bias, refs.fc2_has_bias, x);
}

struct VisionGridMeta {
  int64_t t;
  int64_t h;
  int64_t w;
  int64_t merge_size;
  int64_t merged_h;
  int64_t merged_w;
  int64_t raw_tokens;
  int64_t merged_tokens;
};

std::string make_vision_setup_cache_key(
    uintptr_t vision_key,
    const VisionGridMeta& meta,
    const torch::Device& device,
    torch::ScalarType dtype) {
  std::string key = std::to_string(vision_key);
  key.push_back('|');
  key += std::to_string(meta.t);
  key.push_back('x');
  key += std::to_string(meta.h);
  key.push_back('x');
  key += std::to_string(meta.w);
  key.push_back('|');
  key += std::to_string(meta.merge_size);
  key.push_back('|');
  key += std::to_string(static_cast<int>(device.type()));
  key.push_back(':');
  key += std::to_string(normalize_device_index(device));
  key.push_back('|');
  key += std::to_string(static_cast<int>(dtype));
  return key;
}

VisionGridMeta parse_single_image_grid(const torch::Tensor& grid_thw, int64_t merge_size) {
  auto grid_cpu = grid_thw.device().is_cpu() ? grid_thw.to(torch::kLong) : grid_thw.to(torch::kCPU, torch::kLong);
  TORCH_CHECK(grid_cpu.numel() == 3, "AICAS C++ vision path expects one image grid_thw");
  auto flat = grid_cpu.reshape({-1});
  VisionGridMeta meta{
      flat[0].item<int64_t>(),
      flat[1].item<int64_t>(),
      flat[2].item<int64_t>(),
      merge_size,
      0,
      0,
      0,
      0,
  };
  meta.merged_h = meta.h / meta.merge_size;
  meta.merged_w = meta.w / meta.merge_size;
  meta.raw_tokens = meta.t * meta.h * meta.w;
  meta.merged_tokens = meta.raw_tokens / (meta.merge_size * meta.merge_size);
  return meta;
}

VisionGridMeta make_single_image_grid_meta(
    int64_t t,
    int64_t h,
    int64_t w,
    int64_t merge_size) {
  VisionGridMeta meta{
      t,
      h,
      w,
      merge_size,
      0,
      0,
      0,
      0,
  };
  meta.merged_h = meta.h / meta.merge_size;
  meta.merged_w = meta.w / meta.merge_size;
  meta.raw_tokens = meta.t * meta.h * meta.w;
  meta.merged_tokens = meta.raw_tokens / (meta.merge_size * meta.merge_size);
  return meta;
}

VisionGridMeta parse_or_make_single_image_grid(
    const torch::Tensor& grid_thw,
    int64_t merge_size,
    int64_t grid_t,
    int64_t grid_h,
    int64_t grid_w) {
  if (grid_t > 0 && grid_h > 0 && grid_w > 0) {
    return make_single_image_grid_meta(grid_t, grid_h, grid_w, merge_size);
  }
  return parse_single_image_grid(grid_thw, merge_size);
}

torch::Tensor vision_patch_embed_forward_refs(
    const VisionRefs& refs,
    const torch::Tensor& pixel_values) {
  const int64_t patch_flat_dim = refs.patch_in_channels * refs.patch_temporal_size * refs.patch_size * refs.patch_size;
  if (cpp_vision_patch_linear_enabled() &&
      pixel_values.dim() == 2 &&
      pixel_values.size(1) == patch_flat_dim &&
      refs.patch_proj_weight_flat.defined()) {
    auto x = pixel_values.to(refs.patch_proj_weight_flat.scalar_type());
    return linear_from_weight_bias(
        refs.patch_proj_weight_flat,
        refs.patch_proj_bias,
        refs.patch_proj_has_bias,
        x);
  }

  auto x = pixel_values.view(
      {-1, refs.patch_in_channels, refs.patch_temporal_size, refs.patch_size, refs.patch_size});
  x = x.to(refs.patch_proj_weight.scalar_type());
  x = at::conv3d(
      x,
      refs.patch_proj_weight,
      refs.patch_proj_has_bias ? std::optional<torch::Tensor>(refs.patch_proj_bias) : std::nullopt,
      {refs.patch_temporal_size, refs.patch_size, refs.patch_size},
      {0, 0, 0},
      {1, 1, 1},
      1);
  return x.view({-1, refs.patch_embed_dim});
}

torch::Tensor build_vision_pos_embeds_single_image(
    const VisionRefs& refs,
    const VisionGridMeta& meta,
    const torch::Device& device,
    torch::ScalarType dtype) {
  const float scale_h = meta.h > 1 ? static_cast<float>(refs.num_grid_per_side - 1) / static_cast<float>(meta.h - 1) : 0.0f;
  const float scale_w = meta.w > 1 ? static_cast<float>(refs.num_grid_per_side - 1) / static_cast<float>(meta.w - 1) : 0.0f;
  std::vector<int64_t> indices(4 * static_cast<size_t>(meta.raw_tokens));
  std::vector<float> weights(4 * static_cast<size_t>(meta.raw_tokens));

  int64_t out_idx = 0;
  for (int64_t tt = 0; tt < meta.t; ++tt) {
    (void)tt;
    for (int64_t mh = 0; mh < meta.merged_h; ++mh) {
      for (int64_t mw = 0; mw < meta.merged_w; ++mw) {
        for (int64_t ih = 0; ih < meta.merge_size; ++ih) {
          int64_t h_idx = mh * meta.merge_size + ih;
          float h_pos = static_cast<float>(h_idx) * scale_h;
          int64_t h_floor = static_cast<int64_t>(std::floor(h_pos));
          int64_t h_ceil = std::min(h_floor + 1, refs.num_grid_per_side - 1);
          float dh = h_pos - static_cast<float>(h_floor);
          for (int64_t iw = 0; iw < meta.merge_size; ++iw) {
            int64_t w_idx = mw * meta.merge_size + iw;
            float w_pos = static_cast<float>(w_idx) * scale_w;
            int64_t w_floor = static_cast<int64_t>(std::floor(w_pos));
            int64_t w_ceil = std::min(w_floor + 1, refs.num_grid_per_side - 1);
            float dw = w_pos - static_cast<float>(w_floor);
            int64_t idx00 = h_floor * refs.num_grid_per_side + w_floor;
            int64_t idx01 = h_floor * refs.num_grid_per_side + w_ceil;
            int64_t idx10 = h_ceil * refs.num_grid_per_side + w_floor;
            int64_t idx11 = h_ceil * refs.num_grid_per_side + w_ceil;
            indices[0 * meta.raw_tokens + out_idx] = idx00;
            indices[1 * meta.raw_tokens + out_idx] = idx01;
            indices[2 * meta.raw_tokens + out_idx] = idx10;
            indices[3 * meta.raw_tokens + out_idx] = idx11;
            weights[0 * meta.raw_tokens + out_idx] = (1.0f - dh) * (1.0f - dw);
            weights[1 * meta.raw_tokens + out_idx] = (1.0f - dh) * dw;
            weights[2 * meta.raw_tokens + out_idx] = dh * (1.0f - dw);
            weights[3 * meta.raw_tokens + out_idx] = dh * dw;
            ++out_idx;
          }
        }
      }
    }
  }
  auto idx_cpu = torch::from_blob(indices.data(), {4, meta.raw_tokens}, torch::TensorOptions().dtype(torch::kLong)).clone();
  auto weight_cpu = torch::from_blob(weights.data(), {4, meta.raw_tokens}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
  auto idx = idx_cpu.to(device, torch::kLong, false, false);
  auto weight = weight_cpu.to(device, dtype, false, false).unsqueeze(-1);
  auto pos_weight = refs.pos_embed_weight.to(device, dtype, false, false);
  auto pos_embeds = at::embedding(pos_weight, idx, -1, false, false) * weight;
  return pos_embeds.select(0, 0) + pos_embeds.select(0, 1) + pos_embeds.select(0, 2) + pos_embeds.select(0, 3);
}

std::pair<torch::Tensor, torch::Tensor> build_vision_rotary_cos_sin_single_image(
    const VisionRefs& refs,
    const VisionGridMeta& meta,
    const torch::Device& device) {
  auto inv_freq = refs.rotary_inv_freq.to(device, torch::kFloat32, false, false);
  auto rotary_dim = inv_freq.size(0);
  std::vector<int64_t> row_ids_vec(static_cast<size_t>(meta.raw_tokens));
  std::vector<int64_t> col_ids_vec(static_cast<size_t>(meta.raw_tokens));
  int64_t out_idx = 0;
  for (int64_t tt = 0; tt < meta.t; ++tt) {
    (void)tt;
    for (int64_t mh = 0; mh < meta.merged_h; ++mh) {
      for (int64_t mw = 0; mw < meta.merged_w; ++mw) {
        for (int64_t ih = 0; ih < meta.merge_size; ++ih) {
          int64_t h_idx = mh * meta.merge_size + ih;
          for (int64_t iw = 0; iw < meta.merge_size; ++iw) {
            int64_t w_idx = mw * meta.merge_size + iw;
            row_ids_vec[static_cast<size_t>(out_idx)] = h_idx;
            col_ids_vec[static_cast<size_t>(out_idx)] = w_idx;
            ++out_idx;
          }
        }
      }
    }
  }
  auto row_ids_cpu = torch::from_blob(row_ids_vec.data(), {meta.raw_tokens}, torch::TensorOptions().dtype(torch::kLong)).clone();
  auto col_ids_cpu = torch::from_blob(col_ids_vec.data(), {meta.raw_tokens}, torch::TensorOptions().dtype(torch::kLong)).clone();
  auto row_ids = row_ids_cpu.to(device, torch::kLong, false, false);
  auto col_ids = col_ids_cpu.to(device, torch::kLong, false, false);
  auto row_emb = row_ids.to(torch::kFloat32).unsqueeze(1) * inv_freq.unsqueeze(0);
  auto col_emb = col_ids.to(torch::kFloat32).unsqueeze(1) * inv_freq.unsqueeze(0);
  auto rotary_pos_emb = at::cat({row_emb, col_emb}, -1);
  auto emb = at::cat({rotary_pos_emb, rotary_pos_emb}, -1);
  return std::make_pair(emb.cos(), emb.sin());
}

std::vector<int64_t> build_vision_chunk_lengths_single_image(const VisionGridMeta& meta) {
  std::vector<int64_t> chunks;
  chunks.reserve(static_cast<size_t>(meta.t));
  int64_t chunk = meta.h * meta.w;
  for (int64_t idx = 0; idx < meta.t; ++idx) {
    chunks.emplace_back(chunk);
  }
  return chunks;
}

py::tuple vision_runtime_forward_refs(
    const VisionRefs& vision_refs,
    py::sequence deepstack_indexes,
    torch::Tensor hidden_states,
    py::object first_norm1_obj,
    const torch::Tensor& rotary_cos,
    const torch::Tensor& rotary_sin,
    const std::vector<int64_t>& chunk_lengths,
    py::object add_layernorm_launcher_obj,
    py::object add_layernorm_function_obj,
    py::object add_layernorm_packed_metadata_obj,
    py::object cuda_current_stream_obj,
    int64_t add_layernorm_block_size,
    bool interlayer_fuse_enabled,
    bool cached_norm_handoff_enabled,
    py::object vision_rotary_launcher_obj,
    py::object vision_rotary_function_obj,
    py::object vision_rotary_packed_metadata_obj,
    int64_t vision_rotary_block_d,
    bool final_merger_norm_is_training) {
  int64_t layer_count = static_cast<int64_t>(vision_refs.layers.size());
  int64_t deepstack_count = static_cast<int64_t>(py::len(deepstack_indexes));
  py::list deepstack_features;
  py::object cached_norm1 = first_norm1_obj;

  for (int64_t layer_idx = 0; layer_idx < layer_count; ++layer_idx) {
    const auto& refs = vision_refs.layers[static_cast<size_t>(layer_idx)];

    torch::Tensor normed;
    if (!cached_norm1.is_none()) {
      normed = cached_norm1.cast<torch::Tensor>();
      cached_norm1 = py::none();
    } else {
      normed = layernorm_last_dim(hidden_states, refs.norm1_weight, refs.norm1_bias, refs.norm1_has_bias, refs.norm1_eps);
    }

    auto attn_out = vision_attention_forward_refs(
        refs,
        normed,
        chunk_lengths,
        rotary_cos,
        rotary_sin,
        vision_rotary_launcher_obj,
        vision_rotary_function_obj,
        vision_rotary_packed_metadata_obj,
        cuda_current_stream_obj,
        vision_rotary_block_d);

    auto norm2_pair = vision_add_layernorm_or_refs(
        hidden_states,
        attn_out,
        refs.norm2_weight,
        refs.norm2_bias,
        refs.norm2_has_bias,
        refs.norm2_eps,
        add_layernorm_launcher_obj,
        add_layernorm_function_obj,
        add_layernorm_packed_metadata_obj,
        cuda_current_stream_obj,
        add_layernorm_block_size);
    auto residual = norm2_pair.first;
    auto norm2_in = norm2_pair.second;

    auto mlp_out = vision_mlp_forward_refs(refs, norm2_in);
    bool has_next_block = layer_idx + 1 < layer_count;
    if (cached_norm_handoff_enabled && interlayer_fuse_enabled && has_next_block) {
      const auto& next_refs = vision_refs.layers[static_cast<size_t>(layer_idx + 1)];
      auto next_norm_pair = vision_add_layernorm_or_refs(
          residual,
          mlp_out,
          next_refs.norm1_weight,
          next_refs.norm1_bias,
          next_refs.norm1_has_bias,
          next_refs.norm1_eps,
          add_layernorm_launcher_obj,
          add_layernorm_function_obj,
          add_layernorm_packed_metadata_obj,
          cuda_current_stream_obj,
          add_layernorm_block_size);
      hidden_states = next_norm_pair.first;
      cached_norm1 = py::cast(next_norm_pair.second);
    } else {
      if (cached_norm_handoff_enabled && !has_next_block && !final_merger_norm_is_training) {
        const auto& merger_refs = vision_refs.final_merger;
        auto merger_norm_pair = vision_add_layernorm_or_refs(
            residual,
            mlp_out,
            merger_refs.norm_weight,
            merger_refs.norm_bias,
            merger_refs.norm_has_bias,
            merger_refs.norm_eps,
            add_layernorm_launcher_obj,
            add_layernorm_function_obj,
            add_layernorm_packed_metadata_obj,
            cuda_current_stream_obj,
            add_layernorm_block_size);
        hidden_states = merger_norm_pair.first;
        cached_norm1 = py::cast(merger_norm_pair.second);
      } else {
        hidden_states = residual + mlp_out;
      }
    }

    for (int64_t ds_idx = 0; ds_idx < deepstack_count; ++ds_idx) {
      int64_t deepstack_layer = py::reinterpret_borrow<py::object>(
          deepstack_indexes[static_cast<py::ssize_t>(ds_idx)]).cast<int64_t>();
      if (deepstack_layer == layer_idx) {
        TORCH_CHECK(
            ds_idx < static_cast<int64_t>(vision_refs.deepstack_mergers.size()),
            "internal error: missing C++ deepstack merger refs");
        deepstack_features.append(vision_merger_forward_refs(
            vision_refs.deepstack_mergers[static_cast<size_t>(ds_idx)],
            hidden_states));
      }
    }
  }

  auto image_embeds = vision_merger_forward_refs(vision_refs.final_merger, hidden_states, cached_norm1);
  return py::make_tuple(image_embeds, deepstack_features);
}

py::tuple vision_runtime_forward(
    py::object blocks_obj,
    py::object deepstack_indexes_obj,
    py::object deepstack_mergers_obj,
    py::object merger_obj,
    torch::Tensor hidden_states,
    py::object first_norm1_obj,
    const torch::Tensor& cu_seqlens,
    const torch::Tensor& rotary_cos,
    const torch::Tensor& rotary_sin,
    py::object chunk_lengths_obj,
    py::dict extra_kwargs,
    py::object add_layernorm_launcher_obj,
    py::object add_layernorm_function_obj,
    py::object add_layernorm_packed_metadata_obj,
    py::object cuda_current_stream_obj,
    int64_t add_layernorm_block_size,
    bool interlayer_fuse_enabled,
    bool cached_norm_handoff_enabled,
    py::object vision_rotary_launcher_obj,
    py::object vision_rotary_function_obj,
    py::object vision_rotary_packed_metadata_obj,
    int64_t vision_rotary_block_d) {
  vision_runtime_forward_counter().fetch_add(1, std::memory_order_relaxed);

  auto blocks = blocks_obj.cast<py::sequence>();
  auto deepstack_indexes = deepstack_indexes_obj.cast<py::sequence>();
  auto deepstack_mergers = deepstack_mergers_obj.cast<py::sequence>();
  int64_t layer_count = static_cast<int64_t>(py::len(blocks));
  auto chunk_lengths = parse_vision_chunk_lengths(chunk_lengths_obj, hidden_states.size(0));
  (void)cu_seqlens;
  (void)extra_kwargs;

  uintptr_t blocks_key = reinterpret_cast<uintptr_t>(blocks_obj.ptr());
  int scalar_type_key = static_cast<int>(hidden_states.scalar_type());
  auto* refs_cache = vision_refs_cache_store();
  auto refs_it = refs_cache->find(blocks_key);
  if (refs_it == refs_cache->end() ||
      refs_it->second.scalar_type_key != scalar_type_key ||
      refs_it->second.layer_count != layer_count) {
    (*refs_cache)[blocks_key] = build_vision_refs(blocks, deepstack_mergers, merger_obj, hidden_states.scalar_type());
    refs_it = refs_cache->find(blocks_key);
  }
  const VisionRefs& vision_refs = refs_it->second;
  TORCH_CHECK(
      static_cast<int64_t>(vision_refs.layers.size()) == layer_count,
      "internal error: stale C++ vision refs layer count");

  return vision_runtime_forward_refs(
      vision_refs,
      deepstack_indexes,
      hidden_states,
      first_norm1_obj,
      rotary_cos,
      rotary_sin,
      chunk_lengths,
      add_layernorm_launcher_obj,
      add_layernorm_function_obj,
      add_layernorm_packed_metadata_obj,
      cuda_current_stream_obj,
      add_layernorm_block_size,
      interlayer_fuse_enabled,
      cached_norm_handoff_enabled,
      vision_rotary_launcher_obj,
      vision_rotary_function_obj,
      vision_rotary_packed_metadata_obj,
      vision_rotary_block_d,
      py::bool_(merger_obj.attr("training")));
}

py::tuple vision_get_image_features_aten(
    py::object vision_model_obj,
    const torch::Tensor& pixel_values,
    const torch::Tensor& image_grid_thw,
    py::object add_layernorm_launcher_obj,
    py::object add_layernorm_function_obj,
    py::object add_layernorm_packed_metadata_obj,
    py::object cuda_current_stream_obj,
    int64_t add_layernorm_block_size,
    bool interlayer_fuse_enabled,
    bool cached_norm_handoff_enabled,
    py::object vision_rotary_launcher_obj,
    py::object vision_rotary_function_obj,
    py::object vision_rotary_packed_metadata_obj,
    int64_t vision_rotary_block_d,
    int64_t grid_t,
    int64_t grid_h,
    int64_t grid_w) {
  vision_get_image_features_counter().fetch_add(1, std::memory_order_relaxed);

  auto blocks = vision_model_obj.attr("blocks").cast<py::sequence>();
  auto deepstack_indexes = vision_model_obj.attr("deepstack_visual_indexes").cast<py::sequence>();
  int64_t layer_count = static_cast<int64_t>(py::len(blocks));
  auto meta_probe_merge_size = py::int_(vision_model_obj.attr("spatial_merge_size"));
  auto grid_meta = parse_or_make_single_image_grid(image_grid_thw, meta_probe_merge_size, grid_t, grid_h, grid_w);

  uintptr_t vision_key = reinterpret_cast<uintptr_t>(vision_model_obj.ptr());
  int scalar_type_key = static_cast<int>(pixel_values.scalar_type());
  auto* refs_cache = vision_refs_cache_store();
  auto refs_it = refs_cache->find(vision_key);
  if (refs_it == refs_cache->end() ||
      refs_it->second.scalar_type_key != scalar_type_key ||
      refs_it->second.layer_count != layer_count) {
    (*refs_cache)[vision_key] = build_vision_refs_from_model(vision_model_obj, pixel_values.scalar_type());
    refs_it = refs_cache->find(vision_key);
  }
  const VisionRefs& vision_refs = refs_it->second;

  auto hidden_states = vision_patch_embed_forward_refs(vision_refs, pixel_values);
  auto setup_key = make_vision_setup_cache_key(
      vision_key,
      grid_meta,
      hidden_states.device(),
      hidden_states.scalar_type());
  bool setup_cache_enabled = cpp_vision_setup_cache_enabled();
  auto* setup_cache = setup_cache_enabled ? vision_setup_cache_store() : nullptr;
  auto setup_it = setup_cache_enabled ? setup_cache->find(setup_key) : vision_setup_cache_store()->end();
  bool setup_cache_hit =
      setup_cache_enabled &&
      setup_it != setup_cache->end() &&
      setup_it->second.pos_embeds.defined() &&
      setup_it->second.rotary_cos.defined() &&
      setup_it->second.rotary_sin.defined() &&
      setup_it->second.t == grid_meta.t &&
      setup_it->second.h == grid_meta.h &&
      setup_it->second.w == grid_meta.w &&
      setup_it->second.merge_size == grid_meta.merge_size &&
      setup_it->second.device_type_key == static_cast<int>(hidden_states.device().type()) &&
      setup_it->second.device_index == normalize_device_index(hidden_states.device()) &&
      setup_it->second.scalar_type_key == static_cast<int>(hidden_states.scalar_type()) &&
      setup_it->second.pos_embeds.device() == hidden_states.device() &&
      setup_it->second.pos_embeds.scalar_type() == hidden_states.scalar_type();

  torch::Tensor pos_embeds;
  torch::Tensor rotary_cos;
  torch::Tensor rotary_sin;
  std::vector<int64_t> chunk_lengths;
  if (setup_cache_hit) {
    vision_setup_cache_hit_counter().fetch_add(1, std::memory_order_relaxed);
    const auto& setup_entry = setup_it->second;
    pos_embeds = setup_entry.pos_embeds;
    rotary_cos = setup_entry.rotary_cos;
    rotary_sin = setup_entry.rotary_sin;
    chunk_lengths = setup_entry.chunk_lengths;
  } else {
    if (setup_cache_enabled) {
      vision_setup_cache_miss_counter().fetch_add(1, std::memory_order_relaxed);
    }
    pos_embeds = build_vision_pos_embeds_single_image(
        vision_refs,
        grid_meta,
        hidden_states.device(),
        hidden_states.scalar_type());
    auto rotary_pair = build_vision_rotary_cos_sin_single_image(
        vision_refs,
        grid_meta,
        hidden_states.device());
    rotary_cos = rotary_pair.first;
    rotary_sin = rotary_pair.second;
    chunk_lengths = build_vision_chunk_lengths_single_image(grid_meta);
    if (setup_cache_enabled) {
      (*setup_cache)[setup_key] = VisionSetupCacheEntry{
          vision_key,
          grid_meta.t,
          grid_meta.h,
          grid_meta.w,
          grid_meta.merge_size,
          static_cast<int>(hidden_states.device().type()),
          normalize_device_index(hidden_states.device()),
          static_cast<int>(hidden_states.scalar_type()),
          pos_embeds,
          rotary_cos,
          rotary_sin,
          chunk_lengths,
      };
    }
  }

  py::object first_norm1 = py::none();
  if (cached_norm_handoff_enabled && layer_count > 0) {
    const auto& first_refs = vision_refs.layers[0];
    auto first_pair = vision_add_layernorm_or_refs(
        hidden_states,
        pos_embeds,
        first_refs.norm1_weight,
        first_refs.norm1_bias,
        first_refs.norm1_has_bias,
        first_refs.norm1_eps,
        add_layernorm_launcher_obj,
        add_layernorm_function_obj,
        add_layernorm_packed_metadata_obj,
        cuda_current_stream_obj,
        add_layernorm_block_size);
    hidden_states = first_pair.first;
    first_norm1 = py::cast(first_pair.second);
  } else {
    hidden_states = hidden_states + pos_embeds;
  }

  auto runtime_out = vision_runtime_forward_refs(
      vision_refs,
      deepstack_indexes,
      hidden_states,
      first_norm1,
      rotary_cos,
      rotary_sin,
      chunk_lengths,
      add_layernorm_launcher_obj,
      add_layernorm_function_obj,
      add_layernorm_packed_metadata_obj,
      cuda_current_stream_obj,
      add_layernorm_block_size,
      interlayer_fuse_enabled,
      cached_norm_handoff_enabled,
      vision_rotary_launcher_obj,
      vision_rotary_function_obj,
      vision_rotary_packed_metadata_obj,
      vision_rotary_block_d,
      false);
  auto image_embeds = runtime_out[0].cast<torch::Tensor>();
  auto deepstack_features = runtime_out[1].cast<py::list>();
  py::list image_embed_splits;
  image_embed_splits.append(image_embeds);
  return py::make_tuple(image_embed_splits, deepstack_features);
}

void validate_prefill_text_runtime_boundary(
    const torch::Tensor& hidden_states,
    const torch::Tensor& text_position_ids,
    const torch::Tensor& cache_position,
    const torch::Tensor& rotary_cos,
    const torch::Tensor& rotary_sin,
    py::object attention_mask_obj,
    py::object visual_pos_masks_obj,
    int64_t deepstack_count,
    int64_t layer_count) {
  TORCH_CHECK(hidden_states.defined(), "hidden_states must be defined");
  TORCH_CHECK(text_position_ids.defined(), "text_position_ids must be defined");
  TORCH_CHECK(cache_position.defined(), "cache_position must be defined");
  TORCH_CHECK(rotary_cos.defined(), "rotary_cos must be defined");
  TORCH_CHECK(rotary_sin.defined(), "rotary_sin must be defined");
  TORCH_CHECK(layer_count >= 0, "layer_count must be non-negative");
  TORCH_CHECK(deepstack_count >= 0, "deepstack_count must be non-negative");
  TORCH_CHECK(hidden_states.dim() == 3, "hidden_states must be rank-3");
  TORCH_CHECK(text_position_ids.dim() == 2, "text_position_ids must be rank-2");
  TORCH_CHECK(cache_position.dim() == 1, "cache_position must be rank-1");
  TORCH_CHECK(rotary_cos.sizes() == rotary_sin.sizes(), "rotary cos/sin shape mismatch");
  TORCH_CHECK(
      hidden_states.size(0) == text_position_ids.size(0),
      "batch mismatch between hidden_states and text_position_ids");
  TORCH_CHECK(
      hidden_states.size(1) == text_position_ids.size(1),
      "seq mismatch between hidden_states and text_position_ids");
  TORCH_CHECK(
      hidden_states.size(1) == cache_position.size(0),
      "seq mismatch between hidden_states and cache_position");

  if (!attention_mask_obj.is_none()) {
    auto attention_mask = attention_mask_obj.cast<torch::Tensor>();
    TORCH_CHECK(attention_mask.defined(), "attention_mask object must be a tensor when provided");
  }
  if (!visual_pos_masks_obj.is_none()) {
    auto visual_pos_masks = visual_pos_masks_obj.cast<torch::Tensor>();
    TORCH_CHECK(visual_pos_masks.defined(), "visual_pos_masks object must be a tensor when provided");
  }
}

torch::Tensor prefill_text_runtime_run_aten_ops(
    py::iterable layers,
    torch::Tensor hidden_states,
    const torch::Tensor& text_position_ids,
    const torch::Tensor& cache_position,
    const torch::Tensor& rotary_cos,
    const torch::Tensor& rotary_sin,
    py::object attention_mask_obj,
    py::object past_key_values_obj,
    py::object visual_pos_masks_obj,
    py::object deepstack_visual_embeds_obj,
    py::object text_model_obj,
    py::object fused_add_rmsnorm_obj,
    py::dict extra_kwargs,
    py::object cuda_current_stream_obj,
    py::object prefill_qk_rope_launcher_obj,
    py::object prefill_qk_rope_function_obj,
    py::object prefill_qk_rope_packed_metadata_obj,
    int64_t prefill_qk_rope_block_d,
    py::object add_rmsnorm_launcher_obj,
    py::object add_rmsnorm_function_obj,
    py::object add_rmsnorm_packed_metadata_obj,
    int64_t add_rmsnorm_block_size,
    py::object rmsnorm_launcher_obj,
    py::object rmsnorm_function_obj,
    py::object rmsnorm_packed_metadata_obj,
    int64_t rmsnorm_block_size,
    py::object swiglu_launcher_obj,
    py::object swiglu_function_obj,
    py::object swiglu_packed_metadata_obj,
    int64_t swiglu_block_size,
    py::object kv_cache_write_launcher_obj,
    py::object kv_cache_write_function_obj,
    py::object kv_cache_write_packed_metadata_obj,
    int64_t kv_cache_write_block_t,
    int64_t kv_cache_write_block_d,
    py::object final_norm_weight_obj,
    double final_norm_eps) {
  int64_t layer_count = py::len(layers);
  int64_t deepstack_count = deepstack_visual_embeds_obj.is_none() ? 0 : py::len(deepstack_visual_embeds_obj);
  if (cpp_runtime_validate_enabled()) {
    validate_prefill_text_runtime_boundary(
        hidden_states,
        text_position_ids,
        cache_position,
        rotary_cos,
        rotary_sin,
        attention_mask_obj,
        visual_pos_masks_obj,
        deepstack_count,
        layer_count);
  }
  TORCH_CHECK(!fused_add_rmsnorm_obj.is_none(), "fused_add_rmsnorm callable must be provided");
  TORCH_CHECK(!final_norm_weight_obj.is_none(), "final_norm_weight must be provided");
  bool use_cached_prefill_qk_rope =
      !prefill_qk_rope_launcher_obj.is_none() && !prefill_qk_rope_function_obj.is_none() &&
      !prefill_qk_rope_packed_metadata_obj.is_none() && prefill_qk_rope_block_d > 0 &&
      !cuda_current_stream_obj.is_none();
  bool use_cached_add_rmsnorm =
      !add_rmsnorm_launcher_obj.is_none() && !add_rmsnorm_function_obj.is_none() &&
      !add_rmsnorm_packed_metadata_obj.is_none() && add_rmsnorm_block_size > 0 && !cuda_current_stream_obj.is_none();
  bool use_cached_rmsnorm =
      !rmsnorm_launcher_obj.is_none() && !rmsnorm_function_obj.is_none() && !rmsnorm_packed_metadata_obj.is_none() &&
      rmsnorm_block_size > 0 && !cuda_current_stream_obj.is_none();
  bool use_cached_swiglu =
      !swiglu_launcher_obj.is_none() && !swiglu_function_obj.is_none() &&
      !swiglu_packed_metadata_obj.is_none() && swiglu_block_size > 0 &&
      !cuda_current_stream_obj.is_none();
  bool use_cached_kv_cache_write =
      !kv_cache_write_launcher_obj.is_none() && !kv_cache_write_function_obj.is_none() &&
      !kv_cache_write_packed_metadata_obj.is_none() && kv_cache_write_block_t > 0 &&
      kv_cache_write_block_d > 0 && !cuda_current_stream_obj.is_none();
  bool use_interlayer_add_rmsnorm_handoff =
      cpp_text_interlayer_add_rmsnorm_enabled();
  bool use_final_add_rmsnorm = cpp_text_final_add_rmsnorm_enabled();
  py::object stream_handle = py::none();
  if (use_cached_prefill_qk_rope || use_cached_add_rmsnorm || use_cached_rmsnorm ||
      use_cached_swiglu || use_cached_kv_cache_write) {
    stream_handle = resolve_cuda_stream_handle(cuda_current_stream_obj, hidden_states.device().index());
  }

  std::optional<torch::Tensor> attention_mask = std::nullopt;
  if (!attention_mask_obj.is_none()) {
    attention_mask = attention_mask_obj.cast<torch::Tensor>();
  }
  uintptr_t layers_key = reinterpret_cast<uintptr_t>(layers.ptr());
  int scalar_type_key = static_cast<int>(hidden_states.scalar_type());
  auto* layer_cache = aten_layer_cache_store();
  auto cache_it = layer_cache->find(layers_key);
  bool can_reuse_refs =
      cache_it != layer_cache->end() &&
      cache_it->second.scalar_type_key == scalar_type_key &&
      cache_it->second.layer_count == layer_count;

  const std::vector<AtenLayerRefs>* layer_refs_ptr = nullptr;
  if (can_reuse_refs) {
    layer_refs_ptr = &cache_it->second.refs;
  } else {
    std::vector<AtenLayerRefs> fresh_refs;
    fresh_refs.reserve(static_cast<size_t>(layer_count));
    for (py::handle layer_handle : layers) {
      py::object layer = py::reinterpret_borrow<py::object>(layer_handle);
      AtenLayerRefs refs{
        layer.attr("self_attn"),
        layer.attr("mlp"),
        torch::Tensor(),
        torch::Tensor(),
        false,
        torch::Tensor(),
        torch::Tensor(),
        false,
        torch::Tensor(),
        torch::Tensor(),
        false,
        torch::Tensor(),
        torch::Tensor(),
        false,
        torch::Tensor(),
        torch::Tensor(),
        false,
        torch::Tensor(),
        torch::Tensor(),
        false,
        torch::Tensor(),
        torch::Tensor(),
        false,
        torch::Tensor(),
        torch::Tensor(),
        {},
        false,
        false,
        torch::Tensor(),
        torch::Tensor(),
        {},
        false,
        false,
        torch::Tensor(),
        0.0,
        torch::Tensor(),
        torch::Tensor(),
        0.0,
        0.0,
        0.0,
        0,
        torch::Tensor(),
        0.0,
    };
      py::object q_proj = refs.self_attn.attr("q_proj");
      py::object k_proj = refs.self_attn.attr("k_proj");
      py::object v_proj = refs.self_attn.attr("v_proj");
      py::object o_proj = refs.self_attn.attr("o_proj");
      refs.q_proj_weight = q_proj.attr("weight").cast<torch::Tensor>();
      refs.k_proj_weight = k_proj.attr("weight").cast<torch::Tensor>();
      refs.v_proj_weight = v_proj.attr("weight").cast<torch::Tensor>();
      refs.o_proj_weight = o_proj.attr("weight").cast<torch::Tensor>();
      py::object q_proj_bias_obj = q_proj.attr("bias");
      py::object k_proj_bias_obj = k_proj.attr("bias");
      py::object v_proj_bias_obj = v_proj.attr("bias");
      py::object o_proj_bias_obj = o_proj.attr("bias");
      refs.q_proj_has_bias = !q_proj_bias_obj.is_none();
      refs.k_proj_has_bias = !k_proj_bias_obj.is_none();
      refs.v_proj_has_bias = !v_proj_bias_obj.is_none();
      refs.o_proj_has_bias = !o_proj_bias_obj.is_none();
      if (refs.q_proj_has_bias) {
        refs.q_proj_bias = q_proj_bias_obj.cast<torch::Tensor>();
      }
      if (refs.k_proj_has_bias) {
        refs.k_proj_bias = k_proj_bias_obj.cast<torch::Tensor>();
      }
      if (refs.v_proj_has_bias) {
        refs.v_proj_bias = v_proj_bias_obj.cast<torch::Tensor>();
      }
      if (refs.o_proj_has_bias) {
        refs.o_proj_bias = o_proj_bias_obj.cast<torch::Tensor>();
      }
      py::object mlp_gate_proj = refs.mlp.attr("gate_proj");
      py::object mlp_up_proj = refs.mlp.attr("up_proj");
      py::object mlp_down_proj = refs.mlp.attr("down_proj");
      refs.mlp_gate_proj_weight = mlp_gate_proj.attr("weight").cast<torch::Tensor>();
      refs.mlp_up_proj_weight = mlp_up_proj.attr("weight").cast<torch::Tensor>();
      refs.mlp_down_proj_weight = mlp_down_proj.attr("weight").cast<torch::Tensor>();
      py::object mlp_gate_proj_bias_obj = mlp_gate_proj.attr("bias");
      py::object mlp_up_proj_bias_obj = mlp_up_proj.attr("bias");
      py::object mlp_down_proj_bias_obj = mlp_down_proj.attr("bias");
      refs.mlp_gate_proj_has_bias = !mlp_gate_proj_bias_obj.is_none();
      refs.mlp_up_proj_has_bias = !mlp_up_proj_bias_obj.is_none();
      refs.mlp_down_proj_has_bias = !mlp_down_proj_bias_obj.is_none();
      if (refs.mlp_gate_proj_has_bias) {
        refs.mlp_gate_proj_bias = mlp_gate_proj_bias_obj.cast<torch::Tensor>();
      }
      if (refs.mlp_up_proj_has_bias) {
        refs.mlp_up_proj_bias = mlp_up_proj_bias_obj.cast<torch::Tensor>();
      }
      if (refs.mlp_down_proj_has_bias) {
        refs.mlp_down_proj_bias = mlp_down_proj_bias_obj.cast<torch::Tensor>();
      }
      py::object input_layernorm = layer.attr("input_layernorm");
      refs.input_layernorm_weight = input_layernorm.attr("weight").cast<torch::Tensor>();
      refs.input_layernorm_eps = py::float_(input_layernorm.attr("variance_epsilon"));
      py::object q_norm = refs.self_attn.attr("q_norm");
      py::object k_norm = refs.self_attn.attr("k_norm");
      refs.q_norm_weight = q_norm.attr("weight").cast<torch::Tensor>();
      refs.k_norm_weight = k_norm.attr("weight").cast<torch::Tensor>();
      refs.q_norm_eps = py::float_(q_norm.attr("variance_epsilon"));
      refs.k_norm_eps = py::float_(k_norm.attr("variance_epsilon"));
      refs.attn_scaling = py::float_(refs.self_attn.attr("scaling"));
      refs.head_dim = py::int_(refs.self_attn.attr("head_dim"));
      py::object post_attention_layernorm = layer.attr("post_attention_layernorm");
      refs.post_attention_layernorm_weight = post_attention_layernorm.attr("weight").cast<torch::Tensor>();
      refs.post_attention_layernorm_eps = py::float_(post_attention_layernorm.attr("variance_epsilon"));
      refs.qkv_has_pack = resolve_cached_linear_pack(
          refs.self_attn,
          "_aicas_qkv_fused_linear_cache",
          hidden_states.scalar_type(),
          &refs.qkv_fused_weight,
          &refs.qkv_fused_bias,
          &refs.qkv_split_sizes,
          &refs.qkv_has_bias);
      refs.gate_up_has_pack = resolve_cached_linear_pack(
          refs.mlp,
          "_aicas_gate_up_fused_linear_cache",
          hidden_states.scalar_type(),
          &refs.gate_up_fused_weight,
          &refs.gate_up_fused_bias,
          &refs.gate_up_split_sizes,
          &refs.gate_up_has_bias);
      fresh_refs.emplace_back(std::move(refs));
    }
    (*layer_cache)[layers_key] = AtenLayerCacheEntry{
        scalar_type_key,
        layer_count,
        std::move(fresh_refs),
    };
    layer_refs_ptr = &((*layer_cache)[layers_key].refs);
  }

  bool use_deepstack = !deepstack_visual_embeds_obj.is_none() && deepstack_count > 0;
  int64_t deepstack_contig_offset = -1;
  int64_t deepstack_contig_count = 0;
  if (use_deepstack) {
    TORCH_CHECK(!text_model_obj.is_none(), "text_model must be provided when deepstack is enabled");
    TORCH_CHECK(
        py::hasattr(text_model_obj, "_aicas_visual_token_offset") &&
            py::hasattr(text_model_obj, "_aicas_visual_token_count"),
        "text_model missing deepstack contig metadata");
    deepstack_contig_offset = py::int_(text_model_obj.attr("_aicas_visual_token_offset"));
    deepstack_contig_count = py::int_(text_model_obj.attr("_aicas_visual_token_count"));
    TORCH_CHECK(deepstack_contig_offset >= 0, "invalid deepstack token offset");
    TORCH_CHECK(deepstack_contig_count > 0, "invalid deepstack token count");
    if (cpp_runtime_validate_enabled()) {
      validate_deepstack_contiguous_mask(
          visual_pos_masks_obj,
          deepstack_contig_offset,
          deepstack_contig_count);
    }
  }
  std::vector<torch::Tensor> deepstack_embeds;
  if (use_deepstack) {
    auto deepstack_visual_embeds = deepstack_visual_embeds_obj.cast<py::sequence>();
    deepstack_embeds.reserve(static_cast<size_t>(deepstack_count));
    for (int64_t i = 0; i < deepstack_count; ++i) {
      deepstack_embeds.emplace_back(
          py::reinterpret_borrow<py::object>(deepstack_visual_embeds[static_cast<py::ssize_t>(i)])
              .cast<torch::Tensor>());
    }
  }

  TORCH_CHECK(layer_refs_ptr != nullptr, "internal error: missing aten layer refs");
  int64_t layer_idx = 0;
  std::optional<torch::Tensor> cached_input_norm;
  bool applied_final_add_rmsnorm = false;
  for (const auto& refs : *layer_refs_ptr) {
    py::object residual_obj = py::cast(hidden_states);

    if (cached_input_norm.has_value()) {
      hidden_states = cached_input_norm.value();
      cached_input_norm.reset();
    } else {
      if (use_cached_rmsnorm) {
        hidden_states = launch_rmsnorm_cached_triton(
            rmsnorm_launcher_obj,
            rmsnorm_function_obj,
            rmsnorm_packed_metadata_obj,
            stream_handle,
            rmsnorm_block_size,
            hidden_states,
            refs.input_layernorm_weight,
            refs.input_layernorm_eps);
      } else {
        hidden_states = rmsnorm_last_dim(
            hidden_states,
            refs.input_layernorm_weight,
            refs.input_layernorm_eps);
      }
    }

    auto q_norm_weight = refs.q_norm_weight;
    auto k_norm_weight = refs.k_norm_weight;
    double q_norm_eps = refs.q_norm_eps;
    double k_norm_eps = refs.k_norm_eps;
    double attn_scaling = refs.attn_scaling;
    auto input_shape = hidden_states.sizes().vec();
    int64_t head_dim = refs.head_dim;
    std::vector<int64_t> hidden_shape = {
        input_shape[0],
        input_shape[1],
        -1,
        head_dim,
    };

    torch::Tensor q_states;
    torch::Tensor k_states;
    torch::Tensor v_states;
    std::vector<torch::Tensor> qkv_parts;
    bool used_fused_qkv = refs.qkv_has_pack &&
                          linear_from_resolved_pack(
                              refs.qkv_fused_weight,
                              refs.qkv_fused_bias,
                              refs.qkv_has_bias,
                              refs.qkv_split_sizes,
                              hidden_states,
                              &qkv_parts) &&
                          qkv_parts.size() == 3;
    if (used_fused_qkv) {
      q_states = qkv_parts[0];
      k_states = qkv_parts[1];
      v_states = qkv_parts[2];
    } else {
      q_states = linear_from_weight_bias(refs.q_proj_weight, refs.q_proj_bias, refs.q_proj_has_bias, hidden_states);
      k_states = linear_from_weight_bias(refs.k_proj_weight, refs.k_proj_bias, refs.k_proj_has_bias, hidden_states);
      v_states = linear_from_weight_bias(refs.v_proj_weight, refs.v_proj_bias, refs.v_proj_has_bias, hidden_states);
    }
    auto query_states = q_states.view(hidden_shape);
    auto key_states = k_states.view(hidden_shape);
    auto value_states = v_states.view(hidden_shape).transpose(1, 2);
    query_states = query_states.transpose(1, 2);
    key_states = key_states.transpose(1, 2);

    if (use_cached_prefill_qk_rope) {
      auto qk_rope_out = launch_prefill_qk_rope_cached_triton(
                             prefill_qk_rope_launcher_obj,
                             prefill_qk_rope_function_obj,
                             prefill_qk_rope_packed_metadata_obj,
                             stream_handle,
                             prefill_qk_rope_block_d,
                             query_states,
                             key_states,
                             q_norm_weight,
                             k_norm_weight,
                             rotary_cos,
                             rotary_sin,
                             q_norm_eps,
                             k_norm_eps)
                             .cast<py::tuple>();
      query_states = qk_rope_out[0].cast<torch::Tensor>();
      key_states = qk_rope_out[1].cast<torch::Tensor>();
    } else {
      if (use_cached_rmsnorm) {
        query_states = launch_rmsnorm_cached_triton(
            rmsnorm_launcher_obj,
            rmsnorm_function_obj,
            rmsnorm_packed_metadata_obj,
            stream_handle,
            rmsnorm_block_size,
            query_states,
            q_norm_weight,
            q_norm_eps);
        key_states = launch_rmsnorm_cached_triton(
            rmsnorm_launcher_obj,
            rmsnorm_function_obj,
            rmsnorm_packed_metadata_obj,
            stream_handle,
            rmsnorm_block_size,
            key_states,
            k_norm_weight,
            k_norm_eps);
      } else {
        query_states = rmsnorm_last_dim(
            query_states,
            q_norm_weight,
            q_norm_eps);
        key_states = rmsnorm_last_dim(
            key_states,
            k_norm_weight,
            k_norm_eps);
      }
      std::tie(query_states, key_states) =
          apply_rotary_pos_emb_aten(query_states, key_states, rotary_cos, rotary_sin);
    }

    auto attention_mask_for_sdpa = attention_mask;
    auto key_states_attn = key_states;
    auto value_states_attn = value_states;
    if (key_states_attn.size(2) > query_states.size(2)) {
      key_states_attn = key_states_attn.slice(2, 0, query_states.size(2));
      value_states_attn = value_states_attn.slice(2, 0, query_states.size(2));
    }
    bool enable_gqa = query_states.size(1) != key_states_attn.size(1);
    bool use_causal = !attention_mask_for_sdpa.has_value();
    if (attention_mask_for_sdpa.has_value()) {
      auto mask = attention_mask_for_sdpa.value();
      int64_t q_len = query_states.size(2);
      int64_t kv_len = key_states_attn.size(2);
      bool has_expandable_shape =
          mask.dim() == 4 && mask.size(2) == q_len && mask.size(3) >= kv_len;
      bool can_use_causal_fastpath =
          has_expandable_shape && mask.scalar_type() == torch::kBool;
      if (can_use_causal_fastpath) {
        attention_mask_for_sdpa = std::nullopt;
        use_causal = true;
      } else if (has_expandable_shape && mask.size(3) != kv_len) {
        attention_mask_for_sdpa = mask.slice(3, 0, kv_len);
      }
    }
    auto attn_output = at::scaled_dot_product_attention(
                           query_states,
                           key_states_attn,
                           value_states_attn,
                           attention_mask_for_sdpa,
                           0.0,
                           use_causal,
                           attn_scaling,
                           enable_gqa)
                           .transpose(1, 2)
                           .contiguous()
                           .reshape(input_shape);
    hidden_states = linear_from_weight_bias(refs.o_proj_weight, refs.o_proj_bias, refs.o_proj_has_bias, attn_output);

    if (!past_key_values_obj.is_none()) {
      bool cache_updated_in_cpp = false;
      if (py::hasattr(past_key_values_obj, "layers")) {
        py::object all_layers_obj = past_key_values_obj.attr("layers");
        auto cache_layer_count = static_cast<int64_t>(py::len(all_layers_obj));
        if (layer_idx < cache_layer_count) {
          py::object cache_layer_obj = all_layers_obj[py::int_(layer_idx)];
          cache_updated_in_cpp = try_update_cache_layer_in_cpp(
              cache_layer_obj,
              key_states,
              value_states,
              cache_position,
              use_cached_kv_cache_write ? kv_cache_write_launcher_obj : py::none(),
              use_cached_kv_cache_write ? kv_cache_write_function_obj : py::none(),
              use_cached_kv_cache_write ? kv_cache_write_packed_metadata_obj : py::none(),
              stream_handle,
              use_cached_kv_cache_write ? kv_cache_write_block_t : 0,
              use_cached_kv_cache_write ? kv_cache_write_block_d : 0,
              &key_states,
              &value_states);
        }
      }
      if (!cache_updated_in_cpp) {
        python_cache_update_fallback_counter().fetch_add(1, std::memory_order_relaxed);
        py::dict cache_kwargs;
        cache_kwargs["sin"] = rotary_sin;
        cache_kwargs["cos"] = rotary_cos;
        cache_kwargs["cache_position"] = cache_position;
        py::object _ = past_key_values_obj.attr("update")(
            key_states,
            value_states,
            py::int_(layer_idx),
            cache_kwargs);
        (void)_;
      }
    }

    py::tuple fused_out;
    if (use_cached_add_rmsnorm) {
      fused_out = launch_add_rmsnorm_cached_triton(
                      add_rmsnorm_launcher_obj,
                      add_rmsnorm_function_obj,
                      add_rmsnorm_packed_metadata_obj,
                      stream_handle,
                      add_rmsnorm_block_size,
                      residual_obj.cast<torch::Tensor>(),
                      hidden_states,
                      refs.post_attention_layernorm_weight,
                      refs.post_attention_layernorm_eps)
                      .cast<py::tuple>();
    } else {
      auto aten_out = add_rmsnorm_last_dim(
          residual_obj.cast<torch::Tensor>(),
          hidden_states,
          refs.post_attention_layernorm_weight,
          refs.post_attention_layernorm_eps);
      fused_out = py::make_tuple(aten_out.first, aten_out.second);
    }
    py::object residual_after_norm = fused_out[0];
    hidden_states = fused_out[1].cast<torch::Tensor>();

    hidden_states = mlp_forward_aten_refs(
        refs,
        hidden_states,
        use_cached_swiglu,
        swiglu_launcher_obj,
        swiglu_function_obj,
        swiglu_packed_metadata_obj,
        stream_handle,
        swiglu_block_size);
    bool will_apply_deepstack =
        use_deepstack && layer_idx < deepstack_count;
    bool can_handoff_next_input_norm =
        use_interlayer_add_rmsnorm_handoff &&
        use_cached_add_rmsnorm &&
        layer_idx + 1 < layer_count &&
        !will_apply_deepstack;
    if (can_handoff_next_input_norm) {
      const auto& next_refs = (*layer_refs_ptr)[static_cast<size_t>(layer_idx + 1)];
      auto next_norm_out = launch_add_rmsnorm_cached_triton(
                               add_rmsnorm_launcher_obj,
                               add_rmsnorm_function_obj,
                               add_rmsnorm_packed_metadata_obj,
                               stream_handle,
                               add_rmsnorm_block_size,
                               residual_after_norm.cast<torch::Tensor>(),
                               hidden_states,
                               next_refs.input_layernorm_weight,
                               next_refs.input_layernorm_eps)
                               .cast<py::tuple>();
      hidden_states = next_norm_out[0].cast<torch::Tensor>();
      cached_input_norm = next_norm_out[1].cast<torch::Tensor>();
    } else if (use_final_add_rmsnorm && use_cached_add_rmsnorm && !will_apply_deepstack &&
               layer_idx + 1 == layer_count) {
      auto final_norm_out = launch_add_rmsnorm_cached_triton(
                                add_rmsnorm_launcher_obj,
                                add_rmsnorm_function_obj,
                                add_rmsnorm_packed_metadata_obj,
                                stream_handle,
                                add_rmsnorm_block_size,
                                residual_after_norm.cast<torch::Tensor>(),
                                hidden_states,
                                final_norm_weight_obj.cast<torch::Tensor>(),
                                final_norm_eps)
                                .cast<py::tuple>();
      hidden_states = final_norm_out[1].cast<torch::Tensor>();
      applied_final_add_rmsnorm = true;
    } else {
      hidden_states = residual_after_norm.cast<torch::Tensor>() + hidden_states;
    }

    if (will_apply_deepstack) {
      auto& deepstack_embed = deepstack_embeds[static_cast<size_t>(layer_idx)];
      TORCH_CHECK(
          try_apply_deepstack_contiguous_add(
              &hidden_states,
              deepstack_embed,
              deepstack_contig_offset,
              deepstack_contig_count),
          "C++ deepstack contiguous fastpath preconditions failed");
    }
    ++layer_idx;
  }
  if (applied_final_add_rmsnorm) {
    return hidden_states;
  }
  auto final_norm_weight = final_norm_weight_obj.cast<torch::Tensor>();
  if (use_cached_rmsnorm) {
    hidden_states = launch_rmsnorm_cached_triton(
        rmsnorm_launcher_obj,
        rmsnorm_function_obj,
        rmsnorm_packed_metadata_obj,
        stream_handle,
        rmsnorm_block_size,
        hidden_states,
        final_norm_weight,
        final_norm_eps);
  } else {
    hidden_states = rmsnorm_last_dim(hidden_states, final_norm_weight, final_norm_eps);
  }
  return hidden_states;
}

py::tuple prefill_text_model_forward_aten(
    py::object text_model_obj,
    py::object input_ids_obj,
    py::object attention_mask_obj,
    py::object position_ids_obj,
    py::object past_key_values_obj,
    py::object inputs_embeds_obj,
    py::object use_cache_obj,
    py::object cache_position_obj,
    py::object visual_pos_masks_obj,
    py::object deepstack_visual_embeds_obj,
    py::object create_causal_mask_obj,
    py::object dynamic_cache_cls,
    py::object fused_add_rmsnorm_obj,
    py::dict extra_kwargs,
    py::object cuda_current_stream_obj,
    py::object prefill_qk_rope_launcher_obj,
    py::object prefill_qk_rope_function_obj,
    py::object prefill_qk_rope_packed_metadata_obj,
    int64_t prefill_qk_rope_block_d,
    py::object add_rmsnorm_launcher_obj,
    py::object add_rmsnorm_function_obj,
    py::object add_rmsnorm_packed_metadata_obj,
    int64_t add_rmsnorm_block_size,
    py::object rmsnorm_launcher_obj,
    py::object rmsnorm_function_obj,
    py::object rmsnorm_packed_metadata_obj,
    int64_t rmsnorm_block_size,
    py::object swiglu_launcher_obj,
    py::object swiglu_function_obj,
    py::object swiglu_packed_metadata_obj,
    int64_t swiglu_block_size,
    py::object prefill_rope_launcher_obj,
    py::object prefill_rope_function_obj,
    py::object prefill_rope_packed_metadata_obj,
    int64_t prefill_rope_block_d,
    py::object kv_cache_write_launcher_obj,
    py::object kv_cache_write_function_obj,
    py::object kv_cache_write_packed_metadata_obj,
    int64_t kv_cache_write_block_t,
    int64_t kv_cache_write_block_d,
    py::object final_norm_weight_obj,
    double final_norm_eps) {
  TORCH_CHECK(!text_model_obj.is_none(), "text_model must be provided");

  bool has_input_ids = !input_ids_obj.is_none();
  bool has_inputs_embeds = !inputs_embeds_obj.is_none();
  TORCH_CHECK(
      has_input_ids ^ has_inputs_embeds,
      "You must specify exactly one of input_ids or inputs_embeds");

  bool use_cache = !use_cache_obj.is_none() && py::bool_(use_cache_obj);
  bool is_tracing = py::module_::import("torch").attr("jit").attr("is_tracing")().cast<bool>();
  auto module_config_obj = text_model_obj.attr("config");
  past_key_values_obj = ensure_dynamic_cache(
      past_key_values_obj,
      use_cache,
      is_tracing,
      dynamic_cache_cls,
      module_config_obj);

  torch::Tensor inputs_embeds;
  if (has_inputs_embeds) {
    inputs_embeds = inputs_embeds_obj.cast<torch::Tensor>();
  } else {
    inputs_embeds = embed_tokens_cpp(text_model_obj, input_ids_obj);
  }

  auto cache_position = build_cache_position(past_key_values_obj, cache_position_obj, inputs_embeds);
  auto normalized_positions = normalize_position_ids(position_ids_obj, cache_position, inputs_embeds);
  auto position_ids = normalized_positions[0].cast<torch::Tensor>();
  auto text_position_ids = normalized_positions[1].cast<torch::Tensor>();
  auto attention_mask = build_attention_mask(
      create_causal_mask_obj,
      module_config_obj,
      inputs_embeds,
      attention_mask_obj,
      cache_position,
      past_key_values_obj,
      text_position_ids,
      is_tracing);
  auto position_embeddings = build_position_embeddings_cpp(
      text_model_obj,
      inputs_embeds,
      position_ids,
      prefill_rope_launcher_obj,
      prefill_rope_function_obj,
      prefill_rope_packed_metadata_obj,
      prefill_rope_block_d,
      cuda_current_stream_obj);

  py::object layers_obj = py::none();
  if (py::hasattr(text_model_obj, "_aicas_cpp_runtime_layer_list")) {
    layers_obj = text_model_obj.attr("_aicas_cpp_runtime_layer_list");
  }
  if (layers_obj.is_none()) {
    auto layer_list = py::list();
    for (py::handle layer_handle : text_model_obj.attr("layers")) {
      layer_list.append(py::reinterpret_borrow<py::object>(layer_handle));
    }
    text_model_obj.attr("_aicas_cpp_runtime_layer_list") = layer_list;
    layers_obj = layer_list;
  }

  auto hidden_states = prefill_text_runtime_run_aten_ops(
      layers_obj,
      inputs_embeds,
      text_position_ids,
      cache_position,
      position_embeddings.first,
      position_embeddings.second,
      attention_mask,
      past_key_values_obj,
      visual_pos_masks_obj,
      deepstack_visual_embeds_obj,
      text_model_obj,
      fused_add_rmsnorm_obj,
      extra_kwargs,
      cuda_current_stream_obj,
      prefill_qk_rope_launcher_obj,
      prefill_qk_rope_function_obj,
      prefill_qk_rope_packed_metadata_obj,
      prefill_qk_rope_block_d,
      add_rmsnorm_launcher_obj,
      add_rmsnorm_function_obj,
      add_rmsnorm_packed_metadata_obj,
      add_rmsnorm_block_size,
      rmsnorm_launcher_obj,
      rmsnorm_function_obj,
      rmsnorm_packed_metadata_obj,
      rmsnorm_block_size,
      swiglu_launcher_obj,
      swiglu_function_obj,
      swiglu_packed_metadata_obj,
      swiglu_block_size,
      kv_cache_write_launcher_obj,
      kv_cache_write_function_obj,
      kv_cache_write_packed_metadata_obj,
      kv_cache_write_block_t,
      kv_cache_write_block_d,
      final_norm_weight_obj,
      final_norm_eps);
  return py::make_tuple(hidden_states, past_key_values_obj);
}

py::dict runtime_metadata() {
  py::dict out;
  out["name"] = "aicas_prefill_text_runtime";
  out["stage"] = "runner_aten_ops";
  out["has_cuda"] = torch::cuda::is_available();
  out["notes"] = "Stage-5 runner: C++ drives prefill layer loop plus ATen attention/MLP internals.";
  out["cpp_cache_update_count"] = cpp_cache_update_counter().load(std::memory_order_relaxed);
  out["prefill_kv_cache_write_count"] = prefill_kv_cache_write_counter().load(std::memory_order_relaxed);
  out["python_cache_update_fallback_count"] = python_cache_update_fallback_counter().load(std::memory_order_relaxed);
  out["prefill_qk_rope_count"] = prefill_qk_rope_counter().load(std::memory_order_relaxed);
  out["prefill_rope_count"] = prefill_rope_counter().load(std::memory_order_relaxed);
  out["add_rmsnorm_count"] = add_rmsnorm_counter().load(std::memory_order_relaxed);
  out["rmsnorm_count"] = rmsnorm_counter().load(std::memory_order_relaxed);
  out["swiglu_count"] = swiglu_counter().load(std::memory_order_relaxed);
  out["vision_runtime_forward_count"] = vision_runtime_forward_counter().load(std::memory_order_relaxed);
  out["vision_get_image_features_count"] = vision_get_image_features_counter().load(std::memory_order_relaxed);
  out["vision_setup_cache_enabled"] = cpp_vision_setup_cache_enabled();
  out["vision_setup_cache_hit_count"] = vision_setup_cache_hit_counter().load(std::memory_order_relaxed);
  out["vision_setup_cache_miss_count"] = vision_setup_cache_miss_counter().load(std::memory_order_relaxed);
  out["vision_setup_cache_entries"] = static_cast<int64_t>(vision_setup_cache_store()->size());
  out["text_interlayer_add_rmsnorm_enabled"] = cpp_text_interlayer_add_rmsnorm_enabled();
  out["text_final_add_rmsnorm_enabled"] = cpp_text_final_add_rmsnorm_enabled();
  return out;
}

void reset_runtime_counters() {
  cpp_cache_update_counter().store(0, std::memory_order_relaxed);
  prefill_kv_cache_write_counter().store(0, std::memory_order_relaxed);
  python_cache_update_fallback_counter().store(0, std::memory_order_relaxed);
  prefill_qk_rope_counter().store(0, std::memory_order_relaxed);
  prefill_rope_counter().store(0, std::memory_order_relaxed);
  add_rmsnorm_counter().store(0, std::memory_order_relaxed);
  rmsnorm_counter().store(0, std::memory_order_relaxed);
  swiglu_counter().store(0, std::memory_order_relaxed);
  vision_runtime_forward_counter().store(0, std::memory_order_relaxed);
  vision_get_image_features_counter().store(0, std::memory_order_relaxed);
  vision_setup_cache_hit_counter().store(0, std::memory_order_relaxed);
  vision_setup_cache_miss_counter().store(0, std::memory_order_relaxed);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "AICAS prefill text runtime skeleton";
  m.def(
      "launch_add_layernorm_cached_triton",
      &launch_add_layernorm_cached_triton,
      py::arg("launcher"),
      py::arg("function"),
      py::arg("packed_metadata"),
      py::arg("cuda_current_stream"),
      py::arg("block_size"),
      py::arg("x"),
      py::arg("y"),
      py::arg("weight"),
      py::arg("bias"),
      py::arg("eps"));
  m.def(
      "launch_vision_rotary_cached_triton",
      &launch_vision_rotary_cached_triton,
      py::arg("launcher"),
      py::arg("function"),
      py::arg("packed_metadata"),
      py::arg("cuda_current_stream"),
      py::arg("block_d"),
      py::arg("q"),
      py::arg("k"),
      py::arg("cos"),
      py::arg("sin"),
      py::arg("out_q") = py::none(),
      py::arg("out_k") = py::none());
  m.def(
      "launch_prefill_rope_cached_triton",
      &launch_prefill_rope_cached_triton,
      py::arg("launcher"),
      py::arg("function"),
      py::arg("packed_metadata"),
      py::arg("cuda_current_stream"),
      py::arg("block_d"),
      py::arg("position_ids"),
      py::arg("inv_freq"),
      py::arg("out_cos"),
      py::arg("out_sin"),
      py::arg("seq_len"),
      py::arg("attention_scaling"),
      py::arg("len_h"),
      py::arg("len_w"));
  m.def(
      "vision_runtime_forward",
      &vision_runtime_forward,
      py::arg("blocks"),
      py::arg("deepstack_indexes"),
      py::arg("deepstack_mergers"),
      py::arg("merger"),
      py::arg("hidden_states"),
      py::arg("first_norm1"),
      py::arg("cu_seqlens"),
      py::arg("rotary_cos"),
      py::arg("rotary_sin"),
      py::arg("chunk_lengths"),
      py::arg("extra_kwargs"),
      py::arg("add_layernorm_launcher") = py::none(),
      py::arg("add_layernorm_function") = py::none(),
      py::arg("add_layernorm_packed_metadata") = py::none(),
      py::arg("cuda_current_stream") = py::none(),
      py::arg("add_layernorm_block_size") = 0,
      py::arg("interlayer_fuse_enabled") = true,
      py::arg("cached_norm_handoff_enabled") = true,
      py::arg("vision_rotary_launcher") = py::none(),
      py::arg("vision_rotary_function") = py::none(),
      py::arg("vision_rotary_packed_metadata") = py::none(),
      py::arg("vision_rotary_block_d") = 0);
  m.def(
      "vision_get_image_features_aten",
      &vision_get_image_features_aten,
      py::arg("vision_model"),
      py::arg("pixel_values"),
      py::arg("image_grid_thw"),
      py::arg("add_layernorm_launcher") = py::none(),
      py::arg("add_layernorm_function") = py::none(),
      py::arg("add_layernorm_packed_metadata") = py::none(),
      py::arg("cuda_current_stream") = py::none(),
      py::arg("add_layernorm_block_size") = 0,
      py::arg("interlayer_fuse_enabled") = true,
      py::arg("cached_norm_handoff_enabled") = true,
      py::arg("vision_rotary_launcher") = py::none(),
      py::arg("vision_rotary_function") = py::none(),
      py::arg("vision_rotary_packed_metadata") = py::none(),
      py::arg("vision_rotary_block_d") = 0,
      py::arg("grid_t") = 0,
      py::arg("grid_h") = 0,
      py::arg("grid_w") = 0);
  m.def(
      "embed_tokens_aten",
      &embed_tokens_cpp,
      py::arg("text_model"),
      py::arg("input_ids"));
  m.def(
      "prefill_text_model_forward_aten",
      &prefill_text_model_forward_aten,
      py::arg("text_model"),
      py::arg("input_ids") = py::none(),
      py::arg("attention_mask") = py::none(),
      py::arg("position_ids") = py::none(),
      py::arg("past_key_values") = py::none(),
      py::arg("inputs_embeds") = py::none(),
      py::arg("use_cache") = py::none(),
      py::arg("cache_position") = py::none(),
      py::arg("visual_pos_masks") = py::none(),
      py::arg("deepstack_visual_embeds") = py::none(),
      py::arg("create_causal_mask"),
      py::arg("dynamic_cache_cls"),
      py::arg("fused_add_rmsnorm"),
      py::arg("extra_kwargs"),
      py::arg("cuda_current_stream") = py::none(),
      py::arg("prefill_qk_rope_launcher") = py::none(),
      py::arg("prefill_qk_rope_function") = py::none(),
      py::arg("prefill_qk_rope_packed_metadata") = py::none(),
      py::arg("prefill_qk_rope_block_d") = 0,
      py::arg("add_rmsnorm_launcher") = py::none(),
      py::arg("add_rmsnorm_function") = py::none(),
      py::arg("add_rmsnorm_packed_metadata") = py::none(),
      py::arg("add_rmsnorm_block_size") = 0,
      py::arg("rmsnorm_launcher") = py::none(),
      py::arg("rmsnorm_function") = py::none(),
      py::arg("rmsnorm_packed_metadata") = py::none(),
      py::arg("rmsnorm_block_size") = 0,
      py::arg("swiglu_launcher") = py::none(),
      py::arg("swiglu_function") = py::none(),
      py::arg("swiglu_packed_metadata") = py::none(),
      py::arg("swiglu_block_size") = 0,
      py::arg("prefill_rope_launcher") = py::none(),
      py::arg("prefill_rope_function") = py::none(),
      py::arg("prefill_rope_packed_metadata") = py::none(),
      py::arg("prefill_rope_block_d") = 0,
      py::arg("kv_cache_write_launcher") = py::none(),
      py::arg("kv_cache_write_function") = py::none(),
      py::arg("kv_cache_write_packed_metadata") = py::none(),
      py::arg("kv_cache_write_block_t") = 0,
      py::arg("kv_cache_write_block_d") = 0,
      py::arg("final_norm_weight"),
      py::arg("final_norm_eps"));
  m.def(
      "prefill_text_runtime_run_aten_ops",
      &prefill_text_runtime_run_aten_ops,
      py::arg("layers"),
      py::arg("hidden_states"),
      py::arg("text_position_ids"),
      py::arg("cache_position"),
      py::arg("rotary_cos"),
      py::arg("rotary_sin"),
      py::arg("attention_mask"),
      py::arg("past_key_values"),
      py::arg("visual_pos_masks"),
      py::arg("deepstack_visual_embeds"),
      py::arg("text_model"),
      py::arg("fused_add_rmsnorm"),
      py::arg("extra_kwargs"),
      py::arg("cuda_current_stream") = py::none(),
      py::arg("prefill_qk_rope_launcher") = py::none(),
      py::arg("prefill_qk_rope_function") = py::none(),
      py::arg("prefill_qk_rope_packed_metadata") = py::none(),
      py::arg("prefill_qk_rope_block_d") = 0,
      py::arg("add_rmsnorm_launcher") = py::none(),
      py::arg("add_rmsnorm_function") = py::none(),
      py::arg("add_rmsnorm_packed_metadata") = py::none(),
      py::arg("add_rmsnorm_block_size") = 0,
      py::arg("rmsnorm_launcher") = py::none(),
      py::arg("rmsnorm_function") = py::none(),
      py::arg("rmsnorm_packed_metadata") = py::none(),
      py::arg("rmsnorm_block_size") = 0,
      py::arg("swiglu_launcher") = py::none(),
      py::arg("swiglu_function") = py::none(),
      py::arg("swiglu_packed_metadata") = py::none(),
      py::arg("swiglu_block_size") = 0,
      py::arg("kv_cache_write_launcher") = py::none(),
      py::arg("kv_cache_write_function") = py::none(),
      py::arg("kv_cache_write_packed_metadata") = py::none(),
      py::arg("kv_cache_write_block_t") = 0,
      py::arg("kv_cache_write_block_d") = 0,
      py::arg("final_norm_weight"),
      py::arg("final_norm_eps"));
  m.def("runtime_metadata", &runtime_metadata);
  m.def("reset_runtime_counters", &reset_runtime_counters);
}
