constexpr uint32_t kHidden = VLM_W8A8_TEXT_HIDDEN;
constexpr uint32_t kKv = VLM_W8A8_TEXT_KV;
constexpr uint32_t kFfn = VLM_W8A8_TEXT_FFN;
constexpr uint32_t kVisionHidden = VLM_W8A8_VISION_HIDDEN;
constexpr uint32_t kVisionFfn = VLM_W8A8_VISION_FFN;
constexpr uint32_t kGenericMaxDim = VLM_W8A8_LINEAR_GENERIC_MAX_DIM;
constexpr uint32_t kTileN = VLM_W8A8_LINEAR_TILE_N;
constexpr uint32_t kWeightPortCount = VLM_W8A8_WEIGHT_PORTS;
constexpr uint32_t kWeightPortTileN = kTileN / kWeightPortCount;
constexpr uint32_t kPeN = 32;
constexpr uint32_t kNPePasses = kTileN / kPeN;
constexpr uint32_t kReducePeN = 1;
constexpr uint32_t kAccBankN = 8;
constexpr uint32_t kQkvoPeM = 6;
constexpr uint32_t kFfnPeM = 6;
constexpr uint32_t kMaxPeM = kQkvoPeM;
constexpr uint32_t kTileK = VLM_W8A8_LINEAR_K_BLOCK;
constexpr uint32_t kScalePerTile = kTileK / VLM_W8A8_QK;
constexpr uint32_t kKSubTile = 8;
constexpr uint32_t kQkSubTileCount = VLM_W8A8_QK / kKSubTile;
constexpr uint32_t kWeightWordsPerLane = kTileK / 16u;
constexpr uint32_t kWeightChunksPerLane = kTileK / 4u;
constexpr uint32_t kDenseTileM = kMaxPeM;
constexpr uint32_t kFfnTileM = kMaxPeM;
constexpr uint32_t kDecodeTileM = 1;
constexpr uint32_t kMaxTileM = kMaxPeM;
constexpr uint32_t kHiddenPadded = ((kHidden + kTileK - 1u) / kTileK) * kTileK;
constexpr uint32_t kHiddenWordsPerRow = kHiddenPadded / 16u;
constexpr uint32_t kHiddenTileCount = kHiddenPadded / kTileK;
constexpr uint32_t kFfnPadded = ((kFfn + kTileK - 1u) / kTileK) * kTileK;
constexpr uint32_t kFfnTileCount = kFfnPadded / kTileK;
constexpr uint32_t kVisionFfnPadded = ((kVisionFfn + kTileK - 1u) / kTileK) * kTileK;
constexpr uint32_t kMaxFfnPadded = (kVisionFfnPadded > kFfnPadded) ? kVisionFfnPadded : kFfnPadded;
constexpr uint32_t kGenericPadded = ((kGenericMaxDim + kTileK - 1u) / kTileK) * kTileK;
constexpr uint32_t kArenaDepthWords = 1048576;
constexpr uint32_t kFfnMidWordsPerRow = kMaxFfnPadded / 16u;
constexpr uint32_t kFfnScaleGroups = kMaxFfnPadded / VLM_W8A8_QK;
constexpr uint32_t kGenericWordsPerRow = kGenericPadded / 16u;
constexpr uint32_t kGenericScaleGroups = kGenericPadded / VLM_W8A8_QK;
constexpr uint32_t kMaxActWordsPerRow = (kGenericWordsPerRow > kFfnMidWordsPerRow) ? kGenericWordsPerRow : kFfnMidWordsPerRow;
constexpr uint32_t kMaxScaleGroups = (kGenericScaleGroups > kFfnScaleGroups) ? kGenericScaleGroups : kFfnScaleGroups;
constexpr uint32_t kMaxActScaleWordsPerRow = (kMaxScaleGroups + 15u) / 16u;
constexpr int kScaleExpDisabled = -128;
constexpr int kV2OutFrac = 24;

typedef ap_int<20> dot_acc_t;
typedef ap_int<8> scale_exp_t;
typedef ap_int<48> scaled_acc_t;
typedef linear_task_t accelerator_task_t;

struct weight_scale_exp_t {
    scale_exp_t e0;
    scale_exp_t e1;
};

constexpr uint32_t kTaskTransferWords = sizeof(accelerator_task_t) / 16u;
constexpr uint32_t kProfileTransferWords = sizeof(vlm_w8a8_profile_t) / 16u;

union task_buffer_t {
    accelerator_task_t task;
    vlm_w8a8_axi_t words[kTaskTransferWords];
};

union profile_buffer_t {
    vlm_w8a8_profile_t profile;
    vlm_w8a8_axi_t words[kProfileTransferWords];
};

static_assert((sizeof(accelerator_task_t) % 16u) == 0u, "accelerator task must be 16-byte aligned");
static_assert((kTileN % kWeightPortCount) == 0u, "tile N must split evenly across weight ports");
static_assert((kTileN % kPeN) == 0u, "tile N must be an integer number of PE passes");
static_assert((VLM_W8A8_QK % kKSubTile) == 0u, "group32 must be an integer number of K subtiles");
static_assert(sizeof(task_buffer_t) == sizeof(accelerator_task_t), "task buffer size mismatch");
static_assert((sizeof(vlm_w8a8_profile_t) % 16u) == 0u, "profile must be 16-byte aligned");
static_assert(sizeof(vlm_w8a8_profile_t) == 128u, "profile layout must remain 128 bytes");
static_assert(sizeof(profile_buffer_t) == sizeof(vlm_w8a8_profile_t), "profile buffer size mismatch");
