`default_nettype none

// Shared Phase-1 SPM/SA definitions.
//
// Purpose:
//   Keep SPM/SA mode, error, and stream constants in one package so
//   independent submodules do not duplicate protocol encodings.
//
// Phase-1 scope:
//   INT8 GEMM/QK is the public mode. PV log8P(U15)xINT8 is an internal compute
//   mode for the attention path. BFP16-M remains unsupported.
package npu_spm_pkg;
    localparam int NPU_SPM_DATA_W       = 256;
    localparam int NPU_SPM_WORD_BYTES   = NPU_SPM_DATA_W / 8;
    localparam int NPU_SPM_BYTE_LANES   = NPU_SPM_WORD_BYTES;
    localparam int NPU_ACC_LANES_I32    = 8;
    localparam int NPU_SA_ROWS          = 32;
    localparam int NPU_SA_COLS          = 32;
    localparam int NPU_VIRTUAL_M_INT8   = 32;
    localparam int NPU_K_BLOCK_MAX      = 4096;
    localparam int NPU_BANK_WORDS       = 16384;

    typedef enum logic [1:0] {
        NPU_MODE_INT8      = 2'd0,
        NPU_MODE_BFP16M    = 2'd1,
        NPU_MODE_PV_LOG8_U16I8 = 2'd2,
        NPU_MODE_ATTENTION_QK = 2'd3
    } npu_mode_e;

    typedef enum logic [3:0] {
        NPU_ERR_NONE              = 4'd0,
        NPU_ERR_UNSUPPORTED_MODE  = 4'd1,
        NPU_ERR_ILLEGAL_SHAPE     = 4'd2,
        NPU_ERR_BANK_CONFLICT     = 4'd3,
        NPU_ERR_ALIGN             = 4'd4,
        NPU_ERR_RANGE             = 4'd5,
        NPU_ERR_ODD_DMA_BEAT      = 4'd6,
        NPU_ERR_BAD_KEEP          = 4'd7,
        NPU_ERR_FIFO              = 4'd8,
        NPU_ERR_ILLEGAL_FLAGS     = 4'd9
    } npu_error_e;

    typedef enum logic [1:0] {
        NPU_W_BANK0 = 2'd0,
        NPU_W_BANK1 = 2'd1
    } npu_w_bank_e;

    typedef enum logic [1:0] {
        NPU_MVOUT_FP32_Q8_24          = 2'd0,
        NPU_MVOUT_RAW_I32             = 2'd1,
        NPU_MVOUT_RESERVED2           = 2'd2,
        NPU_MVOUT_ATTENTION_QK_LOGP   = 2'd3
    } npu_mvout_mode_e;

    typedef struct packed {
        logic [31:0] acc0_i32;
        logic [31:0] acc1_i32;
    } npu_pe_int8_pair_t;

    typedef struct packed {
        npu_mode_e   mode;
        logic [3:0]  row_id;
        logic [2:0]  group_id;
        logic [255:0] data;
    } npu_sa_drain_beat_t;

    function automatic logic [31:0] lane_mask_to_byte_strobe(input logic [7:0] lane_mask);
        logic [31:0] strobe;
        begin
            strobe = '0;
            for (int lane = 0; lane < NPU_ACC_LANES_I32; lane++) begin
                strobe[lane*4 +: 4] = {4{lane_mask[lane]}};
            end
            return strobe;
        end
    endfunction

    function automatic logic mode_supported(input npu_mode_e mode);
        return (mode == NPU_MODE_INT8) || (mode == NPU_MODE_PV_LOG8_U16I8);
    endfunction
endpackage

`default_nettype wire
