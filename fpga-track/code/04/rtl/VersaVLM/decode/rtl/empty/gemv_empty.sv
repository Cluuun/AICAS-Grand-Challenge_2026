module gemv_empty #(
    parameter int RF_DATA_WIDTH  = 64,
    parameter int SPM_SIZE       = 1 << 19,
    parameter int SPM_DATA_WIDTH = 512,
    parameter int GEMV_ACC_NUM   = 4,
    parameter int TILE_ELEMS     = 128,
    parameter int ROW_TILE_ELEMS = 32,
    parameter int FIXED_FRAC     = 12,

    localparam int SPM_ADDR_WIDTH = $clog2(SPM_SIZE)
) (
    input  logic                              clk,
    input  logic                              rst_n,

    input  logic [RF_DATA_WIDTH/2-1:0]        matvec_input_mat_addr,
    input  logic [RF_DATA_WIDTH/2-1:0]        matvec_input_vec_addr,
    input  logic [RF_DATA_WIDTH/2-1:0]        matvec_input_act_scale_addr,
    input  logic [RF_DATA_WIDTH/2-1:0]        matvec_input_act_group_stride_bytes,
    input  logic [RF_DATA_WIDTH/4-1:0]        matvec_input_mat_width,
    input  logic [RF_DATA_WIDTH/4-1:0]        matvec_input_mat_height,
    input  logic [RF_DATA_WIDTH/2-1:0]        matvec_output_addr,
    input  logic [RF_DATA_WIDTH/4-1:0]        matvec_input_scale_addr,
    input  logic [1:0]                        gemv_mode,
    input  logic [63:0]                       cfg_decode_flow,
    input  logic                              matvec_req_en,
    output logic                              matvec_busy,
    output logic                              matvec_comp_done,

    output logic                              spm_rd_en,
    output logic [SPM_ADDR_WIDTH-1:0]         spm_rd_addr,
    input  logic [SPM_DATA_WIDTH-1:0]         spm_rd_data,

    output logic                              scale_rd_en,
    output logic [SPM_ADDR_WIDTH-1:0]         scale_rd_addr,
    input  logic [SPM_DATA_WIDTH-1:0]         scale_rd_data,

    output logic                              act_rd_en,
    output logic [SPM_ADDR_WIDTH-1:0]         act_rd_addr,
    input  logic [SPM_DATA_WIDTH-1:0]         act_rd_data,

    output logic                              spm_wr_en,
    output logic [SPM_ADDR_WIDTH-1:0]         spm_wr_addr,
    output logic [SPM_DATA_WIDTH-1:0]         spm_wr_data,
    output logic [SPM_DATA_WIDTH/8-1:0]       spm_wr_mask,

    output logic                              act_wr_en,
    output logic [SPM_ADDR_WIDTH-1:0]         act_wr_addr,
    output logic [SPM_DATA_WIDTH-1:0]         act_wr_data,
    output logic [SPM_DATA_WIDTH/8-1:0]       act_wr_mask,

    input  logic                              preload_acc_en_i,
    input  logic [((GEMV_ACC_NUM <= 1) ? 1 : $clog2(GEMV_ACC_NUM))-1:0] preload_acc_id_i,
    input  logic [((ROW_TILE_ELEMS <= 1) ? 1 : $clog2(ROW_TILE_ELEMS))-1:0] preload_acc_row_i,
    input  logic [31:0]                       preload_acc_data_i
);

    assign matvec_busy      = 1'b0;
    assign matvec_comp_done = 1'b0;
    assign spm_rd_en        = 1'b0;
    assign spm_rd_addr      = '0;
    assign scale_rd_en      = 1'b0;
    assign scale_rd_addr    = '0;
    assign act_rd_en        = 1'b0;
    assign act_rd_addr      = '0;
    assign spm_wr_en        = 1'b0;
    assign spm_wr_addr      = '0;
    assign spm_wr_data      = '0;
    assign spm_wr_mask      = '0;
    assign act_wr_en        = 1'b0;
    assign act_wr_addr      = '0;
    assign act_wr_data      = '0;
    assign act_wr_mask      = '0;

endmodule
