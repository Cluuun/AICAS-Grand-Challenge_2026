module im2col_top
  import npu_config_pkg::*;
#(
    parameter int RF_DATA_WIDTH      = npu_config_pkg::RF_DATA_WIDTH,
    parameter int PE_DATA_WIDTH_IN   = npu_config_pkg::PE_DATA_WIDTH_IN,
    parameter int INPUT_WIDTH_MAX    = npu_config_pkg::INPUT_WIDTH_MAX,
    parameter int ARRAY_HEIGHT       = npu_config_pkg::ARRAY_HEIGHT,
    parameter int SPM_SIZE           = npu_config_pkg::SPM_SIZE,
    parameter int SPM_ADDR_WIDTH     = $clog2(SPM_SIZE)
)
(
    input  logic                                       clk,
    input  logic                                       rst_n,

    //----------------------------------
    // Control
    //----------------------------------
    input  logic                                       cfg_compute_dataflow,        // 0 for weight stationary, 1 for output stationary
    input  logic [1:0]                                 cfg_compute_padding_left,    // Number of left padding data columns for the input
    input  logic [1:0]                                 cfg_compute_padding_right,   // Number of right padding data columns for the input
    input  logic [1:0]                                 cfg_compute_padding_top,     // Number of top padding data columns for the input
    input  logic [1:0]                                 cfg_compute_padding_bottom,  // Number of bottom padding data columns for the input
    input  logic [1:0]                                 cfg_compute_padding_mode,    // Only zero padding supported currently
    input  logic [3:0]                                 cfg_compute_weight_shape_m1,    // 0 => 1
    input  logic [1:0]                                 cfg_compute_weight_stride_m1,   // 0 => 1
    input  logic [4:0]                                 cfg_compute_weight_dilation_m1, // 0 => 1
    input  logic                                       cfg_compute_is_groupconv,
    input  logic [1:0]                                 cfg_compute_int_type,
    input  logic [1:0]                                 cfg_compute_optype,

    input  logic [RF_DATA_WIDTH/2-1:0]                 sa_input_ifm_spm_addr,          
    input  logic [$clog2(INPUT_WIDTH_MAX)-1:0]         sa_input_ifm_col_num_m1,           
    input  logic [$clog2(ARRAY_HEIGHT)-1:0]            sa_input_ifm_row_num_m1,           
    input  logic [RF_DATA_WIDTH/4-1:0]                 sa_input_ifm_stride,           
    input  logic [$clog2(ARRAY_HEIGHT)-1:0]            sa_input_ifm_channel_m1,      // can be less than 32-1

    input  logic [RF_DATA_WIDTH/8-1:0]                 sa_input_ofm_col_num_m1,
    input  logic [RF_DATA_WIDTH/8-1:0]                 sa_input_ofm_row_num_m1,


    //----------------------------------
    // Scratchpad
    //----------------------------------
    output logic                                       sa_spm_rd_en,
    output logic [SPM_ADDR_WIDTH-1:0]                  sa_spm_rd_addr,
    input  logic [ARRAY_HEIGHT-1:0][PE_DATA_WIDTH_IN-1:0] sa_spm_rd_data_in,

    //----------------------------------
    // Interface to systolic array
    //----------------------------------
    input  logic                                       im2col_sa_ready,
    output logic                                       im2col_sa_valid,
    output logic [ARRAY_HEIGHT-1:0][PE_DATA_WIDTH_IN-1:0]  im2col_sa_data,

    //----------------------------------
    // Command handshake
    //----------------------------------
    input  logic                                       im2col_req_en,
    output logic                                       im2col_busy,
    output logic                                       im2col_done
);

logic fifo_wr_en;
logic [PE_DATA_WIDTH_IN * ARRAY_HEIGHT -1:0] fifo_data_in;
logic [PE_DATA_WIDTH_IN * ARRAY_HEIGHT -1:0] fifo_data_out;
logic fifo_full;
logic fifo_empty;

logic [RF_DATA_WIDTH/8:0] ofm_x;
logic [RF_DATA_WIDTH/8:0] ofm_y;
logic [RF_DATA_WIDTH/8:0] ofm_x_send;
logic [RF_DATA_WIDTH/8:0] ofm_y_send;
logic [4:0] kernel_kx;
logic [4:0] kernel_kx_send;
logic [4:0] kernel_ky;
logic [4:0] kernel_ky_send;

typedef enum logic [1:0] {
    ZERO_PADDING
} padding_t;

padding_t padding_mode;

assign padding_mode = padding_t'(cfg_compute_padding_mode);

logic [$clog2(ARRAY_HEIGHT)-1:0] lane_idx;

logic [4:0] cfg_compute_weight_shape;
logic [2:0] cfg_compute_weight_stride;
logic [5:0] cfg_compute_weight_dilation;

logic [$clog2(INPUT_WIDTH_MAX):0]          sa_input_ifm_col_num;
logic [$clog2(ARRAY_HEIGHT):0]             sa_input_ifm_row_num;
logic [RF_DATA_WIDTH/4:0]                  sa_input_ifm_stride_r;
logic [$clog2(ARRAY_HEIGHT):0]             sa_input_ifm_channel;
logic [RF_DATA_WIDTH/8:0]                  sa_input_ofm_col_num;
logic [RF_DATA_WIDTH/8:0]                  sa_input_ofm_row_num;
logic [1:0]                                cfg_compute_padding_left_r;
logic [1:0]                                cfg_compute_padding_top_r;

always_ff @(posedge clk or negedge rst_n) begin
    if (~rst_n) begin
        cfg_compute_weight_shape    <= '0;
        cfg_compute_weight_stride   <= '0;
        cfg_compute_weight_dilation <= '0;
        sa_input_ifm_col_num        <= '0;
        sa_input_ifm_row_num        <= '0;
        sa_input_ifm_stride_r       <= '0;
        sa_input_ifm_channel        <= '0;
        sa_input_ofm_col_num        <= '0;
        sa_input_ofm_row_num        <= '0;
        cfg_compute_padding_left_r  <= '0;
        cfg_compute_padding_top_r   <= '0;
    end else if (im2col_req_en) begin
        cfg_compute_weight_shape    <= cfg_compute_weight_shape_m1 + 1'b1;
        cfg_compute_weight_stride   <= cfg_compute_weight_stride_m1 + 1'b1;
        cfg_compute_weight_dilation <= cfg_compute_weight_dilation_m1 + 1'b1;
        sa_input_ifm_col_num        <= sa_input_ifm_col_num_m1 + 1'b1;
        sa_input_ifm_row_num        <= sa_input_ifm_row_num_m1 + 1'b1;
        sa_input_ifm_stride_r       <= sa_input_ifm_stride;
        sa_input_ifm_channel        <= sa_input_ifm_channel_m1 + 1'b1;
        sa_input_ofm_col_num        <= sa_input_ofm_col_num_m1 + 1'b1;
        sa_input_ofm_row_num        <= sa_input_ofm_row_num_m1 + 1'b1;
        cfg_compute_padding_left_r  <= cfg_compute_padding_left;
        cfg_compute_padding_top_r   <= cfg_compute_padding_top;
    end else begin
        cfg_compute_weight_shape    <= cfg_compute_weight_shape;
        cfg_compute_weight_stride   <= cfg_compute_weight_stride;
        cfg_compute_weight_dilation <= cfg_compute_weight_dilation;
        sa_input_ifm_col_num        <= sa_input_ifm_col_num;
        sa_input_ifm_row_num        <= sa_input_ifm_row_num;
        sa_input_ifm_stride_r       <= sa_input_ifm_stride_r;
        sa_input_ifm_channel        <= sa_input_ifm_channel;
        sa_input_ofm_col_num        <= sa_input_ofm_col_num;
        sa_input_ofm_row_num        <= sa_input_ofm_row_num;
        cfg_compute_padding_left_r  <= cfg_compute_padding_left_r;
        cfg_compute_padding_top_r   <= cfg_compute_padding_top_r;
    end
end

// FSM
typedef enum logic [1:0] { 
    IDLE,
    BUSY,
    DONE
} state_t;

state_t state, state_next;
logic finish;
logic hold_padding_lane;
logic will_finish;

assign hold_padding_lane = (state == BUSY) &&
                           (ofm_x == '0) &&
                           (ofm_y == '0) &&
                           (lane_idx != '0);
assign will_finish = (state == BUSY) &&
                     ~fifo_full &&
                     ~hold_padding_lane &&
                     (ofm_x >= sa_input_ofm_col_num - 1) &&
                     (ofm_y >= sa_input_ofm_row_num - 1) &&
                     (kernel_kx >= cfg_compute_weight_shape - 1) &&
                     (kernel_ky >= cfg_compute_weight_shape - 1);

always_ff @(posedge clk or negedge rst_n) begin
    if (~rst_n) begin
        state <= IDLE;
    end else begin
        state <= state_next;
    end
end

always_comb begin
    state_next = state;
    case (state)
        IDLE: begin
            if (im2col_req_en) begin
                state_next = BUSY;
            end
        end
        BUSY: begin
            // `finish` covers the normal path. `will_finish` handles the case
            // where `finish` pulses at lane_idx==ARRAY_HEIGHT-1 and would be
            // missed by next-state logic. Do not use `will_finish` in padding
            // hold cycles (e.g. OFM=1x1), otherwise we can end one block early.
            if ((lane_idx == ARRAY_HEIGHT - 1) && (finish || will_finish)) begin
                state_next = DONE;
            end
        end
        DONE: begin
            if (fifo_empty) begin
                state_next = IDLE;
            end
        end
        default: begin
            state_next = IDLE;
        end
    endcase
end

always_ff @(posedge clk or negedge rst_n) begin
    if (~rst_n) begin
        im2col_busy <= 1'b0;
        im2col_done <= 1'b0;
    end else begin
        im2col_busy <= (state == BUSY);
        im2col_done <= (state == DONE);
    end
end

// Pipeline Stage 1

logic pipe1_valid;
logic pipe1_zero;

always_ff @(posedge clk or negedge rst_n) begin
    if (~rst_n) begin
        ofm_x <= '0;
        ofm_x_send <= '0;
        ofm_y <= '0;
        ofm_y_send <= '0;
        kernel_kx <= '0;
        kernel_kx_send <= '0;
        kernel_ky <= '0;
        kernel_ky_send <= '0;
        pipe1_valid <= 1'b0;
        lane_idx <= '0;
        pipe1_zero <= 1'b0;
        finish <= 1'b0;
    end else begin
        case (state)
            IDLE, DONE: begin
                ofm_x <= '0;
                ofm_y <= '0;
                ofm_x_send <= '0;
                ofm_y_send <= '0;
                kernel_kx <= '0;
                kernel_kx_send <= '0;
                kernel_ky <= '0;
                kernel_ky_send <= '0;
                pipe1_valid <= 1'b0;
                lane_idx <= '0;
                pipe1_zero <= 1'b0;
                finish <= 1'b0;
            end
            BUSY: if (~fifo_full) begin
                if (lane_idx == ARRAY_HEIGHT - 1) begin
                    lane_idx <= '0;
                end else begin
                    lane_idx <= lane_idx + 1;
                end
            if ((ofm_x == '0) && (ofm_y == '0) && (lane_idx != '0)) begin
                pipe1_valid <= 1'b1;
                ofm_x <= ofm_x;
                ofm_y <= ofm_y;
                ofm_x_send <= ofm_x_send;
                ofm_y_send <= ofm_y_send;
                kernel_kx <= kernel_kx;
                kernel_kx_send <= kernel_kx_send;
                kernel_ky <= kernel_ky;
                kernel_ky_send <= kernel_ky_send;
                pipe1_zero <= 1'b1;
                finish <= finish;
            end else if ((ofm_x >= sa_input_ofm_col_num - 1) && (ofm_y >= sa_input_ofm_row_num - 1)) begin
                    ofm_x <= '0;
                    ofm_y <= '0;
                    ofm_x_send <= ofm_x;
                    ofm_y_send <= ofm_y;
                    pipe1_valid <= 1'b1;
                    pipe1_zero <= 1'b0;
                    if (kernel_kx >= cfg_compute_weight_shape - 1) begin
                        kernel_kx <= '0;
                        if (kernel_ky >= cfg_compute_weight_shape - 1) begin
                            kernel_ky <= '0;
                            finish <= 1'b1;
                        end else begin
                            kernel_ky <= kernel_ky + 1;
                            finish <= 1'b0;
                        end
                    end else begin
                        kernel_kx <= kernel_kx + 1;
                        finish <= 1'b0;
                    end
                    kernel_kx_send <= kernel_kx;
                    kernel_ky_send <= kernel_ky;
                end else begin
                    if (ofm_x >= sa_input_ofm_col_num - 1) begin
                        ofm_x <= '0;
                        ofm_y <= ofm_y + 1;
                        ofm_x_send <= ofm_x;
                        ofm_y_send <= ofm_y;
                        pipe1_valid <= 1'b1;
                        pipe1_zero <= 1'b0;
                        kernel_kx <= kernel_kx;
                        kernel_kx_send <= kernel_kx;
                        kernel_ky <= kernel_ky;
                        kernel_ky_send <= kernel_ky;
                        finish <= 1'b0;
                    end else begin
                        ofm_x <= ofm_x + 1;
                        ofm_y <= ofm_y;
                        ofm_x_send <= ofm_x;
                        ofm_y_send <= ofm_y;
                        pipe1_valid <= 1'b1;
                        pipe1_zero <= 1'b0;
                        kernel_kx <= kernel_kx;
                        kernel_kx_send <= kernel_kx;
                        kernel_ky <= kernel_ky;
                        kernel_ky_send <= kernel_ky;
                        finish <= 1'b0;
                    end
                end
            end else begin
                ofm_x <= ofm_x;
                ofm_y <= ofm_y;
                ofm_x_send <= ofm_x_send;
                ofm_y_send <= ofm_y_send;
                pipe1_valid <= 1'b0;
                kernel_kx <= kernel_kx;
                kernel_kx_send <= kernel_kx_send;
                kernel_ky <= kernel_ky;
                kernel_ky_send <= kernel_ky_send;
                lane_idx <= lane_idx;
                pipe1_zero <= pipe1_zero;
                finish <= finish;
            end
        endcase
    end
end

logic signed [RF_DATA_WIDTH/8+1:0] ifm_y_calc;
logic signed [RF_DATA_WIDTH/8+1:0] ifm_x_calc;
logic signed [RF_DATA_WIDTH/8:0] addr_iy;
logic signed [RF_DATA_WIDTH/8:0] addr_ix;
logic addr_is_padding;

always_comb begin
    // ifm_y = - pad_top + ofm_y * stride_y + kernel_y * dilation_y
    ifm_y_calc = $signed({1'b0, ofm_y_send}) * $signed({1'b0, cfg_compute_weight_stride}) 
                  + $signed({1'b0, kernel_ky_send}) * $signed({1'b0, cfg_compute_weight_dilation})
                  - $signed({1'b0, cfg_compute_padding_top_r});
    // ifm_x = - pad_left + ofm_x * stride_x + kernel_x * dilation_x
    ifm_x_calc = $signed({1'b0, ofm_x_send}) * $signed({1'b0, cfg_compute_weight_stride}) 
                  + $signed({1'b0, kernel_kx_send}) * $signed({1'b0, cfg_compute_weight_dilation})
                  - $signed({1'b0, cfg_compute_padding_left_r});

    addr_iy = ifm_y_calc[RF_DATA_WIDTH/8:0];
    addr_ix = ifm_x_calc[RF_DATA_WIDTH/8:0];

    addr_is_padding = (ifm_y_calc < 0) || (ifm_y_calc >= $signed({1'b0, sa_input_ifm_row_num})) ||
                      (ifm_x_calc < 0) || (ifm_x_calc >= $signed({1'b0, sa_input_ifm_col_num}));
end

// Pipeline Stage 2
logic pipe2_valid;
logic pipe2_addr_is_padding;
logic pipe2_zero;

always_ff @(posedge clk or negedge rst_n) begin
    if (~rst_n) begin
        pipe2_valid <= 1'b0;
        sa_spm_rd_en <= 1'b0;
        sa_spm_rd_addr <= '0;
        pipe2_addr_is_padding <= 1'b0;
        pipe2_zero <= 1'b0;
    end else begin
        pipe2_valid <= pipe1_valid;
        sa_spm_rd_addr <= sa_input_ifm_spm_addr + (addr_iy * sa_input_ifm_stride_r + addr_ix) * (sa_input_ifm_channel * PE_DATA_WIDTH / 8);
        pipe2_addr_is_padding <= addr_is_padding;
        pipe2_zero <= pipe1_zero;
        case (padding_mode)
            ZERO_PADDING: begin
                sa_spm_rd_en <= (addr_is_padding | pipe1_zero) ? 1'b0 : pipe1_valid;
            end
            default: begin
                sa_spm_rd_en <= pipe1_zero ? 1'b0 : pipe1_valid;
            end
        endcase
    end
end

// Pipeline Stage 3
logic pipe3_addr_is_padding;
logic pipe3_zero;
always_ff @(posedge clk or negedge rst_n) begin
    if (~rst_n) begin
        fifo_wr_en <= 1'b0;
        pipe3_addr_is_padding <= 1'b0;
        pipe3_zero <= 1'b0;
    end else begin
        fifo_wr_en <= pipe2_valid;
        pipe3_addr_is_padding <= pipe2_addr_is_padding;
        pipe3_zero <= pipe2_zero;
    end
end

always_comb begin
    case (padding_mode)
        ZERO_PADDING: begin
            if (pipe3_addr_is_padding |  pipe3_zero) begin
                fifo_data_in = '0;
            end else begin
                for (int i = 0; i < ARRAY_HEIGHT; i++) begin
                    if (i < sa_input_ifm_channel) begin
                        fifo_data_in[i*PE_DATA_WIDTH_IN +: PE_DATA_WIDTH_IN] = sa_spm_rd_data_in[i];
                    end else begin
                        fifo_data_in[i*PE_DATA_WIDTH_IN +: PE_DATA_WIDTH_IN] = '0;
                    end
                end
            end
        end
        default: begin
            if (pipe3_zero) begin
                fifo_data_in = '0;
            end else begin
                for (int i = 0; i < ARRAY_HEIGHT; i++) begin
                    if (i < sa_input_ifm_channel) begin
                        fifo_data_in[i*PE_DATA_WIDTH_IN +: PE_DATA_WIDTH_IN] = sa_spm_rd_data_in[i];
                    end else begin
                        fifo_data_in[i*PE_DATA_WIDTH_IN +: PE_DATA_WIDTH_IN] = '0;
                    end
                end
            end
        end
    endcase
end

// FIFO
FIFO_DP_GT_1 #(
    .DW(PE_DATA_WIDTH_IN * ARRAY_HEIGHT),
    .DP(4)
) u_im2col_fifo (
    .clk(clk),
    .rst_n(rst_n),

    .fifo_wen(fifo_wr_en),
    .fifo_wdat(fifo_data_in),
    .fifo_full(fifo_full),

    .fifo_ren(im2col_sa_ready),
    .fifo_rdat(fifo_data_out),
    .fifo_empty(fifo_empty)
);

assign im2col_sa_valid = ~fifo_empty;
assign {>>{im2col_sa_data}} = fifo_data_out;
endmodule
