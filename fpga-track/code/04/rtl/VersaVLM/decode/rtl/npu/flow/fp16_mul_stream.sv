`ifndef FP16_MUL_STREAM_SV
`define FP16_MUL_STREAM_SV

module fp16_mul_stream #(
    parameter int USER_WIDTH = 16
) (
    input  logic                  clk,
    input  logic                  rst_n,
    input  logic                  enable_i,

    input  logic                  src0_valid_i,
    input  logic [15:0]           src0_data_i,
    input  logic [USER_WIDTH-1:0] src0_user_i,
    output logic                  src0_pop_o,

    input  logic                  src1_valid_i,
    input  logic [15:0]           src1_data_i,
    output logic                  src1_pop_o,

    output logic                  valid_o,
    output logic [15:0]           data_o,
    output logic [USER_WIDTH-1:0] user_o,
    output logic                  pair_miss_o,
    output logic [USER_WIDTH:0]   pair_count_o,
    output logic [USER_WIDTH:0]   output_count_o
);

    logic pair_fire_w;
    logic mul_valid_w;
    logic [15:0] mul_data_w;
    localparam int MUL_LATENCY = 6;

    logic [USER_WIDTH-1:0] user_pipe_q [0:MUL_LATENCY];
    logic pair_miss_q;
    logic [USER_WIDTH:0] pair_count_q;
    logic [USER_WIDTH:0] output_count_q;

    assign pair_fire_w = enable_i && src0_valid_i && src1_valid_i;
    assign src0_pop_o  = pair_fire_w;
    assign src1_pop_o  = pair_fire_w;
    assign pair_miss_o = pair_miss_q;
    assign pair_count_o = pair_count_q;
    assign output_count_o = output_count_q;

    gemv_fp16_mul_pipe2 #(
        .SUPPORT_INPUT_SUBNORMAL (1'b0),
        .ENABLE_FP16_OUT (1'b1),
        .ENABLE_UQ0P24_OUT (1'b0)
    ) u_mul (
        .clk_i    (clk),
        .rst_ni   (rst_n),
        .valid_i  (pair_fire_w),
        .mul_en_i (1'b1),
        .preserve_subnormal_i (1'b0),
        .a_i      (src0_data_i),
        .b_i      (src1_data_i),
        .z_o      (mul_data_w),
        .uq0p24_o (),
        .valid_o  (mul_valid_w)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i <= MUL_LATENCY; i++) begin
                user_pipe_q[i] <= '0;
            end
            pair_miss_q    <= 1'b0;
            pair_count_q   <= '0;
            output_count_q <= '0;
            valid_o        <= 1'b0;
            data_o         <= '0;
            user_o         <= '0;
        end else begin
            user_pipe_q[0] <= pair_fire_w ? src0_user_i : '0;
            for (int i = 1; i <= MUL_LATENCY; i++) begin
                user_pipe_q[i] <= user_pipe_q[i-1];
            end

            pair_miss_q <= enable_i && src0_valid_i && !src1_valid_i;
            if (!enable_i) begin
                pair_count_q   <= '0;
                output_count_q <= '0;
            end else begin
                if (pair_fire_w)
                    pair_count_q <= pair_count_q + {{USER_WIDTH{1'b0}}, 1'b1};
                if (mul_valid_w)
                    output_count_q <= output_count_q + {{USER_WIDTH{1'b0}}, 1'b1};
            end

            valid_o <= enable_i && mul_valid_w;
            data_o  <= mul_data_w;
            user_o  <= user_pipe_q[MUL_LATENCY];
        end
    end

`ifndef SYNTHESIS
    property p_no_src0_without_src1;
        @(posedge clk) disable iff (!rst_n) enable_i && src0_valid_i |-> src1_valid_i;
    endproperty
    assert property (p_no_src0_without_src1);
`endif

endmodule

`endif
