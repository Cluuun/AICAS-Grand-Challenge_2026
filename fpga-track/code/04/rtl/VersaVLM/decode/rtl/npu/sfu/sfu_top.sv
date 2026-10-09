`ifndef SFU_TOP_SV
`define SFU_TOP_SV

module sfu_top #(
    parameter int USER_WIDTH = 16
) (
    input  logic                  clk,
    input  logic                  rst_n,
    input  logic                  cfg_silu_en,

    input  logic                  valid_i,
    input  logic [15:0]           data_i,
    input  logic [USER_WIDTH-1:0] user_i,

    output logic                  valid_o,
    output logic [15:0]           data_o,
    output logic [USER_WIDTH-1:0] user_o
);

    silu_top #(
        .USER_WIDTH (USER_WIDTH)
    ) u_silu (
        .clk      (clk),
        .rst_n    (rst_n),
        .enable_i (cfg_silu_en),
        .valid_i  (valid_i),
        .fp16_i   (data_i),
        .user_i   (user_i),
        .valid_o  (valid_o),
        .fp16_o   (data_o),
        .user_o   (user_o)
    );

endmodule

`endif
