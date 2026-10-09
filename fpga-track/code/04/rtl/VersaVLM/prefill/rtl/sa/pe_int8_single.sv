`default_nettype none

// Phase-1 direct PE for INT8 and internal PV log8P(U15)xINT8.
//
// One physical PE maps to one logical output element. The PE consumes one A
// 16-bit lane and one W byte per active step and snapshots one INT32 accumulator.
module pe_int8_single #(
    parameter int INT8_ACC_W  = 32,
    parameter int PE_MAC_LAT  = 3,
    parameter int K_BLOCK_MAX = 4096
) (
    input  wire logic                         clk_i,
    input  wire logic                         rst_i,

    input  wire logic                         step_fire_i,
    input  wire logic                         mac_mode_i,

    input  wire logic                         a_valid_i,
    input  wire logic [15:0]                  a_data_i,
    output logic                              a_valid_o,
    output logic [15:0]                       a_data_o,

    input  wire logic                         w_valid_i,
    input  wire logic [7:0]                   w_data_i,
    output logic                              w_valid_o,
    output logic [7:0]                        w_data_o,

    input  wire logic                         clear_acc_i,
    input  wire logic                         snapshot_acc_i,
    input  wire logic                         drain_en_i,
    output logic                              drain_valid_o,
    output logic [31:0]                       drain_data_o
);

    logic mac_req_valid;
    logic signed [15:0] a_s;
    logic signed [7:0] w_s;
    logic signed [INT8_ACC_W-1:0] acc_dsp;
    logic signed [INT8_ACC_W-1:0] shadow_q;
    logic signed [31:0] acc_i32;

    assign mac_req_valid = step_fire_i & a_valid_i & w_valid_i & mac_mode_i;
    assign a_s = a_data_i;
    assign w_s = w_data_i;

    initial begin
        if ((INT8_ACC_W < 18) || (INT8_ACC_W > 32)) begin
            $error("pe_int8_single: INT8_ACC_W must be in the supported 18..32 bit range");
        end
        if (K_BLOCK_MAX > 4096) begin
            $error("pe_int8_single: K_BLOCK_MAX above 4096 requires accumulator and timing review");
        end
    end

    int8_dsp_mac_lane #(
        .PIPE_STAGES ( PE_MAC_LAT ),
        .A_W         ( 16         ),
        .W_W         ( 8          ),
        .ACC_W       ( INT8_ACC_W )
    ) u_mac (
        .clk_i   ( clk_i         ),
        .rst_i   ( rst_i         ),
        .valid_i ( mac_req_valid ),
        .a_i     ( a_s           ),
        .w_i     ( w_s           ),
        .clear_i ( clear_acc_i   ),
        .acc_o   ( acc_dsp       )
    );

    always_ff @(posedge clk_i) begin
        if (rst_i) begin
            a_valid_o <= 1'b0;
            w_valid_o <= 1'b0;
        end else if (step_fire_i) begin
            a_valid_o <= a_valid_i & mac_mode_i;
            w_valid_o <= w_valid_i & mac_mode_i;
        end
    end

    always_ff @(posedge clk_i) begin
        if (step_fire_i) begin
            a_data_o <= a_data_i;
            w_data_o <= w_data_i;
        end
    end

    always_ff @(posedge clk_i) begin
        if (snapshot_acc_i) begin
            shadow_q <= acc_dsp;
        end
    end

    assign acc_i32 = shadow_q;
    assign drain_valid_o = drain_en_i & mac_mode_i;
    assign drain_data_o = acc_i32;

endmodule

`default_nettype wire
