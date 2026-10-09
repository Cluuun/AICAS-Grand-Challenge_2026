`default_nettype none

// Phase-1 INT8 packed PE.
//
// One physical PE consumes a packed A pair and one shared W lane, then updates
// two independent DSP-backed INT32 accumulators. BFP16-M and other non-INT8
// modes are explicit unsupported-mode errors in Phase 1.
module pe_int8_packed_b #(
    parameter int MODE_W      = 2,
    parameter int MODE_INT8   = 0,
    parameter int INT8_ACC_W  = 32,
    parameter int PE_MAC_LAT  = 3,
    parameter int K_BLOCK_MAX = 4096
) (
    input  wire logic                         clk_i,
    input  wire logic                         rst_i,

    input  wire logic                         step_fire_i,
    input  wire logic [MODE_W-1:0]            mode_i,

    input  wire logic                         a_valid_i,
    input  wire logic [15:0]                  a_data_i,
    output logic                         a_valid_o,
    output logic [15:0]                  a_data_o,

    input  wire logic                         w_valid_i,
    input  wire logic [15:0]                  w_data_i,
    output logic                         w_valid_o,
    output logic [15:0]                  w_data_o,

    input  wire logic                         clear_acc_i,
    input  wire logic                         snapshot_acc_i,
    input  wire logic                         drain_en_i,
    output logic                         drain_valid_o,
    output logic [63:0]                  drain_data_o,

    output logic                         unsupported_mode_error_o
);

    localparam logic [MODE_W-1:0] MODE_INT8_VALUE = MODE_INT8;

    logic int8_mode;
    logic mac_req_valid;
    logic signed [7:0] a_even_s;
    logic signed [7:0] a_odd_s;
    logic signed [7:0] w_s;
    logic signed [INT8_ACC_W-1:0] acc_even_dsp;
    logic signed [INT8_ACC_W-1:0] acc_odd_dsp;
    logic signed [INT8_ACC_W-1:0] shadow_even_q;
    logic signed [INT8_ACC_W-1:0] shadow_odd_q;
    logic signed [31:0] acc_even_i32;
    logic signed [31:0] acc_odd_i32;

    assign int8_mode = (mode_i == MODE_INT8_VALUE);
    assign mac_req_valid = step_fire_i & a_valid_i & w_valid_i & int8_mode;
    assign a_even_s = a_data_i[7:0];
    assign a_odd_s  = a_data_i[15:8];
    assign w_s      = w_data_i[7:0];
    assign unsupported_mode_error_o = (!int8_mode) &
        (step_fire_i | clear_acc_i | snapshot_acc_i | drain_en_i);

    initial begin
        if ((INT8_ACC_W < 18) || (INT8_ACC_W > 32)) begin
            $error("pe_int8_packed_b: INT8_ACC_W must be in the supported 18..32 bit range");
        end
        if (K_BLOCK_MAX > 4096) begin
            $error("pe_int8_packed_b: K_BLOCK_MAX above 4096 requires accumulator and timing review");
        end
    end

    int8_dsp_mac_lane #(
        .PIPE_STAGES ( PE_MAC_LAT ),
        .ACC_W       ( INT8_ACC_W )
    ) u_mac_even (
        .clk_i   ( clk_i         ),
        .rst_i   ( rst_i         ),
        .valid_i ( mac_req_valid ),
        .a_i     ( a_even_s      ),
        .w_i     ( w_s           ),
        .clear_i ( clear_acc_i   ),
        .acc_o   ( acc_even_dsp  )
    );

    int8_dsp_mac_lane #(
        .PIPE_STAGES ( PE_MAC_LAT ),
        .ACC_W       ( INT8_ACC_W )
    ) u_mac_odd (
        .clk_i   ( clk_i         ),
        .rst_i   ( rst_i         ),
        .valid_i ( mac_req_valid ),
        .a_i     ( a_odd_s       ),
        .w_i     ( w_s           ),
        .clear_i ( clear_acc_i   ),
        .acc_o   ( acc_odd_dsp   )
    );

    always_ff @(posedge clk_i) begin
        if (rst_i) begin
            a_valid_o <= 1'b0;
            w_valid_o <= 1'b0;
        end else if (step_fire_i) begin
            a_valid_o <= a_valid_i & int8_mode;
            w_valid_o <= w_valid_i & int8_mode;
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
            shadow_even_q <= acc_even_dsp;
            shadow_odd_q  <= acc_odd_dsp;
        end
    end

    assign acc_even_i32 = shadow_even_q;
    assign acc_odd_i32  = shadow_odd_q;

    assign drain_valid_o = drain_en_i & int8_mode;
    assign drain_data_o  = {acc_odd_i32, acc_even_i32};

endmodule

`default_nettype wire
