`timescale 1 ns / 1 ps

module mmult_axi_profile_regs #
(
    parameter integer C_S_AXI_PROFILE_ADDR_WIDTH = 8,
    parameter integer C_S_AXI_PROFILE_DATA_WIDTH = 32
)
(
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME ap_clk, ASSOCIATED_BUSIF s_axi_profile, ASSOCIATED_RESET ap_rst_n, FREQ_HZ 250000000" *)
    input  wire                                      ap_clk,
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME ap_rst_n, POLARITY ACTIVE_LOW" *)
    input  wire                                      ap_rst_n,

    input  wire                                      ctrl_awvalid,
    input  wire                                      ctrl_awready,
    input  wire [7:0]                                ctrl_awaddr,
    input  wire                                      ctrl_wvalid,
    input  wire                                      ctrl_wready,
    input  wire [31:0]                               ctrl_wdata,

    input  wire                                      a_arvalid,
    input  wire                                      a_arready,
    input  wire [7:0]                                a_arlen,
    input  wire                                      a_rvalid,
    input  wire                                      a_rready,

    input  wire                                      b_arvalid,
    input  wire                                      b_arready,
    input  wire [7:0]                                b_arlen,
    input  wire                                      b_rvalid,
    input  wire                                      b_rready,

    input  wire                                      c_awvalid,
    input  wire                                      c_awready,
    input  wire [7:0]                                c_awlen,
    input  wire                                      c_wvalid,
    input  wire                                      c_wready,
    input  wire                                      c_bvalid,
    input  wire                                      c_bready,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi_profile AWADDR" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME s_axi_profile, PROTOCOL AXI4LITE, ADDR_WIDTH 8, DATA_WIDTH 32, FREQ_HZ 250000000, ID_WIDTH 0, AWUSER_WIDTH 0, ARUSER_WIDTH 0, WUSER_WIDTH 0, RUSER_WIDTH 0, BUSER_WIDTH 0, HAS_BURST 0, HAS_LOCK 0, HAS_PROT 0, HAS_CACHE 0, HAS_QOS 0, HAS_REGION 0, HAS_WSTRB 1, HAS_BRESP 1, HAS_RRESP 1, MAX_BURST_LENGTH 1, NUM_READ_OUTSTANDING 1, NUM_WRITE_OUTSTANDING 1, SUPPORTS_NARROW_BURST 0" *)
    input  wire [C_S_AXI_PROFILE_ADDR_WIDTH-1:0]     s_axi_profile_AWADDR,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi_profile AWVALID" *)
    input  wire                                      s_axi_profile_AWVALID,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi_profile AWREADY" *)
    output wire                                      s_axi_profile_AWREADY,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi_profile WDATA" *)
    input  wire [C_S_AXI_PROFILE_DATA_WIDTH-1:0]     s_axi_profile_WDATA,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi_profile WSTRB" *)
    input  wire [(C_S_AXI_PROFILE_DATA_WIDTH/8)-1:0] s_axi_profile_WSTRB,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi_profile WVALID" *)
    input  wire                                      s_axi_profile_WVALID,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi_profile WREADY" *)
    output wire                                      s_axi_profile_WREADY,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi_profile BRESP" *)
    output reg  [1:0]                                s_axi_profile_BRESP,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi_profile BVALID" *)
    output reg                                       s_axi_profile_BVALID,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi_profile BREADY" *)
    input  wire                                      s_axi_profile_BREADY,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi_profile ARADDR" *)
    input  wire [C_S_AXI_PROFILE_ADDR_WIDTH-1:0]     s_axi_profile_ARADDR,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi_profile ARVALID" *)
    input  wire                                      s_axi_profile_ARVALID,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi_profile ARREADY" *)
    output wire                                      s_axi_profile_ARREADY,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi_profile RDATA" *)
    output reg  [C_S_AXI_PROFILE_DATA_WIDTH-1:0]     s_axi_profile_RDATA,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi_profile RRESP" *)
    output reg  [1:0]                                s_axi_profile_RRESP,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi_profile RVALID" *)
    output reg                                       s_axi_profile_RVALID,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi_profile RREADY" *)
    input  wire                                      s_axi_profile_RREADY
);

localparam [31:0] PROFILE_MAGIC = 32'h50524631;
localparam [31:0] PROFILE_VERSION = 32'h00000001;

localparam [7:0] REG_CONTROL               = 8'h00;
localparam [7:0] REG_STATUS                = 8'h04;
localparam [7:0] REG_MAGIC                 = 8'h08;
localparam [7:0] REG_VERSION               = 8'h0c;
localparam [7:0] REG_LOAD_A_BUSY_LO        = 8'h10;
localparam [7:0] REG_LOAD_A_BUSY_HI        = 8'h14;
localparam [7:0] REG_LOAD_B_BUSY_LO        = 8'h18;
localparam [7:0] REG_LOAD_B_BUSY_HI        = 8'h1c;
localparam [7:0] REG_STORE_C_BUSY_LO       = 8'h20;
localparam [7:0] REG_STORE_C_BUSY_HI       = 8'h24;
localparam [7:0] REG_LOAD_A_READ_BEATS_LO  = 8'h28;
localparam [7:0] REG_LOAD_A_READ_BEATS_HI  = 8'h2c;
localparam [7:0] REG_LOAD_B_READ_BEATS_LO  = 8'h30;
localparam [7:0] REG_LOAD_B_READ_BEATS_HI  = 8'h34;
localparam [7:0] REG_STORE_C_WRITE_BEATS_LO = 8'h38;
localparam [7:0] REG_STORE_C_WRITE_BEATS_HI = 8'h3c;
localparam [7:0] REG_LOAD_A_AR_STALL_LO    = 8'h40;
localparam [7:0] REG_LOAD_A_AR_STALL_HI    = 8'h44;
localparam [7:0] REG_LOAD_B_AR_STALL_LO    = 8'h48;
localparam [7:0] REG_LOAD_B_AR_STALL_HI    = 8'h4c;
localparam [7:0] REG_STORE_C_AW_STALL_LO   = 8'h50;
localparam [7:0] REG_STORE_C_AW_STALL_HI   = 8'h54;
localparam [7:0] REG_STORE_C_W_STALL_LO    = 8'h58;
localparam [7:0] REG_STORE_C_W_STALL_HI    = 8'h5c;
localparam [7:0] REG_STORE_C_B_STALL_LO    = 8'h60;
localparam [7:0] REG_STORE_C_B_STALL_HI    = 8'h64;

reg        ctrl_aw_seen;
reg [7:0]  ctrl_awaddr_reg;
reg        ctrl_w_seen;
reg [31:0] ctrl_wdata_reg;

reg        capture_active;
reg        start_seen;

reg [15:0] a_outstanding_beats;
reg [15:0] b_outstanding_beats;
reg [15:0] c_outstanding_write_beats;
reg [7:0]  c_outstanding_write_responses;

reg [63:0] load_a_busy_cycles;
reg [63:0] load_b_busy_cycles;
reg [63:0] store_c_busy_cycles;
reg [63:0] load_a_read_beats;
reg [63:0] load_b_read_beats;
reg [63:0] store_c_write_beats;
reg [63:0] load_a_ar_stall_cycles;
reg [63:0] load_b_ar_stall_cycles;
reg [63:0] store_c_aw_stall_cycles;
reg [63:0] store_c_w_stall_cycles;
reg [63:0] store_c_b_stall_cycles;

reg                                       profile_aw_seen;
reg [C_S_AXI_PROFILE_ADDR_WIDTH-1:0]      profile_awaddr_reg;
reg                                       profile_w_seen;
reg [C_S_AXI_PROFILE_DATA_WIDTH-1:0]      profile_wdata_reg;
reg [(C_S_AXI_PROFILE_DATA_WIDTH/8)-1:0]  profile_wstrb_reg;

wire ctrl_aw_fire = ctrl_awvalid && ctrl_awready;
wire ctrl_w_fire = ctrl_wvalid && ctrl_wready;
wire ctrl_write_fire = (ctrl_aw_seen || ctrl_aw_fire) && (ctrl_w_seen || ctrl_w_fire);
wire [7:0] ctrl_write_addr = ctrl_aw_fire ? ctrl_awaddr : ctrl_awaddr_reg;
wire [31:0] ctrl_write_data = ctrl_w_fire ? ctrl_wdata : ctrl_wdata_reg;
wire start_pulse = ctrl_write_fire && (ctrl_write_addr == 8'h00) && ctrl_write_data[0];

wire profile_aw_fire = s_axi_profile_AWVALID && s_axi_profile_AWREADY;
wire profile_w_fire = s_axi_profile_WVALID && s_axi_profile_WREADY;
wire profile_write_fire = (profile_aw_seen || profile_aw_fire) && (profile_w_seen || profile_w_fire) && !s_axi_profile_BVALID;
wire [C_S_AXI_PROFILE_ADDR_WIDTH-1:0] profile_write_addr = profile_aw_fire ? s_axi_profile_AWADDR : profile_awaddr_reg;
wire [C_S_AXI_PROFILE_DATA_WIDTH-1:0] profile_write_data = profile_w_fire ? s_axi_profile_WDATA : profile_wdata_reg;
wire profile_sw_reset = profile_write_fire &&
                        (profile_write_addr[7:0] == REG_CONTROL) &&
                        profile_write_data[0];

wire a_ar_fire = a_arvalid && a_arready;
wire a_r_fire = a_rvalid && a_rready;
wire b_ar_fire = b_arvalid && b_arready;
wire b_r_fire = b_rvalid && b_rready;
wire c_aw_fire = c_awvalid && c_awready;
wire c_w_fire = c_wvalid && c_wready;
wire c_b_fire = c_bvalid && c_bready;

wire [15:0] a_ar_beats = {8'd0, a_arlen} + 16'd1;
wire [15:0] b_ar_beats = {8'd0, b_arlen} + 16'd1;
wire [15:0] c_aw_beats = {8'd0, c_awlen} + 16'd1;

wire a_busy = capture_active && (a_arvalid || a_rvalid || (a_outstanding_beats != 0));
wire b_busy = capture_active && (b_arvalid || b_rvalid || (b_outstanding_beats != 0));
wire c_busy = capture_active && (c_awvalid || c_wvalid || c_bvalid ||
                                 (c_outstanding_write_beats != 0) ||
                                 (c_outstanding_write_responses != 0));

assign s_axi_profile_AWREADY = !profile_aw_seen && !s_axi_profile_BVALID;
assign s_axi_profile_WREADY = !profile_w_seen && !s_axi_profile_BVALID;
assign s_axi_profile_ARREADY = !s_axi_profile_RVALID;

function [31:0] profile_reg_read;
    input [7:0] reg_addr;
    begin
        case (reg_addr)
            REG_CONTROL:                profile_reg_read = 32'h00000000;
            REG_STATUS:                 profile_reg_read = {30'd0, start_seen, capture_active};
            REG_MAGIC:                  profile_reg_read = PROFILE_MAGIC;
            REG_VERSION:                profile_reg_read = PROFILE_VERSION;
            REG_LOAD_A_BUSY_LO:         profile_reg_read = load_a_busy_cycles[31:0];
            REG_LOAD_A_BUSY_HI:         profile_reg_read = load_a_busy_cycles[63:32];
            REG_LOAD_B_BUSY_LO:         profile_reg_read = load_b_busy_cycles[31:0];
            REG_LOAD_B_BUSY_HI:         profile_reg_read = load_b_busy_cycles[63:32];
            REG_STORE_C_BUSY_LO:        profile_reg_read = store_c_busy_cycles[31:0];
            REG_STORE_C_BUSY_HI:        profile_reg_read = store_c_busy_cycles[63:32];
            REG_LOAD_A_READ_BEATS_LO:   profile_reg_read = load_a_read_beats[31:0];
            REG_LOAD_A_READ_BEATS_HI:   profile_reg_read = load_a_read_beats[63:32];
            REG_LOAD_B_READ_BEATS_LO:   profile_reg_read = load_b_read_beats[31:0];
            REG_LOAD_B_READ_BEATS_HI:   profile_reg_read = load_b_read_beats[63:32];
            REG_STORE_C_WRITE_BEATS_LO: profile_reg_read = store_c_write_beats[31:0];
            REG_STORE_C_WRITE_BEATS_HI: profile_reg_read = store_c_write_beats[63:32];
            REG_LOAD_A_AR_STALL_LO:     profile_reg_read = load_a_ar_stall_cycles[31:0];
            REG_LOAD_A_AR_STALL_HI:     profile_reg_read = load_a_ar_stall_cycles[63:32];
            REG_LOAD_B_AR_STALL_LO:     profile_reg_read = load_b_ar_stall_cycles[31:0];
            REG_LOAD_B_AR_STALL_HI:     profile_reg_read = load_b_ar_stall_cycles[63:32];
            REG_STORE_C_AW_STALL_LO:    profile_reg_read = store_c_aw_stall_cycles[31:0];
            REG_STORE_C_AW_STALL_HI:    profile_reg_read = store_c_aw_stall_cycles[63:32];
            REG_STORE_C_W_STALL_LO:     profile_reg_read = store_c_w_stall_cycles[31:0];
            REG_STORE_C_W_STALL_HI:     profile_reg_read = store_c_w_stall_cycles[63:32];
            REG_STORE_C_B_STALL_LO:     profile_reg_read = store_c_b_stall_cycles[31:0];
            REG_STORE_C_B_STALL_HI:     profile_reg_read = store_c_b_stall_cycles[63:32];
            default:                    profile_reg_read = 32'h00000000;
        endcase
    end
endfunction

always @(posedge ap_clk) begin
    if (!ap_rst_n) begin
        ctrl_aw_seen <= 1'b0;
        ctrl_awaddr_reg <= 8'd0;
        ctrl_w_seen <= 1'b0;
        ctrl_wdata_reg <= 32'd0;

        capture_active <= 1'b0;
        start_seen <= 1'b0;

        a_outstanding_beats <= 16'd0;
        b_outstanding_beats <= 16'd0;
        c_outstanding_write_beats <= 16'd0;
        c_outstanding_write_responses <= 8'd0;

        load_a_busy_cycles <= 64'd0;
        load_b_busy_cycles <= 64'd0;
        store_c_busy_cycles <= 64'd0;
        load_a_read_beats <= 64'd0;
        load_b_read_beats <= 64'd0;
        store_c_write_beats <= 64'd0;
        load_a_ar_stall_cycles <= 64'd0;
        load_b_ar_stall_cycles <= 64'd0;
        store_c_aw_stall_cycles <= 64'd0;
        store_c_w_stall_cycles <= 64'd0;
        store_c_b_stall_cycles <= 64'd0;

        profile_aw_seen <= 1'b0;
        profile_awaddr_reg <= {C_S_AXI_PROFILE_ADDR_WIDTH{1'b0}};
        profile_w_seen <= 1'b0;
        profile_wdata_reg <= {C_S_AXI_PROFILE_DATA_WIDTH{1'b0}};
        profile_wstrb_reg <= {(C_S_AXI_PROFILE_DATA_WIDTH/8){1'b0}};
        s_axi_profile_BRESP <= 2'b00;
        s_axi_profile_BVALID <= 1'b0;
        s_axi_profile_RDATA <= {C_S_AXI_PROFILE_DATA_WIDTH{1'b0}};
        s_axi_profile_RRESP <= 2'b00;
        s_axi_profile_RVALID <= 1'b0;
    end else begin
        if (ctrl_write_fire) begin
            ctrl_aw_seen <= 1'b0;
            ctrl_w_seen <= 1'b0;
        end else begin
            if (ctrl_aw_fire) begin
                ctrl_aw_seen <= 1'b1;
                ctrl_awaddr_reg <= ctrl_awaddr;
            end
            if (ctrl_w_fire) begin
                ctrl_w_seen <= 1'b1;
                ctrl_wdata_reg <= ctrl_wdata;
            end
        end

        if (profile_write_fire) begin
            profile_aw_seen <= 1'b0;
            profile_w_seen <= 1'b0;
            s_axi_profile_BVALID <= 1'b1;
            s_axi_profile_BRESP <= 2'b00;
        end else begin
            if (profile_aw_fire) begin
                profile_aw_seen <= 1'b1;
                profile_awaddr_reg <= s_axi_profile_AWADDR;
            end
            if (profile_w_fire) begin
                profile_w_seen <= 1'b1;
                profile_wdata_reg <= s_axi_profile_WDATA;
                profile_wstrb_reg <= s_axi_profile_WSTRB;
            end
            if (s_axi_profile_BVALID && s_axi_profile_BREADY) begin
                s_axi_profile_BVALID <= 1'b0;
            end
        end

        if (s_axi_profile_ARVALID && s_axi_profile_ARREADY) begin
            s_axi_profile_RVALID <= 1'b1;
            s_axi_profile_RRESP <= 2'b00;
            s_axi_profile_RDATA <= profile_reg_read(s_axi_profile_ARADDR[7:0]);
        end else if (s_axi_profile_RVALID && s_axi_profile_RREADY) begin
            s_axi_profile_RVALID <= 1'b0;
        end

        if (start_pulse || profile_sw_reset) begin
            capture_active <= start_pulse;
            start_seen <= start_pulse;

            a_outstanding_beats <= 16'd0;
            b_outstanding_beats <= 16'd0;
            c_outstanding_write_beats <= 16'd0;
            c_outstanding_write_responses <= 8'd0;

            load_a_busy_cycles <= 64'd0;
            load_b_busy_cycles <= 64'd0;
            store_c_busy_cycles <= 64'd0;
            load_a_read_beats <= 64'd0;
            load_b_read_beats <= 64'd0;
            store_c_write_beats <= 64'd0;
            load_a_ar_stall_cycles <= 64'd0;
            load_b_ar_stall_cycles <= 64'd0;
            store_c_aw_stall_cycles <= 64'd0;
            store_c_w_stall_cycles <= 64'd0;
            store_c_b_stall_cycles <= 64'd0;
        end else if (capture_active) begin
            if (a_busy) begin
                load_a_busy_cycles <= load_a_busy_cycles + 64'd1;
            end
            if (b_busy) begin
                load_b_busy_cycles <= load_b_busy_cycles + 64'd1;
            end
            if (c_busy) begin
                store_c_busy_cycles <= store_c_busy_cycles + 64'd1;
            end

            if (a_r_fire) begin
                load_a_read_beats <= load_a_read_beats + 64'd1;
            end
            if (b_r_fire) begin
                load_b_read_beats <= load_b_read_beats + 64'd1;
            end
            if (c_w_fire) begin
                store_c_write_beats <= store_c_write_beats + 64'd1;
            end

            if (a_arvalid && !a_arready) begin
                load_a_ar_stall_cycles <= load_a_ar_stall_cycles + 64'd1;
            end
            if (b_arvalid && !b_arready) begin
                load_b_ar_stall_cycles <= load_b_ar_stall_cycles + 64'd1;
            end
            if (c_awvalid && !c_awready) begin
                store_c_aw_stall_cycles <= store_c_aw_stall_cycles + 64'd1;
            end
            if (c_wvalid && !c_wready) begin
                store_c_w_stall_cycles <= store_c_w_stall_cycles + 64'd1;
            end
            if (c_bvalid && !c_bready) begin
                store_c_b_stall_cycles <= store_c_b_stall_cycles + 64'd1;
            end

            case ({a_ar_fire, a_r_fire})
                2'b10: a_outstanding_beats <= a_outstanding_beats + a_ar_beats;
                2'b01: a_outstanding_beats <= (a_outstanding_beats != 0) ? (a_outstanding_beats - 16'd1) : 16'd0;
                2'b11: a_outstanding_beats <= a_outstanding_beats + a_ar_beats - 16'd1;
                default: a_outstanding_beats <= a_outstanding_beats;
            endcase

            case ({b_ar_fire, b_r_fire})
                2'b10: b_outstanding_beats <= b_outstanding_beats + b_ar_beats;
                2'b01: b_outstanding_beats <= (b_outstanding_beats != 0) ? (b_outstanding_beats - 16'd1) : 16'd0;
                2'b11: b_outstanding_beats <= b_outstanding_beats + b_ar_beats - 16'd1;
                default: b_outstanding_beats <= b_outstanding_beats;
            endcase

            case ({c_aw_fire, c_w_fire})
                2'b10: c_outstanding_write_beats <= c_outstanding_write_beats + c_aw_beats;
                2'b01: c_outstanding_write_beats <= (c_outstanding_write_beats != 0) ? (c_outstanding_write_beats - 16'd1) : 16'd0;
                2'b11: c_outstanding_write_beats <= c_outstanding_write_beats + c_aw_beats - 16'd1;
                default: c_outstanding_write_beats <= c_outstanding_write_beats;
            endcase

            case ({c_aw_fire, c_b_fire})
                2'b10: c_outstanding_write_responses <= c_outstanding_write_responses + 8'd1;
                2'b01: c_outstanding_write_responses <= (c_outstanding_write_responses != 0) ? (c_outstanding_write_responses - 8'd1) : 8'd0;
                2'b11: c_outstanding_write_responses <= c_outstanding_write_responses;
                default: c_outstanding_write_responses <= c_outstanding_write_responses;
            endcase
        end
    end
end

endmodule
