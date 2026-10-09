`timescale 1 ns / 1 ps

module smolvlm_axi_profile_regs #
(
    parameter integer C_S_AXI_PROFILE_ADDR_WIDTH = 8,
    parameter integer C_S_AXI_PROFILE_DATA_WIDTH = 32
)
(
    input  wire                                      ap_clk,
    input  wire                                      ap_rst_n,

    input  wire                                      ctrl_awvalid,
    input  wire                                      ctrl_awready,
    input  wire [7:0]                                ctrl_awaddr,
    input  wire                                      ctrl_wvalid,
    input  wire                                      ctrl_wready,
    input  wire [31:0]                               ctrl_wdata,
    input  wire                                      ctrl_arvalid,
    input  wire                                      ctrl_arready,
    input  wire [7:0]                                ctrl_araddr,
    input  wire                                      ctrl_rvalid,
    input  wire                                      ctrl_rready,
    input  wire [31:0]                               ctrl_rdata,

    input  wire                                      d_awvalid,
    input  wire                                      d_awready,
    input  wire [7:0]                                d_awlen,
    input  wire                                      d_wvalid,
    input  wire                                      d_wready,
    input  wire [15:0]                               d_wstrb,
    input  wire                                      d_arvalid,
    input  wire                                      d_arready,
    input  wire [7:0]                                d_arlen,
    input  wire                                      d_rvalid,
    input  wire                                      d_rready,
    input  wire                                      d_bvalid,
    input  wire                                      d_bready,

    input  wire                                      w_arvalid,
    input  wire                                      w_arready,
    input  wire [7:0]                                w_arlen,
    input  wire                                      w_rvalid,
    input  wire                                      w_rready,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi_profile AWADDR" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME s_axi_profile, PROTOCOL AXI4LITE, ADDR_WIDTH 8, DATA_WIDTH 32, FREQ_HZ 200000000, ID_WIDTH 0, AWUSER_WIDTH 0, ARUSER_WIDTH 0, WUSER_WIDTH 0, RUSER_WIDTH 0, BUSER_WIDTH 0, HAS_BURST 0, HAS_LOCK 0, HAS_PROT 0, HAS_CACHE 0, HAS_QOS 0, HAS_REGION 0, HAS_WSTRB 1, HAS_BRESP 1, HAS_RRESP 1, MAX_BURST_LENGTH 1, NUM_READ_OUTSTANDING 1, NUM_WRITE_OUTSTANDING 1, SUPPORTS_NARROW_BURST 0" *)
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

localparam [31:0] PROFILE_MAGIC = 32'h41584931;
localparam [31:0] PROFILE_VERSION = 32'h00000001;

localparam [7:0] REG_CONTROL                 = 8'h00;
localparam [7:0] REG_STATUS                  = 8'h04;
localparam [7:0] REG_MAGIC                   = 8'h08;
localparam [7:0] REG_VERSION                 = 8'h0c;
localparam [7:0] REG_AXI_CYCLES_LO           = 8'h10;
localparam [7:0] REG_AXI_CYCLES_HI           = 8'h14;
localparam [7:0] REG_AXI_MEM_BUSY_LO         = 8'h18;
localparam [7:0] REG_AXI_MEM_BUSY_HI         = 8'h1c;
localparam [7:0] REG_DATA_R_BUSY_LO          = 8'h20;
localparam [7:0] REG_DATA_R_BUSY_HI          = 8'h24;
localparam [7:0] REG_DATA_W_BUSY_LO          = 8'h28;
localparam [7:0] REG_DATA_W_BUSY_HI          = 8'h2c;
localparam [7:0] REG_WEIGHT_R_BUSY_LO        = 8'h30;
localparam [7:0] REG_WEIGHT_R_BUSY_HI        = 8'h34;
localparam [7:0] REG_DATA_R_BEATS_LO         = 8'h38;
localparam [7:0] REG_DATA_R_BEATS_HI         = 8'h3c;
localparam [7:0] REG_DATA_W_BEATS_LO         = 8'h40;
localparam [7:0] REG_DATA_W_BEATS_HI         = 8'h44;
localparam [7:0] REG_WEIGHT_R_BEATS_LO       = 8'h48;
localparam [7:0] REG_WEIGHT_R_BEATS_HI       = 8'h4c;
localparam [7:0] REG_DATA_W_BYTES_LO         = 8'h50;
localparam [7:0] REG_DATA_W_BYTES_HI         = 8'h54;
localparam [7:0] REG_DATA_R_TXNS_LO          = 8'h58;
localparam [7:0] REG_DATA_R_TXNS_HI          = 8'h5c;
localparam [7:0] REG_DATA_W_TXNS_LO          = 8'h60;
localparam [7:0] REG_DATA_W_TXNS_HI          = 8'h64;
localparam [7:0] REG_WEIGHT_R_TXNS_LO        = 8'h68;
localparam [7:0] REG_WEIGHT_R_TXNS_HI        = 8'h6c;
localparam [7:0] REG_DATA_AR_STALL_LO        = 8'h70;
localparam [7:0] REG_DATA_AR_STALL_HI        = 8'h74;
localparam [7:0] REG_DATA_AW_STALL_LO        = 8'h78;
localparam [7:0] REG_DATA_AW_STALL_HI        = 8'h7c;
localparam [7:0] REG_DATA_W_STALL_LO         = 8'h80;
localparam [7:0] REG_DATA_W_STALL_HI         = 8'h84;
localparam [7:0] REG_DATA_R_STALL_LO         = 8'h88;
localparam [7:0] REG_DATA_R_STALL_HI         = 8'h8c;
localparam [7:0] REG_DATA_B_STALL_LO         = 8'h90;
localparam [7:0] REG_DATA_B_STALL_HI         = 8'h94;
localparam [7:0] REG_WEIGHT_AR_STALL_LO      = 8'h98;
localparam [7:0] REG_WEIGHT_AR_STALL_HI      = 8'h9c;
localparam [7:0] REG_WEIGHT_R_STALL_LO       = 8'ha0;
localparam [7:0] REG_WEIGHT_R_STALL_HI       = 8'ha4;

reg        ctrl_aw_seen;
reg [7:0]  ctrl_awaddr_reg;
reg        ctrl_w_seen;
reg [31:0] ctrl_wdata_reg;
reg        ctrl_ar_seen;
reg [7:0]  ctrl_araddr_reg;

reg        capture_active;
reg        capture_frozen;
reg        start_seen;
reg        done_seen;

reg [15:0] d_outstanding_read_beats;
reg [15:0] d_outstanding_write_beats;
reg [7:0]  d_outstanding_write_responses;
reg [15:0] w_outstanding_read_beats;

reg [63:0] axi_cycles;
reg [63:0] axi_mem_busy_cycles;
reg [63:0] data_r_busy_cycles;
reg [63:0] data_w_busy_cycles;
reg [63:0] weight_r_busy_cycles;
reg [63:0] data_r_beats;
reg [63:0] data_w_beats;
reg [63:0] weight_r_beats;
reg [63:0] data_w_bytes;
reg [63:0] data_r_txns;
reg [63:0] data_w_txns;
reg [63:0] weight_r_txns;
reg [63:0] data_ar_stall_cycles;
reg [63:0] data_aw_stall_cycles;
reg [63:0] data_w_stall_cycles;
reg [63:0] data_r_stall_cycles;
reg [63:0] data_b_stall_cycles;
reg [63:0] weight_ar_stall_cycles;
reg [63:0] weight_r_stall_cycles;

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

wire ctrl_ar_fire = ctrl_arvalid && ctrl_arready;
wire ctrl_r_fire = ctrl_rvalid && ctrl_rready;
wire [7:0] ctrl_r_addr = ctrl_ar_seen ? ctrl_araddr_reg : ctrl_araddr;
wire done_pulse = capture_active && !capture_frozen &&
                  ctrl_r_fire && (ctrl_r_addr == 8'h00) && ctrl_rdata[1];

wire d_aw_fire = d_awvalid && d_awready;
wire d_w_fire = d_wvalid && d_wready;
wire d_ar_fire = d_arvalid && d_arready;
wire d_r_fire = d_rvalid && d_rready;
wire d_b_fire = d_bvalid && d_bready;
wire w_ar_fire = w_arvalid && w_arready;
wire w_r_fire = w_rvalid && w_rready;

wire [15:0] d_aw_beats = {8'd0, d_awlen} + 16'd1;
wire [15:0] d_ar_beats = {8'd0, d_arlen} + 16'd1;
wire [15:0] w_ar_beats = {8'd0, w_arlen} + 16'd1;

wire counting = capture_active && !capture_frozen;
wire data_r_busy = counting && (d_arvalid || d_rvalid || (d_outstanding_read_beats != 0));
wire data_w_busy = counting && (d_awvalid || d_wvalid || d_bvalid ||
                                (d_outstanding_write_beats != 0) ||
                                (d_outstanding_write_responses != 0));
wire weight_r_busy = counting && (w_arvalid || w_rvalid || (w_outstanding_read_beats != 0));
wire any_mem_busy = data_r_busy || data_w_busy || weight_r_busy;

wire profile_aw_fire = s_axi_profile_AWVALID && s_axi_profile_AWREADY;
wire profile_w_fire = s_axi_profile_WVALID && s_axi_profile_WREADY;
wire profile_write_fire = (profile_aw_seen || profile_aw_fire) &&
                          (profile_w_seen || profile_w_fire) &&
                          !s_axi_profile_BVALID;
wire [C_S_AXI_PROFILE_ADDR_WIDTH-1:0] profile_write_addr = profile_aw_fire ? s_axi_profile_AWADDR : profile_awaddr_reg;
wire [C_S_AXI_PROFILE_DATA_WIDTH-1:0] profile_write_data = profile_w_fire ? s_axi_profile_WDATA : profile_wdata_reg;
wire profile_sw_reset = profile_write_fire &&
                        (profile_write_addr[7:0] == REG_CONTROL) &&
                        profile_write_data[0];

assign s_axi_profile_AWREADY = !profile_aw_seen && !s_axi_profile_BVALID;
assign s_axi_profile_WREADY = !profile_w_seen && !s_axi_profile_BVALID;
assign s_axi_profile_ARREADY = !s_axi_profile_RVALID;

function [4:0] popcount16;
    input [15:0] value;
    integer i;
    begin
        popcount16 = 5'd0;
        for (i = 0; i < 16; i = i + 1) begin
            popcount16 = popcount16 + value[i];
        end
    end
endfunction

function [31:0] profile_reg_read;
    input [7:0] reg_addr;
    begin
        case (reg_addr)
            REG_CONTROL:              profile_reg_read = 32'h00000000;
            REG_STATUS:               profile_reg_read = {28'd0, done_seen, start_seen, capture_frozen, capture_active};
            REG_MAGIC:                profile_reg_read = PROFILE_MAGIC;
            REG_VERSION:              profile_reg_read = PROFILE_VERSION;
            REG_AXI_CYCLES_LO:        profile_reg_read = axi_cycles[31:0];
            REG_AXI_CYCLES_HI:        profile_reg_read = axi_cycles[63:32];
            REG_AXI_MEM_BUSY_LO:      profile_reg_read = axi_mem_busy_cycles[31:0];
            REG_AXI_MEM_BUSY_HI:      profile_reg_read = axi_mem_busy_cycles[63:32];
            REG_DATA_R_BUSY_LO:       profile_reg_read = data_r_busy_cycles[31:0];
            REG_DATA_R_BUSY_HI:       profile_reg_read = data_r_busy_cycles[63:32];
            REG_DATA_W_BUSY_LO:       profile_reg_read = data_w_busy_cycles[31:0];
            REG_DATA_W_BUSY_HI:       profile_reg_read = data_w_busy_cycles[63:32];
            REG_WEIGHT_R_BUSY_LO:     profile_reg_read = weight_r_busy_cycles[31:0];
            REG_WEIGHT_R_BUSY_HI:     profile_reg_read = weight_r_busy_cycles[63:32];
            REG_DATA_R_BEATS_LO:      profile_reg_read = data_r_beats[31:0];
            REG_DATA_R_BEATS_HI:      profile_reg_read = data_r_beats[63:32];
            REG_DATA_W_BEATS_LO:      profile_reg_read = data_w_beats[31:0];
            REG_DATA_W_BEATS_HI:      profile_reg_read = data_w_beats[63:32];
            REG_WEIGHT_R_BEATS_LO:    profile_reg_read = weight_r_beats[31:0];
            REG_WEIGHT_R_BEATS_HI:    profile_reg_read = weight_r_beats[63:32];
            REG_DATA_W_BYTES_LO:      profile_reg_read = data_w_bytes[31:0];
            REG_DATA_W_BYTES_HI:      profile_reg_read = data_w_bytes[63:32];
            REG_DATA_R_TXNS_LO:       profile_reg_read = data_r_txns[31:0];
            REG_DATA_R_TXNS_HI:       profile_reg_read = data_r_txns[63:32];
            REG_DATA_W_TXNS_LO:       profile_reg_read = data_w_txns[31:0];
            REG_DATA_W_TXNS_HI:       profile_reg_read = data_w_txns[63:32];
            REG_WEIGHT_R_TXNS_LO:     profile_reg_read = weight_r_txns[31:0];
            REG_WEIGHT_R_TXNS_HI:     profile_reg_read = weight_r_txns[63:32];
            REG_DATA_AR_STALL_LO:     profile_reg_read = data_ar_stall_cycles[31:0];
            REG_DATA_AR_STALL_HI:     profile_reg_read = data_ar_stall_cycles[63:32];
            REG_DATA_AW_STALL_LO:     profile_reg_read = data_aw_stall_cycles[31:0];
            REG_DATA_AW_STALL_HI:     profile_reg_read = data_aw_stall_cycles[63:32];
            REG_DATA_W_STALL_LO:      profile_reg_read = data_w_stall_cycles[31:0];
            REG_DATA_W_STALL_HI:      profile_reg_read = data_w_stall_cycles[63:32];
            REG_DATA_R_STALL_LO:      profile_reg_read = data_r_stall_cycles[31:0];
            REG_DATA_R_STALL_HI:      profile_reg_read = data_r_stall_cycles[63:32];
            REG_DATA_B_STALL_LO:      profile_reg_read = data_b_stall_cycles[31:0];
            REG_DATA_B_STALL_HI:      profile_reg_read = data_b_stall_cycles[63:32];
            REG_WEIGHT_AR_STALL_LO:   profile_reg_read = weight_ar_stall_cycles[31:0];
            REG_WEIGHT_AR_STALL_HI:   profile_reg_read = weight_ar_stall_cycles[63:32];
            REG_WEIGHT_R_STALL_LO:    profile_reg_read = weight_r_stall_cycles[31:0];
            REG_WEIGHT_R_STALL_HI:    profile_reg_read = weight_r_stall_cycles[63:32];
            default:                  profile_reg_read = 32'h00000000;
        endcase
    end
endfunction

task clear_counters;
    begin
        d_outstanding_read_beats <= 16'd0;
        d_outstanding_write_beats <= 16'd0;
        d_outstanding_write_responses <= 8'd0;
        w_outstanding_read_beats <= 16'd0;
        axi_cycles <= 64'd0;
        axi_mem_busy_cycles <= 64'd0;
        data_r_busy_cycles <= 64'd0;
        data_w_busy_cycles <= 64'd0;
        weight_r_busy_cycles <= 64'd0;
        data_r_beats <= 64'd0;
        data_w_beats <= 64'd0;
        weight_r_beats <= 64'd0;
        data_w_bytes <= 64'd0;
        data_r_txns <= 64'd0;
        data_w_txns <= 64'd0;
        weight_r_txns <= 64'd0;
        data_ar_stall_cycles <= 64'd0;
        data_aw_stall_cycles <= 64'd0;
        data_w_stall_cycles <= 64'd0;
        data_r_stall_cycles <= 64'd0;
        data_b_stall_cycles <= 64'd0;
        weight_ar_stall_cycles <= 64'd0;
        weight_r_stall_cycles <= 64'd0;
    end
endtask

always @(posedge ap_clk) begin
    if (!ap_rst_n) begin
        ctrl_aw_seen <= 1'b0;
        ctrl_awaddr_reg <= 8'd0;
        ctrl_w_seen <= 1'b0;
        ctrl_wdata_reg <= 32'd0;
        ctrl_ar_seen <= 1'b0;
        ctrl_araddr_reg <= 8'd0;
        capture_active <= 1'b0;
        capture_frozen <= 1'b0;
        start_seen <= 1'b0;
        done_seen <= 1'b0;
        clear_counters();
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

        if (ctrl_ar_fire) begin
            ctrl_ar_seen <= 1'b1;
            ctrl_araddr_reg <= ctrl_araddr;
        end
        if (ctrl_r_fire) begin
            ctrl_ar_seen <= 1'b0;
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
            capture_frozen <= 1'b0;
            start_seen <= start_pulse;
            done_seen <= 1'b0;
            clear_counters();
        end else if (counting) begin
            axi_cycles <= axi_cycles + 64'd1;
            if (any_mem_busy) begin
                axi_mem_busy_cycles <= axi_mem_busy_cycles + 64'd1;
            end
            if (data_r_busy) begin
                data_r_busy_cycles <= data_r_busy_cycles + 64'd1;
            end
            if (data_w_busy) begin
                data_w_busy_cycles <= data_w_busy_cycles + 64'd1;
            end
            if (weight_r_busy) begin
                weight_r_busy_cycles <= weight_r_busy_cycles + 64'd1;
            end

            if (d_r_fire) begin
                data_r_beats <= data_r_beats + 64'd1;
            end
            if (d_w_fire) begin
                data_w_beats <= data_w_beats + 64'd1;
                data_w_bytes <= data_w_bytes + {59'd0, popcount16(d_wstrb)};
            end
            if (w_r_fire) begin
                weight_r_beats <= weight_r_beats + 64'd1;
            end
            if (d_ar_fire) begin
                data_r_txns <= data_r_txns + 64'd1;
            end
            if (d_aw_fire) begin
                data_w_txns <= data_w_txns + 64'd1;
            end
            if (w_ar_fire) begin
                weight_r_txns <= weight_r_txns + 64'd1;
            end

            if (d_arvalid && !d_arready) begin
                data_ar_stall_cycles <= data_ar_stall_cycles + 64'd1;
            end
            if (d_awvalid && !d_awready) begin
                data_aw_stall_cycles <= data_aw_stall_cycles + 64'd1;
            end
            if (d_wvalid && !d_wready) begin
                data_w_stall_cycles <= data_w_stall_cycles + 64'd1;
            end
            if (d_rvalid && !d_rready) begin
                data_r_stall_cycles <= data_r_stall_cycles + 64'd1;
            end
            if (d_bvalid && !d_bready) begin
                data_b_stall_cycles <= data_b_stall_cycles + 64'd1;
            end
            if (w_arvalid && !w_arready) begin
                weight_ar_stall_cycles <= weight_ar_stall_cycles + 64'd1;
            end
            if (w_rvalid && !w_rready) begin
                weight_r_stall_cycles <= weight_r_stall_cycles + 64'd1;
            end

            case ({d_ar_fire, d_r_fire})
                2'b10: d_outstanding_read_beats <= d_outstanding_read_beats + d_ar_beats;
                2'b01: d_outstanding_read_beats <= (d_outstanding_read_beats != 0) ? (d_outstanding_read_beats - 16'd1) : 16'd0;
                2'b11: d_outstanding_read_beats <= d_outstanding_read_beats + d_ar_beats - 16'd1;
                default: d_outstanding_read_beats <= d_outstanding_read_beats;
            endcase

            case ({d_aw_fire, d_w_fire})
                2'b10: d_outstanding_write_beats <= d_outstanding_write_beats + d_aw_beats;
                2'b01: d_outstanding_write_beats <= (d_outstanding_write_beats != 0) ? (d_outstanding_write_beats - 16'd1) : 16'd0;
                2'b11: d_outstanding_write_beats <= d_outstanding_write_beats + d_aw_beats - 16'd1;
                default: d_outstanding_write_beats <= d_outstanding_write_beats;
            endcase

            case ({d_aw_fire, d_b_fire})
                2'b10: d_outstanding_write_responses <= d_outstanding_write_responses + 8'd1;
                2'b01: d_outstanding_write_responses <= (d_outstanding_write_responses != 0) ? (d_outstanding_write_responses - 8'd1) : 8'd0;
                default: d_outstanding_write_responses <= d_outstanding_write_responses;
            endcase

            case ({w_ar_fire, w_r_fire})
                2'b10: w_outstanding_read_beats <= w_outstanding_read_beats + w_ar_beats;
                2'b01: w_outstanding_read_beats <= (w_outstanding_read_beats != 0) ? (w_outstanding_read_beats - 16'd1) : 16'd0;
                2'b11: w_outstanding_read_beats <= w_outstanding_read_beats + w_ar_beats - 16'd1;
                default: w_outstanding_read_beats <= w_outstanding_read_beats;
            endcase

            if (done_pulse) begin
                capture_frozen <= 1'b1;
                done_seen <= 1'b1;
            end
        end
    end
end

endmodule
