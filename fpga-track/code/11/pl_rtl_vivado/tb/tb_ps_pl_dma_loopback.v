`timescale 1ns/1ps

module tb_ps_pl_dma_loopback;

    localparam integer AXIS_DATA_W = 32;
    localparam integer AXIL_ADDR_W = 6;
    localparam integer AXIL_DATA_W = 32;

    localparam [AXIL_ADDR_W-1:0] REG_CTRL      = 6'h00;
    localparam [AXIL_ADDR_W-1:0] REG_STATUS    = 6'h04;
    localparam [AXIL_ADDR_W-1:0] REG_RX_COUNT  = 6'h08;
    localparam [AXIL_ADDR_W-1:0] REG_TX_COUNT  = 6'h0c;
    localparam [AXIL_ADDR_W-1:0] REG_LAST_WORD = 6'h10;
    localparam [AXIL_ADDR_W-1:0] REG_VERSION   = 6'h14;

    localparam [31:0] VERSION_WORD = 32'h4450_4C31;

    reg clk;
    reg rst_n;

    reg  [AXIL_ADDR_W-1:0]     awaddr;
    reg  [2:0]                 awprot;
    reg                        awvalid;
    wire                       awready;
    reg  [AXIL_DATA_W-1:0]     wdata;
    reg  [(AXIL_DATA_W/8)-1:0] wstrb;
    reg                        wvalid;
    wire                       wready;
    wire [1:0]                 bresp;
    wire                       bvalid;
    reg                        bready;
    reg  [AXIL_ADDR_W-1:0]     araddr;
    reg  [2:0]                 arprot;
    reg                        arvalid;
    wire                       arready;
    wire [AXIL_DATA_W-1:0]     rdata;
    wire [1:0]                 rresp;
    wire                       rvalid;
    reg                        rready;

    reg  [AXIS_DATA_W-1:0]     s_tdata;
    reg  [(AXIS_DATA_W/8)-1:0] s_tkeep;
    reg                        s_tlast;
    reg                        s_tvalid;
    wire                       s_tready;

    wire [AXIS_DATA_W-1:0]     m_tdata;
    wire [(AXIS_DATA_W/8)-1:0] m_tkeep;
    wire                       m_tlast;
    wire                       m_tvalid;
    reg                        m_tready;

    reg [31:0] expected_data [0:2];
    reg       expected_last [0:2];
    integer recv_idx;

    ps_pl_dma_loopback #(
        .AXIS_DATA_W(AXIS_DATA_W),
        .AXIL_ADDR_W(AXIL_ADDR_W),
        .AXIL_DATA_W(AXIL_DATA_W)
    ) dut (
        .s_axi_control_aclk(clk),
        .s_axi_control_aresetn(rst_n),
        .s_axi_control_awaddr(awaddr),
        .s_axi_control_awprot(awprot),
        .s_axi_control_awvalid(awvalid),
        .s_axi_control_awready(awready),
        .s_axi_control_wdata(wdata),
        .s_axi_control_wstrb(wstrb),
        .s_axi_control_wvalid(wvalid),
        .s_axi_control_wready(wready),
        .s_axi_control_bresp(bresp),
        .s_axi_control_bvalid(bvalid),
        .s_axi_control_bready(bready),
        .s_axi_control_araddr(araddr),
        .s_axi_control_arprot(arprot),
        .s_axi_control_arvalid(arvalid),
        .s_axi_control_arready(arready),
        .s_axi_control_rdata(rdata),
        .s_axi_control_rresp(rresp),
        .s_axi_control_rvalid(rvalid),
        .s_axi_control_rready(rready),
        .s_axis_in_tdata(s_tdata),
        .s_axis_in_tkeep(s_tkeep),
        .s_axis_in_tlast(s_tlast),
        .s_axis_in_tvalid(s_tvalid),
        .s_axis_in_tready(s_tready),
        .m_axis_out_tdata(m_tdata),
        .m_axis_out_tkeep(m_tkeep),
        .m_axis_out_tlast(m_tlast),
        .m_axis_out_tvalid(m_tvalid),
        .m_axis_out_tready(m_tready)
    );

    always #5 clk = ~clk;

    always @(posedge clk) begin
        if (s_tvalid && s_tready) begin
            $display("TB: input handshake data=%08x last=%0d time=%0t", s_tdata, s_tlast, $time);
        end
        if (m_tvalid && m_tready) begin
            $display("TB: output handshake idx=%0d data=%08x last=%0d time=%0t", recv_idx, m_tdata, m_tlast, $time);
            if (recv_idx >= 3) begin
                $fatal(1, "received more stream words than expected");
            end
            if (m_tdata !== expected_data[recv_idx]) begin
                $fatal(1, "stream data mismatch at index %0d: got %08x expected %08x",
                       recv_idx, m_tdata, expected_data[recv_idx]);
            end
            if (m_tkeep !== 4'hf) begin
                $fatal(1, "stream keep mismatch at index %0d: got %x expected f",
                       recv_idx, m_tkeep);
            end
            if (m_tlast !== expected_last[recv_idx]) begin
                $fatal(1, "stream last mismatch at index %0d: got %0d expected %0d",
                       recv_idx, m_tlast, expected_last[recv_idx]);
            end
            recv_idx <= recv_idx + 1;
        end
    end

    task axi_write(input [AXIL_ADDR_W-1:0] addr, input [31:0] value);
    begin
        @(posedge clk);
        awaddr  <= addr;
        awprot  <= 3'b000;
        awvalid <= 1'b1;
        wdata   <= value;
        wstrb   <= 4'hf;
        wvalid  <= 1'b1;
        bready  <= 1'b0;

        while (!(awready && wready)) begin
            @(posedge clk);
        end

        @(posedge clk);
        awvalid <= 1'b0;
        wvalid  <= 1'b0;
        bready  <= 1'b1;

        while (!bvalid) begin
            @(posedge clk);
        end
        if (bresp !== 2'b00) begin
            $fatal(1, "axi_write got BRESP=%0b", bresp);
        end

        @(posedge clk);
        bready <= 1'b0;
    end
    endtask

    task axi_read(input [AXIL_ADDR_W-1:0] addr, output [31:0] value);
    begin
        @(posedge clk);
        araddr  <= addr;
        arprot  <= 3'b000;
        arvalid <= 1'b1;
        rready  <= 1'b0;

        while (!arready) begin
            @(posedge clk);
        end

        @(posedge clk);
        arvalid <= 1'b0;
        rready  <= 1'b1;

        while (!rvalid) begin
            @(posedge clk);
        end
        if (rresp !== 2'b00) begin
            $fatal(1, "axi_read got RRESP=%0b", rresp);
        end
        value = rdata;

        @(posedge clk);
        rready <= 1'b0;
    end
    endtask

    task stream_send(input [31:0] data_word, input bit last_word);
    begin
        @(posedge clk);
        s_tdata  <= data_word;
        s_tkeep  <= 4'hf;
        s_tlast  <= last_word;
        s_tvalid <= 1'b1;

        @(posedge clk);
        while (!(s_tvalid && s_tready)) begin
            @(posedge clk);
        end

        s_tvalid <= 1'b0;
        s_tlast  <= 1'b0;
    end
    endtask

    reg [31:0] rdval;

    initial begin
        #200000;
        $fatal(1, "timeout waiting for loopback test to finish");
    end

    initial begin
        clk    = 1'b0;
        rst_n  = 1'b0;
        awaddr = '0;
        awprot = '0;
        awvalid = 1'b0;
        wdata  = '0;
        wstrb  = '0;
        wvalid = 1'b0;
        bready = 1'b0;
        araddr = '0;
        arprot = '0;
        arvalid = 1'b0;
        rready = 1'b0;
        s_tdata = '0;
        s_tkeep = '0;
        s_tlast = 1'b0;
        s_tvalid = 1'b0;
        m_tready = 1'b0;
        recv_idx = 0;
        expected_data[0] = 32'h1122_3344;
        expected_data[1] = 32'h5566_7788;
        expected_data[2] = 32'hdead_beef;
        expected_last[0] = 1'b0;
        expected_last[1] = 1'b0;
        expected_last[2] = 1'b1;

        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        $display("TB: read VERSION");
        axi_read(REG_VERSION, rdval);
        if (rdval !== VERSION_WORD) begin
            $fatal(1, "VERSION mismatch: got %08x expected %08x", rdval, VERSION_WORD);
        end

        $display("TB: read STATUS after reset");
        axi_read(REG_STATUS, rdval);
        if (rdval !== 32'h0) begin
            $fatal(1, "STATUS after reset mismatch: got %08x expected 00000000", rdval);
        end

        $display("TB: enable");
        axi_write(REG_CTRL, 32'h1);

        $display("TB: read CTRL");
        axi_read(REG_CTRL, rdval);
        if (rdval !== 32'h1) begin
            $fatal(1, "CTRL mismatch after enable: got %08x expected 00000001", rdval);
        end

        $display("TB: read STATUS after enable");
        axi_read(REG_STATUS, rdval);
        if (rdval !== 32'h3) begin
            $fatal(1, "STATUS mismatch after enable: got %08x expected 00000003", rdval);
        end

        $display("TB: stream send #0");
        @(negedge clk);
        m_tready = 1'b1;
        stream_send(expected_data[0], expected_last[0]);
        wait (recv_idx == 1);

        $display("TB: stream send #1 with backpressure");
        @(negedge clk);
        m_tready = 1'b0;
        repeat (2) @(posedge clk);
        stream_send(expected_data[1], expected_last[1]);
        repeat (3) @(posedge clk);
        @(negedge clk);
        m_tready = 1'b1;

        wait (recv_idx == 2);
        repeat (2) @(posedge clk);
        $display("TB: stream send #2");
        stream_send(expected_data[2], expected_last[2]);

        wait (recv_idx == 3);
        repeat (2) @(posedge clk);

        $display("TB: read counters");
        axi_read(REG_RX_COUNT, rdval);
        if (rdval !== 32'd3) begin
            $fatal(1, "RX_COUNT mismatch: got %0d expected 3", rdval);
        end

        axi_read(REG_TX_COUNT, rdval);
        if (rdval !== 32'd3) begin
            $fatal(1, "TX_COUNT mismatch: got %0d expected 3", rdval);
        end

        axi_read(REG_LAST_WORD, rdval);
        if (rdval !== expected_data[2]) begin
            $fatal(1, "LAST_WORD mismatch: got %08x expected %08x", rdval, expected_data[2]);
        end

        $display("TB: soft reset");
        axi_write(REG_CTRL, 32'h2);

        axi_read(REG_CTRL, rdval);
        if (rdval !== 32'h0) begin
            $fatal(1, "CTRL mismatch after soft reset: got %08x expected 00000000", rdval);
        end

        axi_read(REG_STATUS, rdval);
        if (rdval !== 32'h0) begin
            $fatal(1, "STATUS mismatch after soft reset: got %08x expected 00000000", rdval);
        end

        axi_read(REG_RX_COUNT, rdval);
        if (rdval !== 32'd0) begin
            $fatal(1, "RX_COUNT after soft reset mismatch: got %0d expected 0", rdval);
        end

        axi_read(REG_TX_COUNT, rdval);
        if (rdval !== 32'd0) begin
            $fatal(1, "TX_COUNT after soft reset mismatch: got %0d expected 0", rdval);
        end

        $display("PASS: ps_pl_dma_loopback AXI-Lite + AXI-Stream test");
        $finish;
    end

endmodule
