`timescale 1ns/1ps

`ifndef EXPECTED_MAC
`define EXPECTED_MAC 4096
`endif

`ifndef TIMEOUT_CYCLES
`define TIMEOUT_CYCLES 20000
`endif

module tb_sdf2_pingpong;
  reg clk = 1'b0;
  reg reset = 1'b1;
  reg start = 1'b0;

  wire done;
  wire error;
  wire signed [63:0] checksum;
  wire signed [63:0] expectedChecksum;
  wire [31:0] cycleCount;
  wire [31:0] loadCycles;
  wire [31:0] computeCycles;
  wire [31:0] overlapCycles;
  wire [31:0] macCount;
  wire [31:0] accProbe;
  wire [7:0] phase;

  always #5 clk = ~clk;

  Sdf2RamPingPongSelfTest dut (
    .clk(clk),
    .reset(reset),
    .start(start),
    .done(done),
    .error(error),
    .checksum(checksum),
    .expectedChecksum(expectedChecksum),
    .cycleCount(cycleCount),
    .loadCycles(loadCycles),
    .computeCycles(computeCycles),
    .overlapCycles(overlapCycles),
    .macCount(macCount),
    .accProbe(accProbe),
    .phase(phase)
  );

  integer i;
  integer timeout_cycles;
  integer expected_mac_count;
  initial begin
    timeout_cycles = `TIMEOUT_CYCLES;
    expected_mac_count = `EXPECTED_MAC;

    $display("SDF2 ping-pong simulation start");
    $display("EXPECT expected_mac_count=%0d timeout_cycles=%0d", expected_mac_count, timeout_cycles);
    repeat (8) @(posedge clk);
    reset <= 1'b0;
    repeat (4) @(posedge clk);
    start <= 1'b1;
    @(posedge clk);
    start <= 1'b0;

    for (i = 0; i < timeout_cycles; i = i + 1) begin
      @(posedge clk);
      if (done) begin
        $display("DONE error=%0d checksum=%0d expected=%0d cycles=%0d load=%0d compute=%0d overlap=%0d mac=%0d phase=%0d",
                 error, checksum, expectedChecksum, cycleCount, loadCycles, computeCycles, overlapCycles, macCount, phase);
        if (^checksum === 1'bx || ^error === 1'bx) begin
          $fatal(1, "checksum/error contains X");
        end
        if (error) begin
          $fatal(1, "checksum mismatch");
        end
        if (overlapCycles == 0) begin
          $fatal(1, "expected non-zero load/compute overlap");
        end
        if (macCount != expected_mac_count) begin
          $fatal(1, "unexpected macCount %0d", macCount);
        end
        $finish;
      end
    end

    $fatal(1, "timeout waiting for done, phase=%0d", phase);
  end
endmodule
