// tb_pe_array_contract_liveness.sv --- compact HLS/RTL stream contract gate.
//
// This test is intentionally small and watchdog-driven. It checks that the RTL
// PE wrapper can consume the hardware contract used by vlm_engine_v4:
//   128 compute beats, each 32-beat group starts with clear+en and ends with
//   en+commit. The PE autonomously drains 32 rows per group with no extra ctrl
//   beats; the final group commit also carries last. If the HLS producer already
//   presents the next group while drain is active, PE must backpressure compute
//   until the current group drain completes.
`timescale 1ns/1ps

module tb_pe_array_contract_liveness;
    localparam int K = 128;
    localparam int M = 32;
    localparam int N = 32;
    localparam int GROUP = 32;
    localparam int GROUPS = K / GROUP;
    localparam int PSUM_BEATS_PER_ROW = 4;
    localparam int PSUM_BEATS = M * GROUPS * PSUM_BEATS_PER_ROW;
    localparam int WATCHDOG_CYCLES = 5000;

    logic clk = 0;
    logic rst_n = 0;
    always #5 clk = ~clk;

    logic [255:0] a_data, w_data;
    logic [15:0]  ctrl_data;
    logic         a_valid, w_valid, ctrl_valid;
    wire          a_ready, w_ready, ctrl_ready;
    wire [255:0] psum_data;
    wire          psum_valid;
    logic         psum_ready;
    wire          psum_last;

    pe_array_axis dut (
        .ap_clk(clk),
        .ap_rst_n(rst_n),
        .s_axis_a_tdata(a_data),
        .s_axis_a_tvalid(a_valid),
        .s_axis_a_tready(a_ready),
        .s_axis_w_tdata(w_data),
        .s_axis_w_tvalid(w_valid),
        .s_axis_w_tready(w_ready),
        .s_axis_ctrl_tdata(ctrl_data),
        .s_axis_ctrl_tvalid(ctrl_valid),
        .s_axis_ctrl_tready(ctrl_ready),
        .m_axis_psum_tdata(psum_data),
        .m_axis_psum_tvalid(psum_valid),
        .m_axis_psum_tready(psum_ready),
        .m_axis_psum_tlast(psum_last)
    );

    task automatic reset_dut;
    begin
        a_data = 256'd0;
        w_data = 256'd0;
        ctrl_data = 16'd0;
        a_valid = 1'b0;
        w_valid = 1'b0;
        ctrl_valid = 1'b0;
        psum_ready = 1'b1;
        rst_n = 1'b0;
        repeat (6) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);
    end
    endtask

    task automatic send_compute_beat(input logic [15:0] ctrl);
    begin
        @(negedge clk);
        a_data = {32{8'h01}};
        w_data = {32{8'h01}};
        ctrl_data = ctrl;
        a_valid = 1'b1;
        w_valid = 1'b1;
        ctrl_valid = 1'b1;
        do @(posedge clk); while (!(a_ready && w_ready && ctrl_ready));
    end
    endtask

    task automatic run_contract_case(input string name, input bit backpressure);
        int psum_count;
        int cycle;
        logic [15:0] ctrl;
    begin
        $display("[CONTRACT] START %s", name);
        reset_dut();

        psum_count = 0;
        cycle = 0;

        for (int g = 0; g < GROUPS; g++) begin
            for (int kg = 0; kg < GROUP; kg++) begin
                int k = g * GROUP + kg;
                ctrl = 16'b0010;          // en
                if (kg == 0) ctrl[0] = 1'b1; // clear group
                if (kg == GROUP - 1) begin
                    ctrl[2] = 1'b1;       // commit group
                    if (k == K - 1)
                        ctrl[3] = 1'b1;   // final tile TLAST
                end
                send_compute_beat(ctrl);
            end

            @(negedge clk);
            a_valid = 1'b0;
            w_valid = 1'b0;
            ctrl_valid = 1'b0;
            ctrl_data = 16'b0000;

            while ((psum_count % (M * PSUM_BEATS_PER_ROW)) != 0 ||
                   psum_count == g * M * PSUM_BEATS_PER_ROW) begin
                psum_ready = backpressure ? ((cycle % 7) != 3) : 1'b1;
                @(posedge clk);
                cycle++;
                if (cycle > WATCHDOG_CYCLES) begin
                    $fatal(1, "[CONTRACT] TIMEOUT %s psum_count=%0d ctrl_valid=%0b ctrl_ready=%0b psum_valid=%0b",
                           name, psum_count, ctrl_valid, ctrl_ready, psum_valid);
                end

                if (!(psum_valid && psum_ready))
                    continue;
                for (int lane = 0; lane < N / PSUM_BEATS_PER_ROW; lane++) begin
                    logic signed [31:0] got;
                    got = psum_data[lane*32 +: 32];
                    if (got !== GROUP) begin
                        $fatal(1, "[CONTRACT] BAD_PSUM %s beat=%0d lane=%0d got=%0d exp=%0d",
                               name, psum_count, lane, got, GROUP);
                    end
                end
                if (psum_last !== (psum_count == PSUM_BEATS - 1)) begin
                    $fatal(1, "[CONTRACT] BAD_TLAST %s beat=%0d got=%0b",
                           name, psum_count, psum_last);
                end
                psum_count++;
            end
        end

        $display("[CONTRACT] PASS %s cycles=%0d", name, cycle);
    end
    endtask

    task automatic run_continuous_contract_case(input string name, input bit backpressure);
        int psum_count;
        int cycle;
        logic [15:0] ctrl;
    begin
        $display("[CONTRACT] START %s", name);
        reset_dut();

        psum_count = 0;
        cycle = 0;

        fork
            begin : producer
                for (int g = 0; g < GROUPS; g++) begin
                    for (int kg = 0; kg < GROUP; kg++) begin
                        int k = g * GROUP + kg;
                        ctrl = 16'b0010;             // en
                        if (kg == 0) ctrl[0] = 1'b1; // clear group
                        if (kg == GROUP - 1) begin
                            ctrl[2] = 1'b1;          // commit group
                            if (k == K - 1)
                                ctrl[3] = 1'b1;      // final tile TLAST
                        end
                        send_compute_beat(ctrl);
                    end
                end

                @(negedge clk);
                a_valid = 1'b0;
                w_valid = 1'b0;
                ctrl_valid = 1'b0;
                ctrl_data = 16'b0000;
            end

            begin : consumer
                while (psum_count < PSUM_BEATS) begin
                    psum_ready = backpressure ? ((cycle % 7) != 3) : 1'b1;
                    @(posedge clk);
                    cycle++;
                    if (cycle > WATCHDOG_CYCLES) begin
                        $fatal(1, "[CONTRACT] TIMEOUT %s psum_count=%0d ctrl_valid=%0b ctrl_ready=%0b psum_valid=%0b",
                               name, psum_count, ctrl_valid, ctrl_ready, psum_valid);
                    end

                    if (!(psum_valid && psum_ready))
                        continue;
                    for (int lane = 0; lane < N / PSUM_BEATS_PER_ROW; lane++) begin
                        logic signed [31:0] got;
                        got = psum_data[lane*32 +: 32];
                        if (got !== GROUP) begin
                            $fatal(1, "[CONTRACT] BAD_PSUM %s beat=%0d lane=%0d got=%0d exp=%0d",
                                   name, psum_count, lane, got, GROUP);
                        end
                    end
                    if (psum_last !== (psum_count == PSUM_BEATS - 1)) begin
                        $fatal(1, "[CONTRACT] BAD_TLAST %s beat=%0d got=%0b",
                               name, psum_count, psum_last);
                    end
                    psum_count++;
                end
            end
        join

        $display("[CONTRACT] PASS %s cycles=%0d", name, cycle);
    end
    endtask

    initial begin
        run_contract_case("commit-autonomous-drain", 1'b0);
        run_contract_case("commit-autonomous-drain-backpressure", 1'b1);
        run_continuous_contract_case("commit-autonomous-drain-continuous-producer", 1'b0);
        run_continuous_contract_case("commit-autonomous-drain-continuous-producer-backpressure", 1'b1);
        $display("[CONTRACT] PASS: HLS/RTL commit/autonomous-drain stream contract is live");
        $finish;
    end
endmodule
