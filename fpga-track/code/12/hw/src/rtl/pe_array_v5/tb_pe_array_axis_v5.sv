// tb_pe_array_axis_v5.sv --- self-checking testbench for pe_array_axis_v5.
// Runs directed cases first, then a random backpressure case. The directed
// cases make one-cycle, sign/packing, and row/column errors easy to identify.
`timescale 1ns/1ps
module tb_pe_array_axis_v5;
    localparam int K = 128;
    localparam int M = 32;
    localparam int N = 32;
    localparam int GROUP = 128;
    localparam int GROUPS = K / GROUP;
    localparam int PSUM_BEATS_PER_ROW = 4;
    localparam int PSUM_LANES = N / PSUM_BEATS_PER_ROW;
    localparam int PSUM_BEATS = M * GROUPS * PSUM_BEATS_PER_ROW;

    logic clk = 0;  always #1.667 clk = ~clk;  // 300 MHz
    logic rst_n = 0;

    logic [255:0] a_data, w_data;
    logic         a_valid, w_valid;
    logic         a_ready, w_ready;

    logic [15:0]  ctrl_data;
    logic         ctrl_valid, ctrl_ready;

    logic [255:0] psum_data;
    logic         psum_valid, psum_ready;
    logic         psum_last;

    pe_array_axis_v5 dut (
        .ap_clk(clk), .ap_rst_n(rst_n),
        .s_axis_a_tdata(a_data), .s_axis_a_tvalid(a_valid), .s_axis_a_tready(a_ready),
        .s_axis_w_tdata(w_data), .s_axis_w_tvalid(w_valid), .s_axis_w_tready(w_ready),
        .s_axis_ctrl_tdata(ctrl_data), .s_axis_ctrl_tvalid(ctrl_valid), .s_axis_ctrl_tready(ctrl_ready),
        .m_axis_psum_tdata(psum_data), .m_axis_psum_tvalid(psum_valid), .m_axis_psum_tready(psum_ready),
        .m_axis_psum_tlast(psum_last)
    );

    integer a_vec  [0:K-1][0:M-1];
    integer w_vec  [0:K-1][0:N-1];
    integer golden [0:M-1][0:N-1];
    integer golden_group [0:GROUPS-1][0:M-1][0:N-1];
    integer errors = 0;

    task automatic reset_dut;
    begin
        psum_ready = 1;
        a_data = 256'd0;
        w_data = 256'd0;
        ctrl_data = 16'd0;
        a_valid = 0;
        w_valid = 0;
        ctrl_valid = 0;
        rst_n = 0;
        repeat (5) @(posedge clk);
        rst_n = 1;
        repeat (2) @(posedge clk);
    end
    endtask

    task automatic fill_zero;
    begin
        for (int k = 0; k < K; k++) begin
            for (int m = 0; m < M; m++) a_vec[k][m] = 0;
            for (int n = 0; n < N; n++) w_vec[k][n] = 0;
        end
    end
    endtask

    task automatic fill_one;
    begin
        for (int k = 0; k < K; k++) begin
            for (int m = 0; m < M; m++) a_vec[k][m] = 1;
            for (int n = 0; n < N; n++) w_vec[k][n] = 1;
        end
    end
    endtask

    task automatic fill_impulse;
    begin
        fill_zero();
        a_vec[7][5] = 3;
        w_vec[7][11] = -4;
    end
    endtask

    task automatic fill_large_product;
    begin
        fill_zero();
        a_vec[4][2] = 100;
        w_vec[4][3] = 100;
        a_vec[9][6] = -128;
        w_vec[9][7] = 127;
        a_vec[10][8] = -128;
        w_vec[10][9] = -127;
    end
    endtask

    task automatic fill_dense_single_k;
    begin
        fill_zero();
        for (int m = 0; m < M; m++) a_vec[0][m] = m - 16;
        for (int n = 0; n < N; n++) w_vec[0][n] = n - 8;
    end
    endtask

    task automatic fill_dense_multi_k;
    begin
        fill_zero();
        for (int k = 0; k < K; k++) begin
            for (int m = 0; m < M; m++) a_vec[k][m] = ((k + m) % 9) - 4;
            for (int n = 0; n < N; n++) w_vec[k][n] = ((2*k + n) % 11) - 5;
        end
    end
    endtask

    task automatic fill_random;
    begin
        for (int k = 0; k < K; k++) begin
            for (int m = 0; m < M; m++) begin
                logic signed [7:0] v;
                v = $urandom();
                a_vec[k][m] = v;
            end
            for (int n = 0; n < N; n++) begin
                logic signed [7:0] v;
                v = $urandom();
                w_vec[k][n] = v;
            end
        end
    end
    endtask

    task automatic compute_golden;
    begin
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++) begin
                golden[m][n] = 0;
                for (int g = 0; g < GROUPS; g++) golden_group[g][m][n] = 0;
                for (int k = 0; k < K; k++)
                    golden[m][n] += a_vec[k][m] * w_vec[k][n];
                for (int g = 0; g < GROUPS; g++)
                    for (int k = g * GROUP; k < (g + 1) * GROUP; k++)
                        golden_group[g][m][n] += a_vec[k][m] * w_vec[k][n];
            end
    end
    endtask

    task automatic send_compute_beat(
            input logic [255:0] a,
            input logic [255:0] w,
            input logic [15:0]  c);
    begin
        @(negedge clk);
        a_data = a;
        w_data = w;
        ctrl_data = c;
        a_valid = 1;
        w_valid = 1;
        ctrl_valid = 1;
        do @(posedge clk); while (!(a_ready & w_ready & ctrl_ready));
    end
    endtask

    task automatic run_compute_group(input int group);
        logic [255:0] a_beat;
        logic [255:0] w_beat;
        logic [15:0] ctrl;
    begin
        for (int kg = 0; kg < GROUP; kg++) begin
            int k = group * GROUP + kg;
            a_beat = 256'd0;
            w_beat = 256'd0;
            for (int m = 0; m < M; m++) a_beat[m*8 +: 8] = a_vec[k][m][7:0];
            for (int n = 0; n < N; n++) w_beat[n*8 +: 8] = w_vec[k][n][7:0];
            ctrl = 16'b0010;          // en
            if (kg == 0) ctrl[0] = 1'b1; // clear at each group start
            if (kg == GROUP - 1) begin
                ctrl[2] = 1'b1;       // commit final group compute beat
                if (k == K - 1)
                    ctrl[3] = 1'b1;   // final group drain should assert TLAST
            end
            send_compute_beat(a_beat, w_beat, ctrl);
        end

        @(negedge clk);
        a_valid = 0;
        w_valid = 0;
        ctrl_valid = 0;
    end
    endtask

    task automatic drain_and_check_group(input string case_name, input int out_group, input bit use_backpressure);
        int out_row;
        int out_half;
        int drain_cycle;
        int start_errors;
    begin
        out_row = 0;
        out_half = 0;
        drain_cycle = 0;
        start_errors = errors;
        ctrl_data  = 16'b0000;
        ctrl_valid = 0;

        while (out_row < M) begin
            psum_ready = use_backpressure ? ((drain_cycle % 7) != 3) : 1'b1;
            @(posedge clk);
            drain_cycle++;

            if (psum_valid && psum_ready) begin
                for (int lane = 0; lane < PSUM_LANES; lane++) begin
                    logic signed [31:0] got;
                    logic signed [31:0] exp;
                    int col;
                    col = out_half * PSUM_LANES + lane;
                    got = psum_data[lane*32 +: 32];
                    exp = golden_group[out_group][out_row][col];
                    if (got !== exp) begin
                        $error("[%s] PSUM mismatch group=%0d row=%0d col=%0d got=%0d exp=%0d",
                               case_name, out_group, out_row, col, got, exp);
                        errors++;
                    end
                end
                if (psum_last !== ((out_group == GROUPS-1) && (out_row == M-1) && (out_half == PSUM_BEATS_PER_ROW-1))) begin
                    $error("[%s] TLAST mismatch group=%0d row=%0d half=%0d got=%0b",
                           case_name, out_group, out_row, out_half, psum_last);
                    errors++;
                end
                if (out_half == PSUM_BEATS_PER_ROW - 1) begin
                    out_half = 0;
                    out_row++;
                end else begin
                    out_half++;
                end
            end
        end

        ctrl_valid = 0;
        psum_ready = 1;

        if (errors == start_errors)
            $display("[TB] PASS %s", case_name);
        else
            $fatal(1, "[TB] FAIL %s new_errors=%0d total_errors=%0d",
                   case_name, errors - start_errors, errors);
    end
    endtask

    task automatic run_case(input string case_name, input bit use_backpressure);
    begin
        $display("[TB] START %s", case_name);
        reset_dut();
        compute_golden();
        for (int g = 0; g < GROUPS; g++) begin
            run_compute_group(g);
            drain_and_check_group(case_name, g, use_backpressure);
        end
    end
    endtask

    initial begin
        fill_zero();
        run_case("all-zero", 1'b0);

        fill_one();
        run_case("all-one", 1'b0);

        fill_impulse();
        run_case("single-impulse-a5-w11", 1'b0);

        fill_large_product();
        run_case("large-product-sign-extension", 1'b0);

        fill_dense_single_k();
        run_case("dense-single-k", 1'b0);

        fill_dense_multi_k();
        run_case("dense-multi-k-small", 1'b0);

        fill_random();
        run_case("random-no-backpressure", 1'b0);

        fill_random();
        run_case("random-backpressure", 1'b1);

        $display("[TB] PASS: all directed/random PE array checks passed");
        $finish;
    end

    int beat_idx = 0;
    always_ff @(posedge clk)
        if (psum_valid && psum_ready) beat_idx <= beat_idx + 1;
endmodule
