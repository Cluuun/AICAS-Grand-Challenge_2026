// tb_pe_array_axis_v6.sv --- self-checking testbench for pe_array_axis_v6.
`timescale 1ns/1ps
module tb_pe_array_axis_v6;
    localparam int K = 128;
    localparam int M = 8;
    localparam int N = 128;
    localparam int PSUM_BEATS_PER_ROW = 16;
    localparam int PSUM_LANES = 8;

    logic clk = 0; always #1.667 clk = ~clk;
    logic rst_n = 0;

    logic [63:0] a_data;
    logic [511:0] w0_data;
    logic [511:0] w1_data;
    logic a_valid, w0_valid, w1_valid, a_ready, w0_ready, w1_ready;
    logic [15:0] ctrl_data;
    logic ctrl_valid, ctrl_ready;
    logic [255:0] psum_data;
    logic psum_valid, psum_ready, psum_last;

    pe_array_axis_v6 dut (
        .ap_clk(clk), .ap_rst_n(rst_n),
        .s_axis_a_tdata(a_data), .s_axis_a_tvalid(a_valid), .s_axis_a_tready(a_ready),
        .s_axis_w0_tdata(w0_data), .s_axis_w0_tvalid(w0_valid), .s_axis_w0_tready(w0_ready),
        .s_axis_w1_tdata(w1_data), .s_axis_w1_tvalid(w1_valid), .s_axis_w1_tready(w1_ready),
        .s_axis_ctrl_tdata(ctrl_data), .s_axis_ctrl_tvalid(ctrl_valid), .s_axis_ctrl_tready(ctrl_ready),
        .m_axis_psum_tdata(psum_data), .m_axis_psum_tvalid(psum_valid), .m_axis_psum_tready(psum_ready),
        .m_axis_psum_tlast(psum_last)
    );

    integer a_vec [0:K-1][0:M-1];
    integer w_vec [0:K-1][0:N-1];
    integer golden [0:M-1][0:N-1];
    integer errors = 0;

    task automatic reset_dut;
    begin
        psum_ready = 1;
        a_data = '0;
        w0_data = '0;
        w1_data = '0;
        ctrl_data = '0;
        a_valid = 0;
        w0_valid = 0;
        w1_valid = 0;
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
        w_vec[7][111] = -4;
    end
    endtask

    task automatic fill_dense_multi_k;
    begin
        for (int k = 0; k < K; k++) begin
            for (int m = 0; m < M; m++) a_vec[k][m] = ((k + m) % 9) - 4;
            for (int n = 0; n < N; n++) w_vec[k][n] = ((2*k + n) % 11) - 5;
        end
    end
    endtask

    task automatic fill_decode_row0;
    begin
        fill_zero();
        for (int k = 0; k < K; k++) begin
            a_vec[k][0] = ((k * 3) % 13) - 6;
            for (int n = 0; n < N; n++) w_vec[k][n] = ((k + n * 5) % 17) - 8;
        end
    end
    endtask

    task automatic compute_golden;
    begin
        for (int m = 0; m < M; m++) begin
            for (int n = 0; n < N; n++) begin
                golden[m][n] = 0;
                for (int k = 0; k < K; k++) begin
                    golden[m][n] += a_vec[k][m] * w_vec[k][n];
                end
            end
        end
    end
    endtask

    task automatic send_compute_beat(input logic [63:0] a,
                                     input logic [511:0] w0,
                                     input logic [511:0] w1,
                                     input logic [15:0] c);
    begin
        @(negedge clk);
        a_data = a;
        w0_data = w0;
        w1_data = w1;
        ctrl_data = c;
        a_valid = 1;
        w0_valid = 1;
        w1_valid = 1;
        ctrl_valid = 1;
        do @(posedge clk); while (!(a_ready & w0_ready & w1_ready & ctrl_ready));
    end
    endtask

    task automatic run_compute_group;
        logic [63:0] a_beat;
        logic [511:0] w0_beat;
        logic [511:0] w1_beat;
        logic [15:0] ctrl;
    begin
        for (int k = 0; k < K; k++) begin
            a_beat = '0;
            w0_beat = '0;
            w1_beat = '0;
            for (int m = 0; m < M; m++) a_beat[m*8 +: 8] = a_vec[k][m][7:0];
            for (int n = 0; n < 64; n++) w0_beat[n*8 +: 8] = w_vec[k][n][7:0];
            for (int n = 64; n < N; n++) w1_beat[(n-64)*8 +: 8] = w_vec[k][n][7:0];
            ctrl = 16'b0010;
            if (k == 0) ctrl[0] = 1'b1;
            if (k == K - 1) begin
                ctrl[2] = 1'b1;
                ctrl[3] = 1'b1;
            end
            send_compute_beat(a_beat, w0_beat, w1_beat, ctrl);
        end
        @(negedge clk);
        a_valid = 0;
        w0_valid = 0;
        w1_valid = 0;
        ctrl_valid = 0;
    end
    endtask

    task automatic drain_and_check(input string case_name, input bit use_backpressure);
        int row;
        int seg;
        int cycle;
        int start_errors;
    begin
        row = 0;
        seg = 0;
        cycle = 0;
        start_errors = errors;
        while (row < M) begin
            psum_ready = use_backpressure ? ((cycle % 7) != 3) : 1'b1;
            @(posedge clk);
            cycle++;
            if (psum_valid && psum_ready) begin
                for (int lane = 0; lane < PSUM_LANES; lane++) begin
                    int col = seg * PSUM_LANES + lane;
                    logic signed [31:0] got = psum_data[lane*32 +: 32];
                    logic signed [31:0] exp = golden[row][col];
                    if (got !== exp) begin
                        $error("[%s] row=%0d col=%0d got=%0d exp=%0d",
                               case_name, row, col, got, exp);
                        errors++;
                    end
                end
                if (psum_last !== ((row == M-1) && (seg == PSUM_BEATS_PER_ROW-1))) begin
                    $error("[%s] TLAST mismatch row=%0d seg=%0d got=%0b",
                           case_name, row, seg, psum_last);
                    errors++;
                end
                if (seg == PSUM_BEATS_PER_ROW - 1) begin
                    seg = 0;
                    row++;
                end else begin
                    seg++;
                end
            end
        end
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
        run_compute_group();
        drain_and_check(case_name, use_backpressure);
    end
    endtask

    initial begin
        fill_zero();          run_case("zero", 0);
        fill_one();           run_case("ones", 0);
        fill_impulse();       run_case("impulse", 0);
        fill_dense_multi_k(); run_case("dense-backpressure", 1);
        fill_decode_row0();   run_case("decode-row0-only", 1);
        if (errors == 0) begin
            $display("[TB] ALL PASS pe_array_axis_v6");
            $finish;
        end
        $fatal(1, "[TB] FAIL total_errors=%0d", errors);
    end
endmodule
