`default_nettype none

// Phase-1 16x32 physical INT8 systolic array.
//
// The array exposes a 32x32 virtual INT8 GEMM shape because each physical PE
// accumulates two virtual A rows for one W column. Drain order is virtual
// row-major C/O words: even virtual row words 0..3, then odd virtual row words
// 0..3 for each physical row.
module systolic_array_16x32 #(
    parameter int SA_ROWS     = 16,
    parameter int SA_COLS     = 32,
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
    input  wire logic                         clear_acc_i,
    input  wire logic                         snapshot_acc_i,

    input  wire logic [SA_ROWS-1:0]           a_in_valid_i,
    input  wire logic [SA_ROWS-1:0][15:0]     a_in_data_i,

    input  wire logic [SA_COLS-1:0]           w_in_valid_i,
    input  wire logic [SA_COLS-1:0][15:0]     w_in_data_i,

    input  wire logic                         drain_start_i,
    output logic                         drain_valid_o,
    input  wire logic                         drain_ready_i,
    output logic [3:0]                   drain_row_id_o,
    output logic [2:0]                   drain_group_id_o,
    output logic [255:0]                 drain_data_o,
    output logic                         drain_done_o,
    output logic                         drain_busy_o,

    output logic                         unsupported_mode_error_o
);

    localparam int GROUP_SIZE  = 4;
    localparam int GROUPS      = SA_COLS / GROUP_SIZE;
    localparam int ROW_ID_W    = (SA_ROWS <= 1) ? 1 : $clog2(SA_ROWS);
    localparam int GROUP_ID_W  = (GROUPS  <= 1) ? 1 : $clog2(GROUPS);
    localparam logic [MODE_W-1:0] MODE_INT8_VALUE = MODE_INT8;
    localparam logic [ROW_ID_W-1:0] LAST_ROW_ID = SA_ROWS - 1;
    localparam logic [GROUP_ID_W-1:0] LAST_GROUP_ID = GROUPS - 1;

    initial begin
        if (SA_ROWS != 16) begin
            $error("systolic_array_16x32: SA_ROWS must remain 16 in Phase 1");
        end
        if (SA_COLS != 32) begin
            $error("systolic_array_16x32: SA_COLS must remain 32 in Phase 1");
        end
        if ((SA_COLS % GROUP_SIZE) != 0) begin
            $error("systolic_array_16x32: SA_COLS must be divisible by 4");
        end
    end

    logic int8_mode;
    logic drain_busy_q;
    logic drain_issuing_q;
    logic drain_insert_gap_q;
    logic [ROW_ID_W-1:0] drain_inject_row_q;
    logic [GROUP_ID_W-1:0] drain_inject_group_q;
    logic [7:0] drain_output_count_q;
    logic unsupported_mode_start;
    logic drain_advance;
    logic drain_output_fire;
    logic drain_inject_fire;

    logic [SA_ROWS-1:0][SA_COLS-1:0]       pe_a_valid;
    logic [SA_ROWS-1:0][SA_COLS-1:0][15:0] pe_a_data;
    logic [SA_ROWS-1:0][SA_COLS-1:0]       pe_w_valid;
    logic [SA_ROWS-1:0][SA_COLS-1:0][15:0] pe_w_data;
    logic [SA_ROWS-1:0][SA_COLS-1:0]       pe_a_valid_o;
    logic [SA_ROWS-1:0][SA_COLS-1:0][15:0] pe_a_data_o;
    logic [SA_ROWS-1:0][SA_COLS-1:0]       pe_w_valid_o;
    logic [SA_ROWS-1:0][SA_COLS-1:0][15:0] pe_w_data_o;
    logic [SA_ROWS-1:0][SA_COLS-1:0]       pe_w_valid_delayed;
    logic [SA_ROWS-1:0][SA_COLS-1:0][15:0] pe_w_data_delayed;
    logic [SA_ROWS-1:0]                     clear_acc_row_q;
    logic [SA_ROWS-1:0]                     snapshot_acc_row_q;
    logic [SA_ROWS-1:0][SA_COLS-1:0]        clear_acc_pe_q;
    logic [SA_ROWS-1:0][SA_COLS-1:0][63:0] pe_drain_data;
    logic [SA_ROWS-1:0][SA_COLS-1:0]       pe_error;
    logic [SA_ROWS-1:0][GROUPS-1:0][255:0] row_word_data;
    logic [SA_ROWS-1:0][255:0]             row_inject_data;
    logic [SA_ROWS-1:0]                    drain_pipe_valid_q;
    logic [SA_ROWS-1:0][255:0]             drain_pipe_data_q;
    logic [SA_ROWS-1:0][ROW_ID_W-1:0]      drain_pipe_row_q;
    logic [SA_ROWS-1:0][GROUP_ID_W-1:0]    drain_pipe_group_q;

    assign int8_mode = (mode_i == MODE_INT8_VALUE);
    assign unsupported_mode_start = (!int8_mode) &
        (step_fire_i | clear_acc_i | snapshot_acc_i | drain_start_i);
    assign drain_advance = drain_busy_q &&
        (!drain_pipe_valid_q[LAST_ROW_ID] || drain_ready_i);
    assign drain_output_fire = drain_busy_q &&
        drain_pipe_valid_q[LAST_ROW_ID] && drain_ready_i;
    assign drain_inject_fire = drain_issuing_q && !drain_insert_gap_q;

    genvar row;
    genvar col;
    generate
        for (row = 0; row < SA_ROWS; row = row + 1) begin : gen_rows
            always_ff @(posedge clk_i) begin
                if (rst_i) begin
                    clear_acc_row_q[row] <= 1'b0;
                    snapshot_acc_row_q[row] <= 1'b0;
                end else begin
                    clear_acc_row_q[row] <= clear_acc_i;
                    snapshot_acc_row_q[row] <= snapshot_acc_i;
                end
            end

            for (col = 0; col < SA_COLS; col = col + 1) begin : gen_cols
                if (col == 0) begin : gen_a_left_edge
                    assign pe_a_valid[row][col] = a_in_valid_i[row];
                    assign pe_a_data[row][col]  = a_in_data_i[row];
                end else begin : gen_a_from_left
                    assign pe_a_valid[row][col] = pe_a_valid_o[row][col-1];
                    assign pe_a_data[row][col]  = pe_a_data_o[row][col-1];
                end

                if (row == 0) begin : gen_w_top_edge
                    assign pe_w_valid[row][col] = w_in_valid_i[col];
                    assign pe_w_data[row][col]  = w_in_data_i[col];
                end else begin : gen_w_from_top
                    assign pe_w_valid[row][col] = pe_w_valid_delayed[row-1][col];
                    assign pe_w_data[row][col]  = pe_w_data_delayed[row-1][col];
                end

                pe_int8_packed_b #(
                    .MODE_W      ( MODE_W      ),
                    .MODE_INT8   ( MODE_INT8   ),
                    .INT8_ACC_W  ( INT8_ACC_W  ),
                    .PE_MAC_LAT  ( PE_MAC_LAT  ),
                    .K_BLOCK_MAX ( K_BLOCK_MAX )
                ) u_pe (
                    .clk_i                    ( clk_i                    ),
                    .rst_i                    ( rst_i                    ),
                    .step_fire_i              ( step_fire_i              ),
                    .mode_i                   ( mode_i                   ),
                    .a_valid_i                ( pe_a_valid[row][col]     ),
                    .a_data_i                 ( pe_a_data[row][col]      ),
                    .a_valid_o                ( pe_a_valid_o[row][col]   ),
                    .a_data_o                 ( pe_a_data_o[row][col]    ),
                    .w_valid_i                ( pe_w_valid[row][col]     ),
                    .w_data_i                 ( pe_w_data[row][col]      ),
                    .w_valid_o                ( pe_w_valid_o[row][col]   ),
                    .w_data_o                 ( pe_w_data_o[row][col]    ),
                    .clear_acc_i              ( clear_acc_pe_q[row][col] ),
                    .snapshot_acc_i           ( snapshot_acc_row_q[row]  ),
                    .drain_en_i               ( 1'b0                     ),
                    .drain_valid_o            (                          ),
                    .drain_data_o             ( pe_drain_data[row][col]  ),
                    .unsupported_mode_error_o ( pe_error[row][col]       )
                );

                always_ff @(posedge clk_i) begin
                    if (rst_i) begin
                        pe_w_valid_delayed[row][col] <= 1'b0;
                        clear_acc_pe_q[row][col] <= 1'b0;
                    end else begin
                        pe_w_valid_delayed[row][col] <= step_fire_i && pe_w_valid_o[row][col];
                        pe_w_data_delayed[row][col]  <= pe_w_data_o[row][col];
                        clear_acc_pe_q[row][col] <= clear_acc_row_q[row];
                    end
                end
            end
        end
    endgenerate

    genvar group;
    generate
        for (row = 0; row < SA_ROWS; row = row + 1) begin : gen_drain_rows
            for (group = 0; group < GROUPS; group = group + 1) begin : gen_drain_groups
                if (group < (GROUPS / 2)) begin : gen_even_virtual_row
                    assign row_word_data[row][group] = {
                        pe_drain_data[row][(group*8)+7][31:0],
                        pe_drain_data[row][(group*8)+6][31:0],
                        pe_drain_data[row][(group*8)+5][31:0],
                        pe_drain_data[row][(group*8)+4][31:0],
                        pe_drain_data[row][(group*8)+3][31:0],
                        pe_drain_data[row][(group*8)+2][31:0],
                        pe_drain_data[row][(group*8)+1][31:0],
                        pe_drain_data[row][(group*8)+0][31:0]
                    };
                end else begin : gen_odd_virtual_row
                    assign row_word_data[row][group] = {
                        pe_drain_data[row][((group-(GROUPS/2))*8)+7][63:32],
                        pe_drain_data[row][((group-(GROUPS/2))*8)+6][63:32],
                        pe_drain_data[row][((group-(GROUPS/2))*8)+5][63:32],
                        pe_drain_data[row][((group-(GROUPS/2))*8)+4][63:32],
                        pe_drain_data[row][((group-(GROUPS/2))*8)+3][63:32],
                        pe_drain_data[row][((group-(GROUPS/2))*8)+2][63:32],
                        pe_drain_data[row][((group-(GROUPS/2))*8)+1][63:32],
                        pe_drain_data[row][((group-(GROUPS/2))*8)+0][63:32]
                    };
                end
            end

            assign row_inject_data[row] = row_word_data[row][drain_inject_group_q];
        end
    endgenerate

    always_ff @(posedge clk_i) begin
        if (rst_i) begin
            drain_busy_q         <= 1'b0;
            drain_issuing_q      <= 1'b0;
            drain_insert_gap_q   <= 1'b0;
            drain_inject_row_q   <= '0;
            drain_inject_group_q <= '0;
            drain_output_count_q <= '0;
            drain_pipe_valid_q   <= '0;
            for (int pipe_idx = 0; pipe_idx < SA_ROWS; pipe_idx++) begin
                drain_pipe_data_q[pipe_idx]  <= '0;
                drain_pipe_row_q[pipe_idx]   <= '0;
                drain_pipe_group_q[pipe_idx] <= '0;
            end
            drain_done_o         <= 1'b0;
        end else begin
            drain_done_o <= 1'b0;

            if (drain_start_i & int8_mode & !drain_busy_q) begin
                drain_busy_q         <= 1'b1;
                drain_issuing_q      <= 1'b1;
                drain_insert_gap_q   <= 1'b0;
                drain_inject_row_q   <= '0;
                drain_inject_group_q <= '0;
                drain_output_count_q <= '0;
                drain_pipe_valid_q   <= '0;
            end else if (drain_advance) begin
                for (int pipe_idx = SA_ROWS - 1; pipe_idx >= 0; pipe_idx--) begin
                    if (pipe_idx == 0) begin
                        drain_pipe_valid_q[pipe_idx] <= 1'b0;
                        drain_pipe_data_q[pipe_idx]  <= '0;
                        drain_pipe_row_q[pipe_idx]   <= '0;
                        drain_pipe_group_q[pipe_idx] <= '0;
                    end else begin
                        drain_pipe_valid_q[pipe_idx] <= drain_pipe_valid_q[pipe_idx-1];
                        drain_pipe_data_q[pipe_idx]  <= drain_pipe_data_q[pipe_idx-1];
                        drain_pipe_row_q[pipe_idx]   <= drain_pipe_row_q[pipe_idx-1];
                        drain_pipe_group_q[pipe_idx] <= drain_pipe_group_q[pipe_idx-1];
                    end
                end

                if (drain_inject_fire) begin
                    drain_pipe_valid_q[drain_inject_row_q] <= 1'b1;
                    drain_pipe_data_q[drain_inject_row_q]  <= row_inject_data[drain_inject_row_q];
                    drain_pipe_row_q[drain_inject_row_q]   <= drain_inject_row_q;
                    drain_pipe_group_q[drain_inject_row_q] <= drain_inject_group_q;
                end

                if (drain_issuing_q) begin
                    if (drain_insert_gap_q) begin
                        drain_insert_gap_q   <= 1'b0;
                        drain_inject_group_q <= '0;
                        drain_inject_row_q   <= drain_inject_row_q + 1'b1;
                    end else if (drain_inject_group_q == LAST_GROUP_ID) begin
                        if (drain_inject_row_q == LAST_ROW_ID) begin
                            drain_issuing_q <= 1'b0;
                        end else begin
                            drain_insert_gap_q <= 1'b1;
                        end
                    end else begin
                        drain_inject_group_q <= drain_inject_group_q + 1'b1;
                    end
                end

                if (drain_output_fire) begin
                    if (drain_output_count_q == 8'd127) begin
                        drain_busy_q         <= 1'b0;
                        drain_issuing_q      <= 1'b0;
                        drain_insert_gap_q   <= 1'b0;
                        drain_inject_row_q   <= '0;
                        drain_inject_group_q <= '0;
                        drain_output_count_q <= '0;
                        drain_done_o         <= 1'b1;
                    end else begin
                        drain_output_count_q <= drain_output_count_q + 8'd1;
                    end
                end else begin
                    drain_output_count_q <= drain_output_count_q;
                end
            end
        end
    end

    always_comb begin
        logic pe_error_or;
        pe_error_or = 1'b0;

        for (int r = 0; r < SA_ROWS; r = r + 1) begin
            for (int c = 0; c < SA_COLS; c = c + 1) begin
                pe_error_or |= pe_error[r][c];
            end
        end

        unsupported_mode_error_o = unsupported_mode_start | pe_error_or;
    end

    assign drain_valid_o    = drain_pipe_valid_q[LAST_ROW_ID];
    assign drain_busy_o     = drain_busy_q;
    assign drain_row_id_o   = drain_pipe_row_q[LAST_ROW_ID][3:0];
    assign drain_group_id_o = drain_pipe_group_q[LAST_ROW_ID][2:0];
    assign drain_data_o     = drain_pipe_data_q[LAST_ROW_ID];

endmodule

`default_nettype wire
