`default_nettype none

// Phase-1 direct 32x32 systolic array.
//
// Each physical PE maps to one logical output element. Drain order is row-major
// C/O words: for each row, four 256-bit groups of eight INT32 lanes.
module systolic_array_32x32 #(
    parameter int SA_ROWS     = 32,
    parameter int SA_COLS     = 32,
    parameter int MODE_W      = 2,
    parameter int MODE_INT8   = 0,
    parameter int MODE_PV_LOG8_U16I8 = 2,
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
    input  wire logic [SA_COLS-1:0][7:0]      w_in_data_i,

    input  wire logic                         drain_start_i,
    output logic                              drain_valid_o,
    input  wire logic                         drain_ready_i,
    output logic [4:0]                        drain_row_id_o,
    output logic [1:0]                        drain_group_id_o,
    output logic [255:0]                      drain_data_o,
    output logic                              drain_done_o,
    output logic                              drain_busy_o,

    output logic                              unsupported_mode_error_o
);

    localparam int GROUP_SIZE = 8;
    localparam int GROUPS = SA_COLS / GROUP_SIZE;
    localparam int DRAIN_CLUSTER_ROWS = 8;
    localparam int DRAIN_CLUSTERS = SA_ROWS / DRAIN_CLUSTER_ROWS;
    localparam int ROW_ID_W = (SA_ROWS <= 1) ? 1 : $clog2(SA_ROWS);
    localparam int GROUP_ID_W = (GROUPS <= 1) ? 1 : $clog2(GROUPS);
    localparam int CLUSTER_ID_W = (DRAIN_CLUSTERS <= 1) ? 1 : $clog2(DRAIN_CLUSTERS);
    localparam logic [MODE_W-1:0] MODE_INT8_VALUE = MODE_INT8;
    localparam logic [MODE_W-1:0] MODE_PV_LOG8_U16I8_VALUE = MODE_PV_LOG8_U16I8;
    localparam logic [ROW_ID_W-1:0] LAST_ROW_ID = SA_ROWS - 1;
    localparam logic [GROUP_ID_W-1:0] LAST_GROUP_ID = GROUPS - 1;

    initial begin
        if (SA_ROWS != 32) begin
            $error("systolic_array_32x32: SA_ROWS must be 32");
        end
        if (SA_COLS != 32) begin
            $error("systolic_array_32x32: SA_COLS must be 32");
        end
        if ((SA_COLS % GROUP_SIZE) != 0) begin
            $error("systolic_array_32x32: SA_COLS must be divisible by 8");
        end
        if ((SA_ROWS % DRAIN_CLUSTER_ROWS) != 0) begin
            $error("systolic_array_32x32: SA_ROWS must be divisible by drain cluster rows");
        end
    end

    logic int8_mode;
    logic pv_mode;
    logic mac_mode;
    logic drain_scan_active_q;
    logic [ROW_ID_W-1:0] drain_scan_row_q;
    logic [GROUP_ID_W-1:0] drain_scan_group_q;
    logic drain_stage1_valid_q;
    logic [CLUSTER_ID_W-1:0] drain_stage1_cluster_q;
    logic [ROW_ID_W-1:0] drain_stage1_row_q;
    logic [GROUP_ID_W-1:0] drain_stage1_group_q;
    logic drain_stage1_last_q;
    (* keep = "true" *) logic [DRAIN_CLUSTERS-1:0][255:0] drain_cluster_data_q;
    logic drain_out_valid_q;
    logic [ROW_ID_W-1:0] drain_out_row_q;
    logic [GROUP_ID_W-1:0] drain_out_group_q;
    logic drain_out_last_q;
    (* keep = "true" *) logic [255:0] drain_out_data_q;
    logic unsupported_mode_start;
    logic drain_fire;
    logic drain_pipe_advance;
    logic drain_idle;

    logic [SA_ROWS-1:0][SA_COLS-1:0]       pe_a_valid;
    logic [SA_ROWS-1:0][SA_COLS-1:0][15:0] pe_a_data;
    logic [SA_ROWS-1:0][SA_COLS-1:0]       pe_w_valid;
    logic [SA_ROWS-1:0][SA_COLS-1:0][7:0]  pe_w_data;
    logic [SA_ROWS-1:0][SA_COLS-1:0]       pe_a_valid_o;
    logic [SA_ROWS-1:0][SA_COLS-1:0][15:0] pe_a_data_o;
    logic [SA_ROWS-1:0][SA_COLS-1:0]       pe_w_valid_o;
    logic [SA_ROWS-1:0][SA_COLS-1:0][7:0]  pe_w_data_o;
    logic [SA_ROWS-1:0]                    clear_acc_row_q;
    logic [SA_ROWS-1:0]                    snapshot_acc_row_q;
    logic [SA_ROWS-1:0][SA_COLS-1:0]       clear_acc_pe_q;
    logic [SA_ROWS-1:0][SA_COLS-1:0][31:0] pe_drain_data;
    logic [SA_ROWS-1:0][GROUPS-1:0][255:0] row_word_data;

    assign int8_mode = (mode_i == MODE_INT8_VALUE);
    assign pv_mode = (mode_i == MODE_PV_LOG8_U16I8_VALUE);
    assign mac_mode = int8_mode || pv_mode;
    assign unsupported_mode_start = (!mac_mode) &
        (step_fire_i | clear_acc_i | snapshot_acc_i | drain_start_i);
    assign drain_fire = drain_valid_o && drain_ready_i;
    assign drain_pipe_advance = !drain_out_valid_q || drain_ready_i;
    assign drain_idle = !drain_scan_active_q && !drain_stage1_valid_q && !drain_out_valid_q;

    genvar row;
    genvar col;
    genvar group;
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
                    assign pe_w_valid[row][col] = pe_w_valid_o[row-1][col];
                    assign pe_w_data[row][col]  = pe_w_data_o[row-1][col];
                end

                pe_int8_single #(
                    .INT8_ACC_W  ( INT8_ACC_W  ),
                    .PE_MAC_LAT  ( PE_MAC_LAT  ),
                    .K_BLOCK_MAX ( K_BLOCK_MAX )
                ) u_pe (
                    .clk_i                    ( clk_i                    ),
                    .rst_i                    ( rst_i                    ),
                    .step_fire_i              ( step_fire_i              ),
                    .mac_mode_i               ( mac_mode                 ),
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
                    .drain_data_o             ( pe_drain_data[row][col]  )
                );

                always_ff @(posedge clk_i) begin
                    if (rst_i) begin
                        clear_acc_pe_q[row][col] <= 1'b0;
                    end else begin
                        clear_acc_pe_q[row][col] <= clear_acc_row_q[row];
                    end
                end
            end

            for (group = 0; group < GROUPS; group = group + 1) begin : gen_drain_groups
                assign row_word_data[row][group] = {
                    pe_drain_data[row][group*8 + 7],
                    pe_drain_data[row][group*8 + 6],
                    pe_drain_data[row][group*8 + 5],
                    pe_drain_data[row][group*8 + 4],
                    pe_drain_data[row][group*8 + 3],
                    pe_drain_data[row][group*8 + 2],
                    pe_drain_data[row][group*8 + 1],
                    pe_drain_data[row][group*8 + 0]
                };
            end
        end
    endgenerate

    always_ff @(posedge clk_i) begin
        if (rst_i) begin
            drain_scan_active_q <= 1'b0;
            drain_scan_row_q <= '0;
            drain_scan_group_q <= '0;
            drain_stage1_valid_q <= 1'b0;
            drain_stage1_cluster_q <= '0;
            drain_stage1_row_q <= '0;
            drain_stage1_group_q <= '0;
            drain_stage1_last_q <= 1'b0;
            drain_out_valid_q <= 1'b0;
            drain_out_row_q <= '0;
            drain_out_group_q <= '0;
            drain_out_last_q <= 1'b0;
            drain_out_data_q <= '0;
            drain_done_o <= 1'b0;
            for (int cluster_idx = 0; cluster_idx < DRAIN_CLUSTERS; cluster_idx++) begin
                drain_cluster_data_q[cluster_idx] <= '0;
            end
        end else begin
            drain_done_o <= 1'b0;

            if (drain_fire && drain_out_last_q) begin
                drain_done_o <= 1'b1;
            end

            if (drain_pipe_advance) begin
                for (int cluster_idx = 0; cluster_idx < DRAIN_CLUSTERS; cluster_idx++) begin
                    drain_cluster_data_q[cluster_idx] <=
                        row_word_data[cluster_idx * DRAIN_CLUSTER_ROWS +
                                      drain_scan_row_q[2:0]][drain_scan_group_q];
                end

                drain_out_valid_q <= drain_stage1_valid_q;
                drain_out_row_q <= drain_stage1_row_q;
                drain_out_group_q <= drain_stage1_group_q;
                drain_out_last_q <= drain_stage1_last_q;
                drain_out_data_q <= drain_cluster_data_q[drain_stage1_cluster_q];

                if (drain_start_i && mac_mode && drain_idle) begin
                    // PE snapshot is row-registered, so begin scanning one cycle after
                    // drain_start instead of sampling the previous shadow contents.
                    drain_stage1_valid_q <= 1'b0;
                    drain_stage1_last_q <= 1'b0;
                    drain_scan_active_q <= 1'b1;
                    drain_scan_row_q <= '0;
                    drain_scan_group_q <= '0;
                end else if (drain_scan_active_q) begin
                    drain_stage1_valid_q <= 1'b1;
                    drain_stage1_cluster_q <= drain_scan_row_q[ROW_ID_W-1:3];
                    drain_stage1_row_q <= drain_scan_row_q;
                    drain_stage1_group_q <= drain_scan_group_q;
                    drain_stage1_last_q <= (drain_scan_row_q == LAST_ROW_ID) &&
                                           (drain_scan_group_q == LAST_GROUP_ID);
                    if ((drain_scan_row_q == LAST_ROW_ID) &&
                        (drain_scan_group_q == LAST_GROUP_ID)) begin
                        drain_scan_active_q <= 1'b0;
                        drain_scan_row_q <= '0;
                        drain_scan_group_q <= '0;
                    end else if (drain_scan_group_q == LAST_GROUP_ID) begin
                        drain_scan_group_q <= '0;
                        drain_scan_row_q <= drain_scan_row_q + ROW_ID_W'(1);
                    end else begin
                        drain_scan_group_q <= drain_scan_group_q + GROUP_ID_W'(1);
                    end
                end else begin
                    drain_stage1_valid_q <= 1'b0;
                    drain_stage1_last_q <= 1'b0;
                end
            end
        end
    end

    assign unsupported_mode_error_o = unsupported_mode_start;

    assign drain_valid_o = drain_out_valid_q;
    assign drain_busy_o = drain_scan_active_q || drain_stage1_valid_q || drain_out_valid_q;
    assign drain_row_id_o = drain_out_row_q[4:0];
    assign drain_group_id_o = drain_out_group_q[1:0];
    assign drain_data_o = drain_out_data_q;

endmodule

`default_nettype wire
