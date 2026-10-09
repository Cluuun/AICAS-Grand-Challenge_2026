module layernorm_add #(
    parameter DATA_NUM   = 32,
    parameter ADD_BLOCK  = 16,
    parameter DATA_WIDTH = 10,
    parameter SUM_WIDTH  = 18
) (
    input  logic                         clk_i  ,
    input  logic                         rstn_i ,
    input  logic                         valid_i,
    input  logic signed [DATA_WIDTH-1:0] data_i [DATA_NUM-1:0],

    output logic                         valid_o,
    output logic signed [SUM_WIDTH-1:0]  data_o 
);

    // 中间变量：分块求和结果
    logic signed [SUM_WIDTH-1:0] block_sum   [DATA_NUM/ADD_BLOCK-1:0];
    logic signed [SUM_WIDTH-1:0] block_sum_q [DATA_NUM/ADD_BLOCK-1:0];

    logic signed [SUM_WIDTH-1:0] total_sum;
    logic signed [SUM_WIDTH-1:0] total_sum_q;

    logic valid_block;
    logic valid_total;

    // === 分块求和 ===
    always_comb begin
        for (int b = 0; b < DATA_NUM/ADD_BLOCK; b++) begin
            block_sum[b] = '0;
            for (int j = 0; j < ADD_BLOCK; j++) begin
                block_sum[b] += $signed(data_i[b*ADD_BLOCK + j]);
            end
        end
    end

    // === 第一级寄存：保存 block_sum ===
    always_ff @(posedge clk_i or negedge rstn_i) begin
        if (!rstn_i) begin
            for (int b = 0; b < DATA_NUM/ADD_BLOCK; b++) begin
                block_sum_q[b] <= '0;
            end
            valid_block <= 1'b0;
        end else begin
            if (valid_i) begin
                for (int b = 0; b < DATA_NUM/ADD_BLOCK; b++) begin
                    block_sum_q[b] <= block_sum[b];
                end
            end
            valid_block <= valid_i;
        end
    end

    // === 第二级组合逻辑：累加所有 block_sum ===
    always_comb begin
        total_sum = '0;
        for (int b = 0; b < DATA_NUM/ADD_BLOCK; b++) begin
            total_sum += block_sum_q[b];
        end
    end

    // === 第二级寄存：输出 total_sum ===
    always_ff @(posedge clk_i or negedge rstn_i) begin
        if (!rstn_i) begin
            total_sum_q <= '0;
            valid_total <= 1'b0;
        end else begin
            if (valid_block) begin
                total_sum_q <= total_sum;
            end
            valid_total <= valid_block;
        end
    end

    assign data_o  = total_sum_q;
    assign valid_o = valid_total;

endmodule
