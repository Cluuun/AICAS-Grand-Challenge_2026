module pe_os_fpga #(
  parameter DATA_WIDTH_IN   = 8, 
  parameter DATA_WIDTH_OUT  = 32 
) (
  input  logic                      clk_i,
  input  logic                      rstn_i,
  input  logic                      clear_i,   
  input  logic signed [DATA_WIDTH_IN-1:0]   weight_i,
  input  logic signed [DATA_WIDTH_IN-1:0]   active_i,
  
  output logic signed [DATA_WIDTH_IN-1:0]   weight_o,
  output logic signed [DATA_WIDTH_IN-1:0]   active_o,
  
  // 1. 将 use_dsp 属性直接放在变量定义处，强制性更强
  (* use_dsp = "yes" *) output logic signed [DATA_WIDTH_OUT-1:0]  mac_res_o
);

  //-------------------------------------------------------------------------
  // 1. 数据传递逻辑 (Data Flow)
  //    这部分逻辑通常使用普通寄存器 (FF)
  //-------------------------------------------------------------------------
  always_ff @(posedge clk_i) begin
    if (!rstn_i) begin
      active_o <= '0;
      weight_o <= '0;
    end else begin
      active_o <= active_i;
      weight_o <= weight_i; 
    end
  end

  //-------------------------------------------------------------------------
  // 2. 乘加逻辑 (MAC) -> 映射为 DSP
  //    注意：
  //    1. 使用 always_ff 明确表示时序逻辑
  //    2. 乘法和加法写在同一个表达式中，有助于工具推断 MAC 结构
  //    3. 确保 rstn_i 是同步复位（不在敏感列表中包含 negedge）
  //-------------------------------------------------------------------------
  
  // 建议：定义一个中间变量辅助理解，虽然写在一起也可以
  // logic signed [2*DATA_WIDTH_IN-1:0] product;
  // assign product = active_i * weight_i;

  always_ff @(posedge clk_i) begin
    // 优先级：同步复位 > 同步清零 > 累加
    // DSP48 原生支持同步复位 (RST)
    if (!rstn_i) begin
      mac_res_o <= '0;
    end 
    else if (clear_i) begin
      mac_res_o <= '0;
    end 
    else begin
      // 这里的写法符合 P = P + A * B 的 DSP 结构
      mac_res_o <= mac_res_o + (active_i * weight_i);
    end
  end

endmodule