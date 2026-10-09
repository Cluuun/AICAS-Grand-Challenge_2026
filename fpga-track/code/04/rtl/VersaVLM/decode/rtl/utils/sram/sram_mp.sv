

module sram_mp #(
    // TYPE = 0 Register, TYPE = 1 SRAM
    parameter TYPE=0,
    parameter SIZE=64,
    parameter DATA_WIDTH=8,
    parameter RD_PORTS=1,
    parameter WR_PORTS=1,
    parameter RESETVAL=0,
    parameter RESETABLE=0
) (
    input   wire                                   clk          ,
    input   wire                                   rst_n        ,
    //Write Port
    input   wire  [WR_PORTS-1:0]                   wr_en        ,
    input   wire  [WR_PORTS*$clog2(SIZE)-1:0]      write_address,
    input   wire  [WR_PORTS*DATA_WIDTH-1:0]        new_data     ,
    //Read Port
    input   wire  [RD_PORTS-1:0]                   rd_en        ,
    input   wire  [RD_PORTS*$clog2(SIZE)-1:0]      read_address ,
    output  reg   [RD_PORTS*DATA_WIDTH-1:0]        data_out
);

    localparam SEL_BITS = $clog2(SIZE); 
	// #Internal Signals#
	(* ram_style = "block" *) reg  [DATA_WIDTH-1:0] Memory_Array [SIZE-1:0];

	//Push the Data out
    generate
        if (TYPE) begin
            always @(posedge clk or negedge rst_n) begin : DataOUT
                if (!rst_n) begin
                    data_out <= 'b0;
                end
                else begin
                    for (integer i = 0; i < RD_PORTS; i=i+1) begin
                        if (rd_en[i]) begin
                            data_out[i*DATA_WIDTH+:DATA_WIDTH] <= Memory_Array[read_address[i*SEL_BITS+:SEL_BITS]];
                        end
                    end
                end
            end
        end
        else begin
            always @(*) begin : DataOUT
                for (integer i = 0; i < RD_PORTS; i=i+1) begin
                    data_out[i*DATA_WIDTH+:DATA_WIDTH] = Memory_Array[read_address[i*SEL_BITS+:SEL_BITS]];
                end
            end
        end

    endgenerate

    generate
        if(RESETABLE) begin
            //Create Resetable SRAM
            always @(posedge clk or negedge rst_n) begin : Update
                if(!rst_n) begin
                    for (integer i = 0; i <SIZE; i=i+1) begin
                        Memory_Array[i] <= RESETVAL;
                    end
                end else begin
                    for (integer i = 0; i < WR_PORTS; i=i+1) begin
                        if (wr_en[i]) 
                            Memory_Array[write_address[i*SEL_BITS+:SEL_BITS]] <= new_data[i*DATA_WIDTH+:DATA_WIDTH];
                    end
                end
            end
        end else begin
            //Create Non-Resetable SRAM
            always @(posedge clk) begin : Update
                for (integer i = 0; i < WR_PORTS; i=i+1) begin
                    if (wr_en[i]) 
                        Memory_Array[write_address[i*SEL_BITS+:SEL_BITS]] <= new_data[i*DATA_WIDTH+:DATA_WIDTH];
                end
            end
        end
    endgenerate

endmodule