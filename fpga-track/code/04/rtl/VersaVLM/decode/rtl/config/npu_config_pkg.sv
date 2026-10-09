// define NPU parameters

package npu_config_pkg;

    // Config RegisterFile
    parameter RF_DATA_WIDTH         = 64    ;
    //parameter RF_IDX_WIDTH          = 5     ;


    //MMIO AXI-LITE config
    parameter   MMIO_AXI_DATA_WIDTH = 64;
    parameter   MMIO_AXI_ADDR_WIDTH = 32;
    parameter   MMIO_REG_NUMBER     = 32;    //      wo           wr             ro
    parameter   MMIO_REG_WOEND      = 23;    //   0--------woend--------rostart--------number 
    parameter   MMIO_REG_ROSTART    = 26;    //   woend is the last WO reg addr-bias/8,rostart is the first RO reg addr-bias/8
    parameter   MMIO_INTERRUPT_SRC  = 8 ;
    parameter   MMIO_BASE_ADDR      = 32'h0000_0000;
    parameter   MMIO_MAX_ADDR       = 32'h0010_0000;

    //DMA AXI config
    parameter    AXI_ID_WIDTH	    = 4   ;    // Thread ID Width
	parameter    AXI_ADDR_WIDTH	    = 32  ;     // Width of Address Bus
	parameter    AXI_DATA_WIDTH  	= 128 ;    // Width of Data Bus
    parameter    DMA_AXI_PORTS      = 4   ;
    //DRAM config
	parameter    AXI_WR_OUTSTANDING  = 32'd16         ;
	parameter    AXI_RD_OUTSTANDING  = 32'd16         ;
    parameter    ADDR_BASE           = 32'd0          ;
    parameter    DRAMType            = "DDR4"         ;
    parameter    CustomerDRAM        = "none"         ;
    parameter    ClkPeriod           = 1ns            ;
    //if test npu only:
    parameter InitPath        = "./rtl/tb" ;
    // if test npu+zerocore
    //parameter    InitPath        = "../rtl/tb/NPU/rtl/tb" ;

    //quant config
    parameter SCALE_DATA_WIDTH = RF_DATA_WIDTH/4 ;

    //sfu config 
    parameter FX_DATA_WIDTH             = 16;
    parameter SOFTMAX_FRAC_WIDTH        = 10; // frac width before softmax
    parameter GELU_FRAC_WIDTH           = 10;// frac width before/after gelu
    parameter LAYERNORM_FRAC_WIDTH      = 10; // frac width before/after layernorm
    parameter SFU_MAX_INPUT_LENGTH      = 1024; //for layernorm and softmax
    parameter TRANSPOSE_MAX_LENGTH      = 65536;
    parameter GELU_NUM                  = 4;

   
    // Systolic Array Config
    parameter PE_WIDTH              = 16    ;       // number of PEs in a row/column of systolic array
    parameter PE_DATA_WIDTH         = 8    ;
    parameter PE_FPGA_DSP           = 0    ;

    parameter PE_DATA_WIDTH_IN  = 8;
    parameter PE_DATA_WIDTH_OUT = 32;
    parameter INPUT_WIDTH_MAX   = 4096;
    parameter INPUT_HEIGHT_MAX  = 4096;
    parameter ARRAY_WIDTH       = PE_WIDTH;
    parameter ARRAY_HEIGHT      = PE_WIDTH;

    //SPM Config
    parameter SPM_SIZE             = 1<<19;
    parameter SPM_DATA_WIDTH       = PE_DATA_WIDTH_IN*PE_WIDTH  ;
    parameter SPM_ADDR_WIDTH       = $clog2(SPM_SIZE);
    parameter SPM_BANK_NUM         = 4;
    parameter SPM_BANK_DATA_WIDTH  = 128;
    parameter SPM_LINE_DATA_WIDTH  = SPM_BANK_NUM * SPM_BANK_DATA_WIDTH;
    parameter WR_PORTS             = 1;
    parameter RD_PORTS             = 2;
    parameter SPM_FPGA_SRAM        = 0;

    //ACC Config
    parameter ACC_SIZE             = 1<<19;
    parameter ACC_DATA_WIDTH       = PE_DATA_WIDTH_OUT*PE_WIDTH;
    parameter ACC_ADDR_WIDTH       = $clog2(ACC_SIZE);
    // parameter WR_PORTS             = 1;
    //  parameter RD_PORTS             = 2;
    parameter signed [31:0]  ALPHA_1  = 16'd6656    ;  //approximation for 0.1
    parameter signed [31:0]  ALPHA_2  = 16'd13312   ;  //approximation for 0.2
    parameter signed [31:0]  ALPHA_3  = 16'd656     ;  //approximation for 0.01

    // SFU Operator
    parameter SOFTMAX_OP          = 'd0;  
    parameter GELU_OP             = 'd1;  
    parameter LAYERNORM_OP        = 'd2;  
    parameter DOWNSAMPLE_MAX_OP   = 'd3;  //DOWNSAMPLE,support 2*2,stride = 2
    parameter DOWNSAMPLE_AVG_OP   = 'd4;  //not used in resample,support in resample_base
    parameter UPSAMPLE_NEAREST_OP = 'd5;  //not used in resample,support in resample_base
    parameter POOL_MAX_OP         = 'd6;  //POOLING ,support 2*2,stride = 1,not used in resample,support in resample_base
    parameter POOL_AVG_OP         = 'd7;  //not used in resample,support in resample_base
    parameter TRANSPOSE_OP        = 'd8;

    // SFU Disable Config
    parameter DISABLE_SOFTMAX     = 0;
    parameter DISABLE_GELU        = 0;
    parameter DISABLE_LAYERNORM   = 0;
    parameter DISABLE_RESAMPLE    = 0;
    parameter DISABLE_TRANSPOSE   = 0;

    // Resample Type
    parameter DOWNSAMPLE_TYPE     = 2'b00;
    parameter UPSAMPLE_TYPE       = 2'b01;
    parameter POOLING_TYPE        = 2'b10;

    //im2col Config
    parameter DATA_WIDTH_IN       = 16;
    parameter PATCH_GROUP         = PE_WIDTH;

endpackage
