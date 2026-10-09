/* emulation_memcomp Version: c0.1.2-EAC */
/* common_memcomp Version: c0.1.3-EAC */
/* lang compiler Version: 4.9.1-EAC Apr 14 2016 16:02:39 */
//
//       CONFIDENTIAL AND PROPRIETARY SOFTWARE OF ARM PHYSICAL IP, INC.
//      
//       Copyright (c) 1993 - 2025 ARM Physical IP, Inc.  All Rights Reserved.
//      
//       Use of this Software is subject to the terms and conditions of the
//       applicable license agreement with ARM Physical IP, Inc.
//       In addition, this Software is protected by patents, copyright law 
//       and international treaties.
//      
//       The copyright notice(s) in this Software does not indicate actual or
//       intended publication of this Software.
//
//      Emulation model for High Density Dual Port SRAM SVT MVT Compiler
//
//       Instance Name:              sram_8_1024_dp
//       Words:                      1024
//       Bits:                       8
//       Mux:                        16
//       Drive:                      6
//       Write Mask:                 On
//       Write Thru:                 Off
//       Extra Margin Adjustment:    On
//       Test Muxes                  On
//       Power Gating:               Off
//       Retention:                  On
//       Pipeline:                   Off
//       Read Disturb Test:	        Off
//       
//       Creation Date:  Wed Dec 10 20:48:14 2025
//       Version: 	r1p0
//
//      Modeling Assumptions: This model supports full gate level simulation
//          All buses are modeled [MSB:LSB].  All 
//          ports are padded with Verilog primitives.
//
//      Modeling Limitations: None.
//
//      Known Bugs: None.
//
//      Known Work Arounds: N/A
//
`timescale 1 ns/1 ps

module datapath_latch_sram_8_1024_dp (CLK,Q_update,D_update,SE,SI,D,DFTRAMBYP,mem_path,XQ,Q);
	input CLK,Q_update,D_update,SE,SI,D,DFTRAMBYP,mem_path,XQ;
	output Q;

	reg    D_int;
	reg    Q;

   //  Model PHI2 portion
   always @(CLK or SE or SI or D) begin
      if (CLK === 1'b0) begin
         if (SE===1'b1)
           D_int=SI;
         else
           D_int=D;
      end
   end

   // model output side of RAM latch
   always @(Q_update or D_update or mem_path) begin
      if (Q_update===1'b1) begin
         if (DFTRAMBYP===1'b1)
           Q=D_int;
         else
           Q=mem_path;
      end
   end
endmodule // datapath_latch_sram_8_1024_dp

// If POWER_PINS is defined at Simulator Command Line, it selects the module definition with Power Ports
`ifdef POWER_PINS
module sram_8_1024_dp (VDDCE, VDDPE, VSSE, CENYA, WENYA, AYA, CENYB, WENYB, AYB, GWENYA,
    GWENYB, QA, QB, SOA, SOB, CLKA, CENA, WENA, AA, DA, CLKB, CENB, WENB, AB, DB, EMAA,
    EMAWA, EMASA, EMAB, EMAWB, EMASB, TENA, TCENA, TWENA, TAA, TDA, TENB, TCENB, TWENB,
    TAB, TDB, GWENA, GWENB, TGWENA, TGWENB, RET1N, SIA, SEA, DFTRAMBYP, SIB, SEB, COLLDISN);
`else
module sram_8_1024_dp (CENYA, WENYA, AYA, CENYB, WENYB, AYB, GWENYA, GWENYB, QA, QB,
    SOA, SOB, CLKA, CENA, WENA, AA, DA, CLKB, CENB, WENB, AB, DB, EMAA, EMAWA, EMASA,
    EMAB, EMAWB, EMASB, TENA, TCENA, TWENA, TAA, TDA, TENB, TCENB, TWENB, TAB, TDB,
    GWENA, GWENB, TGWENA, TGWENB, RET1N, SIA, SEA, DFTRAMBYP, SIB, SEB, COLLDISN);
`endif

  parameter ASSERT_PREFIX = "";
  parameter BITS = 8;
  parameter WORDS = 1024;
  parameter MUX = 16;
  parameter MEM_WIDTH = 128; // redun block size 4, 64 on left, 64 on right
  parameter MEM_HEIGHT = 64;
  parameter WP_SIZE = 1 ;
  parameter UPM_WIDTH = 3;
  parameter UPMW_WIDTH = 2;
  parameter UPMS_WIDTH = 1;

  output  CENYA;
  output [7:0] WENYA;
  output [9:0] AYA;
  output  CENYB;
  output [7:0] WENYB;
  output [9:0] AYB;
  output  GWENYA;
  output  GWENYB;
  output [7:0] QA;
  output [7:0] QB;
  output [1:0] SOA;
  output [1:0] SOB;
  input  CLKA;
  input  CENA;
  input [7:0] WENA;
  input [9:0] AA;
  input [7:0] DA;
  input  CLKB;
  input  CENB;
  input [7:0] WENB;
  input [9:0] AB;
  input [7:0] DB;
  input [2:0] EMAA;
  input [1:0] EMAWA;
  input  EMASA;
  input [2:0] EMAB;
  input [1:0] EMAWB;
  input  EMASB;
  input  TENA;
  input  TCENA;
  input [7:0] TWENA;
  input [9:0] TAA;
  input [7:0] TDA;
  input  TENB;
  input  TCENB;
  input [7:0] TWENB;
  input [9:0] TAB;
  input [7:0] TDB;
  input  GWENA;
  input  GWENB;
  input  TGWENA;
  input  TGWENB;
  input  RET1N;
  input [1:0] SIA;
  input  SEA;
  input  DFTRAMBYP;
  input [1:0] SIB;
  input  SEB;
  input  COLLDISN;
`ifdef POWER_PINS
  inout VDDCE;
  inout VDDPE;
  inout VSSE;
`endif

  integer row_addressa;
  integer mux_addressa;
  reg [127:0] rowa, row_ta;
  integer row_address;
  integer mux_address;
  reg [127:0] mem [0:63];
  reg [127:0] row, row_t;
  reg LAST_CLKA;
  reg [127:0] row_maska;
  reg [127:0] new_dataa;
  reg [127:0] data_outa;
  reg [127:0] row_mask;
  reg [127:0] new_data;
  reg [127:0] data_out;
  reg [31:0] readLatch0;
  reg [31:0] shifted_readLatch0;
  reg [1:0] read_mux_sel0;
  reg [1:0] read_mux_sel0_p2;
  reg [31:0] readLatch1;
  reg [31:0] shifted_readLatch1;
  reg [1:0] read_mux_sel1;
  reg [1:0] read_mux_sel1_p2;
  reg LAST_CLKB;
  wire [7:0] QA_int;
  reg XQA, QA_update;
  reg XDA_sh, DA_sh_update;
  wire [7:0] DA_int_bmux;
  reg [7:0] mem_path_A;
  wire [7:0] QB_int;
  reg XQB, QB_update;
  reg XDB_sh, DB_sh_update;
  wire [7:0] DB_int_bmux;
  reg [7:0] mem_path_B;
  reg [7:0] writeEnablea;
  reg [7:0] writeEnable;
  reg READ_WRITE, WRITE_WRITE, READ_READ, ROW_CC, COL_CC;
  reg READ_WRITE_1, WRITE_WRITE_1, READ_READ_1;

  wire  CENYA_;
  wire [7:0] WENYA_;
  wire [9:0] AYA_;
  wire  CENYB_;
  wire [7:0] WENYB_;
  wire [9:0] AYB_;
  wire  GWENYA_;
  wire  GWENYB_;
  wire [7:0] QA_;
  wire [7:0] QB_;
  wire [1:0] SOA_;
  wire [1:0] SOB_;
 wire  CLKA_;
  wire  CENA_;
  reg  CENA_int;
  reg  CENA_p2;
  wire [7:0] WENA_;
  reg [7:0] WENA_int;
  wire [9:0] AA_;
  reg [9:0] AA_int;
  wire [7:0] DA_;
  reg [7:0] DA_int;
 wire  CLKB_;
  wire  CENB_;
  reg  CENB_int;
  reg  CENB_p2;
  wire [7:0] WENB_;
  reg [7:0] WENB_int;
  wire [9:0] AB_;
  reg [9:0] AB_int;
  wire [7:0] DB_;
  reg [7:0] DB_int;
  wire [2:0] EMAA_;
  reg [2:0] EMAA_int;
  wire [1:0] EMAWA_;
  reg [1:0] EMAWA_int;
  wire  EMASA_;
  reg  EMASA_int;
  wire [2:0] EMAB_;
  reg [2:0] EMAB_int;
  wire [1:0] EMAWB_;
  reg [1:0] EMAWB_int;
  wire  EMASB_;
  reg  EMASB_int;
  wire  TENA_;
  reg  TENA_int;
  wire  TCENA_;
  reg  TCENA_int;
  reg  TCENA_p2;
  wire [7:0] TWENA_;
  reg [7:0] TWENA_int;
  wire [9:0] TAA_;
  reg [9:0] TAA_int;
  wire [7:0] TDA_;
  reg [7:0] TDA_int;
  wire  TENB_;
  reg  TENB_int;
  wire  TCENB_;
  reg  TCENB_int;
  reg  TCENB_p2;
  wire [7:0] TWENB_;
  reg [7:0] TWENB_int;
  wire [9:0] TAB_;
  reg [9:0] TAB_int;
  wire [7:0] TDB_;
  reg [7:0] TDB_int;
  wire  GWENA_;
  reg  GWENA_int;
  wire  GWENB_;
  reg  GWENB_int;
  wire  TGWENA_;
  reg  TGWENA_int;
  wire  TGWENB_;
  reg  TGWENB_int;
  wire  RET1N_;
  reg  RET1N_int;
  wire [1:0] SIA_;
  wire [1:0] SIA_int;
  wire  SEA_;
  reg  SEA_int;
  wire  DFTRAMBYP_;
  reg  DFTRAMBYP_int;
  reg  DFTRAMBYP_p2;
  reg  DFTRAMBYP_inta;
  wire [1:0] SIB_;
  wire [1:0] SIB_int;
  wire  SEB_;
  reg  SEB_int;
  wire  COLLDISN_;
  reg  COLLDISN_int;

  assign CENYA = CENYA_; 
  assign WENYA[0] = WENYA_[0]; 
  assign WENYA[1] = WENYA_[1]; 
  assign WENYA[2] = WENYA_[2]; 
  assign WENYA[3] = WENYA_[3]; 
  assign WENYA[4] = WENYA_[4]; 
  assign WENYA[5] = WENYA_[5]; 
  assign WENYA[6] = WENYA_[6]; 
  assign WENYA[7] = WENYA_[7]; 
  assign AYA[0] = AYA_[0]; 
  assign AYA[1] = AYA_[1]; 
  assign AYA[2] = AYA_[2]; 
  assign AYA[3] = AYA_[3]; 
  assign AYA[4] = AYA_[4]; 
  assign AYA[5] = AYA_[5]; 
  assign AYA[6] = AYA_[6]; 
  assign AYA[7] = AYA_[7]; 
  assign AYA[8] = AYA_[8]; 
  assign AYA[9] = AYA_[9]; 
  assign CENYB = CENYB_; 
  assign WENYB[0] = WENYB_[0]; 
  assign WENYB[1] = WENYB_[1]; 
  assign WENYB[2] = WENYB_[2]; 
  assign WENYB[3] = WENYB_[3]; 
  assign WENYB[4] = WENYB_[4]; 
  assign WENYB[5] = WENYB_[5]; 
  assign WENYB[6] = WENYB_[6]; 
  assign WENYB[7] = WENYB_[7]; 
  assign AYB[0] = AYB_[0]; 
  assign AYB[1] = AYB_[1]; 
  assign AYB[2] = AYB_[2]; 
  assign AYB[3] = AYB_[3]; 
  assign AYB[4] = AYB_[4]; 
  assign AYB[5] = AYB_[5]; 
  assign AYB[6] = AYB_[6]; 
  assign AYB[7] = AYB_[7]; 
  assign AYB[8] = AYB_[8]; 
  assign AYB[9] = AYB_[9]; 
  assign GWENYA = GWENYA_; 
  assign GWENYB = GWENYB_; 
  assign QA[0] = QA_[0]; 
  assign QA[1] = QA_[1]; 
  assign QA[2] = QA_[2]; 
  assign QA[3] = QA_[3]; 
  assign QA[4] = QA_[4]; 
  assign QA[5] = QA_[5]; 
  assign QA[6] = QA_[6]; 
  assign QA[7] = QA_[7]; 
  assign QB[0] = QB_[0]; 
  assign QB[1] = QB_[1]; 
  assign QB[2] = QB_[2]; 
  assign QB[3] = QB_[3]; 
  assign QB[4] = QB_[4]; 
  assign QB[5] = QB_[5]; 
  assign QB[6] = QB_[6]; 
  assign QB[7] = QB_[7]; 
  assign SOA[0] = SOA_[0]; 
  assign SOA[1] = SOA_[1]; 
  assign SOB[0] = SOB_[0]; 
  assign SOB[1] = SOB_[1]; 
  assign CLKA_ = CLKA;
  assign CENA_ = CENA;
  assign WENA_[0] = WENA[0];
  assign WENA_[1] = WENA[1];
  assign WENA_[2] = WENA[2];
  assign WENA_[3] = WENA[3];
  assign WENA_[4] = WENA[4];
  assign WENA_[5] = WENA[5];
  assign WENA_[6] = WENA[6];
  assign WENA_[7] = WENA[7];
  assign AA_[0] = AA[0];
  assign AA_[1] = AA[1];
  assign AA_[2] = AA[2];
  assign AA_[3] = AA[3];
  assign AA_[4] = AA[4];
  assign AA_[5] = AA[5];
  assign AA_[6] = AA[6];
  assign AA_[7] = AA[7];
  assign AA_[8] = AA[8];
  assign AA_[9] = AA[9];
  assign DA_[0] = DA[0];
  assign DA_[1] = DA[1];
  assign DA_[2] = DA[2];
  assign DA_[3] = DA[3];
  assign DA_[4] = DA[4];
  assign DA_[5] = DA[5];
  assign DA_[6] = DA[6];
  assign DA_[7] = DA[7];
  assign CLKB_ = CLKB;
  assign CENB_ = CENB;
  assign WENB_[0] = WENB[0];
  assign WENB_[1] = WENB[1];
  assign WENB_[2] = WENB[2];
  assign WENB_[3] = WENB[3];
  assign WENB_[4] = WENB[4];
  assign WENB_[5] = WENB[5];
  assign WENB_[6] = WENB[6];
  assign WENB_[7] = WENB[7];
  assign AB_[0] = AB[0];
  assign AB_[1] = AB[1];
  assign AB_[2] = AB[2];
  assign AB_[3] = AB[3];
  assign AB_[4] = AB[4];
  assign AB_[5] = AB[5];
  assign AB_[6] = AB[6];
  assign AB_[7] = AB[7];
  assign AB_[8] = AB[8];
  assign AB_[9] = AB[9];
  assign DB_[0] = DB[0];
  assign DB_[1] = DB[1];
  assign DB_[2] = DB[2];
  assign DB_[3] = DB[3];
  assign DB_[4] = DB[4];
  assign DB_[5] = DB[5];
  assign DB_[6] = DB[6];
  assign DB_[7] = DB[7];
  assign EMAA_[0] = EMAA[0];
  assign EMAA_[1] = EMAA[1];
  assign EMAA_[2] = EMAA[2];
  assign EMAWA_[0] = EMAWA[0];
  assign EMAWA_[1] = EMAWA[1];
  assign EMASA_ = EMASA;
  assign EMAB_[0] = EMAB[0];
  assign EMAB_[1] = EMAB[1];
  assign EMAB_[2] = EMAB[2];
  assign EMAWB_[0] = EMAWB[0];
  assign EMAWB_[1] = EMAWB[1];
  assign EMASB_ = EMASB;
  assign TENA_ = TENA;
  assign TCENA_ = TCENA;
  assign TWENA_[0] = TWENA[0];
  assign TWENA_[1] = TWENA[1];
  assign TWENA_[2] = TWENA[2];
  assign TWENA_[3] = TWENA[3];
  assign TWENA_[4] = TWENA[4];
  assign TWENA_[5] = TWENA[5];
  assign TWENA_[6] = TWENA[6];
  assign TWENA_[7] = TWENA[7];
  assign TAA_[0] = TAA[0];
  assign TAA_[1] = TAA[1];
  assign TAA_[2] = TAA[2];
  assign TAA_[3] = TAA[3];
  assign TAA_[4] = TAA[4];
  assign TAA_[5] = TAA[5];
  assign TAA_[6] = TAA[6];
  assign TAA_[7] = TAA[7];
  assign TAA_[8] = TAA[8];
  assign TAA_[9] = TAA[9];
  assign TDA_[0] = TDA[0];
  assign TDA_[1] = TDA[1];
  assign TDA_[2] = TDA[2];
  assign TDA_[3] = TDA[3];
  assign TDA_[4] = TDA[4];
  assign TDA_[5] = TDA[5];
  assign TDA_[6] = TDA[6];
  assign TDA_[7] = TDA[7];
  assign TENB_ = TENB;
  assign TCENB_ = TCENB;
  assign TWENB_[0] = TWENB[0];
  assign TWENB_[1] = TWENB[1];
  assign TWENB_[2] = TWENB[2];
  assign TWENB_[3] = TWENB[3];
  assign TWENB_[4] = TWENB[4];
  assign TWENB_[5] = TWENB[5];
  assign TWENB_[6] = TWENB[6];
  assign TWENB_[7] = TWENB[7];
  assign TAB_[0] = TAB[0];
  assign TAB_[1] = TAB[1];
  assign TAB_[2] = TAB[2];
  assign TAB_[3] = TAB[3];
  assign TAB_[4] = TAB[4];
  assign TAB_[5] = TAB[5];
  assign TAB_[6] = TAB[6];
  assign TAB_[7] = TAB[7];
  assign TAB_[8] = TAB[8];
  assign TAB_[9] = TAB[9];
  assign TDB_[0] = TDB[0];
  assign TDB_[1] = TDB[1];
  assign TDB_[2] = TDB[2];
  assign TDB_[3] = TDB[3];
  assign TDB_[4] = TDB[4];
  assign TDB_[5] = TDB[5];
  assign TDB_[6] = TDB[6];
  assign TDB_[7] = TDB[7];
  assign GWENA_ = GWENA;
  assign GWENB_ = GWENB;
  assign TGWENA_ = TGWENA;
  assign TGWENB_ = TGWENB;
  assign RET1N_ = RET1N;
  assign SIA_[0] = SIA[0];
  assign SIA_[1] = SIA[1];
  assign SEA_ = SEA;
  assign DFTRAMBYP_ = DFTRAMBYP;
  assign SIB_[0] = SIB[0];
  assign SIB_[1] = SIB[1];
  assign SEB_ = SEB;
  assign COLLDISN_ = COLLDISN;

  assign CENYA_ = RET1N_ ? (DFTRAMBYP_ & (TENA_ ? CENA_ : TCENA_)) : 1'bx;
  assign WENYA_ = RET1N_ ? ({8{DFTRAMBYP_}} & (TENA_ ? WENA_ : TWENA_)) : {8{1'bx}};
  assign AYA_ = RET1N_ ? ({10{DFTRAMBYP_}} & (TENA_ ? AA_ : TAA_)) : {10{1'bx}};
  assign CENYB_ = RET1N_ ? (DFTRAMBYP_ & (TENB_ ? CENB_ : TCENB_)) : 1'bx;
  assign WENYB_ = RET1N_ ? ({8{DFTRAMBYP_}} & (TENB_ ? WENB_ : TWENB_)) : {8{1'bx}};
  assign AYB_ = RET1N_ ? ({10{DFTRAMBYP_}} & (TENB_ ? AB_ : TAB_)) : {10{1'bx}};
  assign GWENYA_ = RET1N_ ? (DFTRAMBYP_ & (TENA_ ? GWENA_ : TGWENA_)) : 1'bx;
  assign GWENYB_ = RET1N_ ? (DFTRAMBYP_ & (TENB_ ? GWENB_ : TGWENB_)) : 1'bx;
  assign QA_ = RET1N_ ? ((QA_int)) : {8{1'bx}};
  assign QB_ = RET1N_ ? ((QB_int)) : {8{1'bx}};
  assign SOA_ = RET1N_ ? ({QA_[4], QA_[3]}) : {2{1'bx}};
  assign SOB_ = RET1N_ ? ({QB_[4], QB_[3]}) : {2{1'bx}};

// If INITIALIZE_MEMORY is defined at Simulator Command Line, it Initializes the Memory with all ZEROS.
`ifdef INITIALIZE_MEMORY
  integer i;
  initial
    for (i = 0; i < MEM_HEIGHT; i = i + 1)
      mem[i] = {MEM_WIDTH{1'b0}};
`endif

  function isBit1;
    input bitval;
    begin
      isBit1 = ( bitval===1'b1 ) ? 1'b1 : 1'b0;
    end
  endfunction


`ifdef ARM_EMU_FAULT
   localparam AWIDTH = 10; // ceiling of log2
   localparam MWIDTH = MEM_WIDTH;
   reg [AWIDTH-1:0] emu_AA_int;
   reg [MWIDTH-1:0] emu_rowa;

   always @(*) begin
     emu_AA_int = TENA_ ? AA_ : TAA_;
     emu_rowa = mem[emu_AA_int >> 4];
   end
`endif

  task readWriteA;
  begin
    if (RET1N_int === 1'b0 && (CENA_int === 1'b0 || DFTRAMBYP_inta === 1'b1)) begin
    end else if (RET1N_int === 1'b0) begin
      // no cycle in retention mode
    end else if ((AA_int >= WORDS) && (CENA_int === 1'b0) && DFTRAMBYP_inta === 1'b0) begin
    end else if (CENA_int === 1'b0 || DFTRAMBYP_inta === 1'b1) begin
      mux_addressa = (AA_int & 4'b1111);
      row_addressa = (AA_int >> 4);
`ifdef ARM_EMU_FAULT
	rowa = emu_rowa;
`else
      if (DFTRAMBYP_inta !== 1'b1) begin
      if (row_addressa > 63)
        rowa = {128{1'bx}};
      else
        rowa = mem[row_addressa];
      end
`endif
        writeEnablea = ~ ( {8{GWENA_int}} | {WENA_int[7], WENA_int[6], WENA_int[5],
          WENA_int[4], WENA_int[3], WENA_int[2], WENA_int[1], WENA_int[0]});
      if (GWENA_int !== 1'b1 || DFTRAMBYP_inta === 1'b1) begin
        row_maska =  ( {15'b000000000000000, writeEnablea[7], 15'b000000000000000, writeEnablea[6],
          15'b000000000000000, writeEnablea[5], 15'b000000000000000, writeEnablea[4],
          15'b000000000000000, writeEnablea[3], 15'b000000000000000, writeEnablea[2],
          15'b000000000000000, writeEnablea[1], 15'b000000000000000, writeEnablea[0]} << mux_addressa);
        new_dataa =  ( {15'b000000000000000, DA_int[7], 15'b000000000000000, DA_int[6],
          15'b000000000000000, DA_int[5], 15'b000000000000000, DA_int[4], 15'b000000000000000, DA_int[3],
          15'b000000000000000, DA_int[2], 15'b000000000000000, DA_int[1], 15'b000000000000000, DA_int[0]} << mux_addressa);
        rowa = (rowa & ~row_maska) | (row_maska & (~row_maska | new_dataa));
        if (DFTRAMBYP_inta === 1'b1 && SEA_int === 1'b0) begin
        end else begin
        mem[row_addressa] = rowa;
        end
      end else begin
        data_outa = (rowa >> (mux_addressa%4));
        readLatch0 = {data_outa[124], data_outa[120], data_outa[116], data_outa[112],
          data_outa[108], data_outa[104], data_outa[100], data_outa[96], data_outa[92],
          data_outa[88], data_outa[84], data_outa[80], data_outa[76], data_outa[72],
          data_outa[68], data_outa[64], data_outa[60], data_outa[56], data_outa[52],
          data_outa[48], data_outa[44], data_outa[40], data_outa[36], data_outa[32],
          data_outa[28], data_outa[24], data_outa[20], data_outa[16], data_outa[12],
          data_outa[8], data_outa[4], data_outa[0]};
        shifted_readLatch0 = (readLatch0 >> AA_int[3:2]);
        mem_path_A = {shifted_readLatch0[28], shifted_readLatch0[24], shifted_readLatch0[20],
          shifted_readLatch0[16], shifted_readLatch0[12], shifted_readLatch0[8], shifted_readLatch0[4],
          shifted_readLatch0[0]};
        	XQA = 1'b0; QA_update = 1'b1;
      end
      if (DFTRAMBYP_inta === 1'b1) begin
        	XQA = 1'b0; QA_update = 1'b1;
      end
    end
  end
  endtask



  always @ (posedge CLKA_) begin
`ifdef POWER_PINS
  if (RET1N_ == 1'b0) begin
`else     
  if (RET1N_ == 1'b0) begin
`endif
      // no cycle in retention mode
  end else begin
      SEA_int = SEA_;
      DFTRAMBYP_inta = DFTRAMBYP_;
      CENA_int = TENA_ ? CENA_ : TCENA_;
      EMAA_int = EMAA_;
      EMAWA_int = EMAWA_;
      EMASA_int = EMASA_;
      TENA_int = TENA_;
      TWENA_int = TWENA_;
      RET1N_int = RET1N_;
      COLLDISN_int = COLLDISN_;
      if (DFTRAMBYP_=== 1'b1 || CENA_int != 1'b1) begin
        WENA_int = TENA_ ? WENA_ : TWENA_;
        AA_int = TENA_ ? AA_ : TAA_;
        DA_int = TENA_ ? DA_ : TDA_;
        TCENA_int = TCENA_;
        TAA_int = TAA_;
        TDA_int = TDA_;
        GWENA_int = TENA_ ? GWENA_ : TGWENA_;
        TGWENA_int = TGWENA_;
        DFTRAMBYP_inta = DFTRAMBYP_;
        if (GWENA_int === 1'b1 || DFTRAMBYP_ == 1'b1) begin
          read_mux_sel0 = (TENA_ ? AA_[3:2] : TAA_[3:2] );
          read_mux_sel0_p2 = ((^read_mux_sel0 === 1'bx) && DFTRAMBYP_p2) ? {2{1'b0}} : read_mux_sel0;
        end
      end
      if (DFTRAMBYP_=== 1'b1 && SEA_ === 1'b1) begin
         read_mux_sel0 = (TENA_ ? AA_[3:2] : TAA_[3:2] );
         read_mux_sel0_p2 = ((^read_mux_sel0 === 1'bx) && DFTRAMBYP_p2) ? {2{1'b0}} : read_mux_sel0;
        XQA = 1'b0; QA_update = 1'b1;
      end else begin
      CENA_int = TENA_ ? CENA_ : TCENA_;
      EMAA_int = EMAA_;
      EMAWA_int = EMAWA_;
      EMASA_int = EMASA_;
      TENA_int = TENA_;
      TWENA_int = TWENA_;
      RET1N_int = RET1N_;
      COLLDISN_int = COLLDISN_;
      if (DFTRAMBYP_=== 1'b1 || CENA_int != 1'b1) begin
        WENA_int = TENA_ ? WENA_ : TWENA_;
        AA_int = TENA_ ? AA_ : TAA_;
        DA_int = TENA_ ? DA_ : TDA_;
        TCENA_int = TCENA_;
        TAA_int = TAA_;
        TDA_int = TDA_;
        GWENA_int = TENA_ ? GWENA_ : TGWENA_;
        TGWENA_int = TGWENA_;
        DFTRAMBYP_inta = DFTRAMBYP_;
        if (GWENA_int === 1'b1 || DFTRAMBYP_ == 1'b1) begin
          read_mux_sel0 = (TENA_ ? AA_[3:2] : TAA_[3:2] );
          read_mux_sel0_p2 = ((^read_mux_sel0 === 1'bx) && DFTRAMBYP_p2) ? {2{1'b0}} : read_mux_sel0;
        end
      end
    readWriteA;
      end
    #0;
`ifdef NO_COLLISIONS
`else     
      if ((1) && (CENA_int !== 1'b1 && CENB_int !== 1'b1 && DFTRAMBYP_ !== 1'b1) && COLLDISN_int === 1'b1 && row_contention(AA_int, AB_int, ({8{GWENA_int}}|WENA_int),
        ({8{GWENB_int}}|WENB_int))) begin
        if (col_contention(AA_int, AB_int)) begin
          COL_CC = 1;
        end
          ROW_CC = 1;
          READ_READ_1 = 0;
          READ_WRITE_1 = 0;
          WRITE_WRITE_1 = 0;
        if (GWENA_int !== 1'b1 && (& WENA_int) !== 1'b1 && GWENB_int !== 1'b1 && (& WENB_int) !== 1'b1) begin
	      if (is_contention(AA_int, AB_int, ({8{GWENA_int}}|WENA_int), ({8{GWENB_int}}|WENB_int))) begin
          COL_CC = 1;
          WRITE_WRITE = 1;
	      end
        end else if (GWENA_int !== 1'b1 && (& WENA_int) !== 1'b1) begin
		if (is_contention(AA_int, AB_int, ({8{GWENA_int}}|WENA_int), ({8{GWENB_int}}|WENB_int))) begin
          COL_CC = 1;
          READ_WRITE = 1;
		end
        end else if (GWENB_int !== 1'b1 && (& WENB_int) !== 1'b1) begin
		if (is_contention(AA_int, AB_int, ({8{GWENA_int}}|WENA_int), ({8{GWENB_int}}|WENB_int))) begin
          COL_CC = 1;
          READ_WRITE = 1;
		end
        end else begin
          COL_CC = 1;
          READ_READ = 1;
        end
        if (!is_contention(AA_int, AB_int, ({8{GWENA_int}}|WENA_int), ({8{GWENB_int}}|WENB_int))) begin
        if (GWENA_int !== 1'b1 && (& WENA_int) !== 1'b1 && GWENB_int !== 1'b1 && (& WENB_int) !== 1'b1) begin
          WRITE_WRITE = 1;
        end else if (!(GWENA_int !== 1'b1 && (& WENA_int) !== 1'b1) && (GWENB_int !== 1'b1 && (& WENB_int) !== 1'b1)) begin
          READ_WRITE = 1;
        end else if ((GWENA_int !== 1'b1 && (& WENA_int) !== 1'b1) && !(GWENB_int !== 1'b1 && (& WENB_int) !== 1'b1)) begin
          READ_WRITE = 1;
        end else begin
        end
        end
      end else if ((1) && (CENA_int !== 1'b1 && CENB_int !== 1'b1 && DFTRAMBYP_ !== 1'b1) && (COLLDISN_int === 1'b0 || COLLDISN_int === 1'bx)  && row_contention(AA_int,
        AB_int, ({8{GWENA_int}}|WENA_int), ({8{GWENB_int}}|WENB_int))) begin
          ROW_CC = 1;
          READ_READ_1 = 0;
          READ_WRITE_1 = 0;
          WRITE_WRITE_1 = 0;
        if (col_contention(AA_int, AB_int)) begin
          COL_CC = 1;
        end
        if (GWENB_int !== 1'b1 && (& WENB_int) !== 1'b1) begin
          WRITE_WRITE_1 = 1;
        end else if (is_contention(AA_int, AB_int, ({8{GWENA_int}}|WENA_int), ({8{GWENB_int}}|WENB_int))) begin
          COL_CC = 1;
          READ_WRITE_1 = 1;
        end else begin
          READ_WRITE_1 = 1;
          READ_READ_1 = 1;
        end
        if (GWENA_int !== 1'b1 && (& WENA_int) !== 1'b1) begin
          if(WRITE_WRITE_1)
            WRITE_WRITE = 1;
          if(READ_WRITE_1) begin
            READ_WRITE = 1;
            READ_WRITE_1 = 0;
          end
        end else if (is_contention(AA_int, AB_int, ({8{GWENA_int}}|WENA_int), ({8{GWENB_int}}|WENB_int))) begin
          COL_CC = 1;
          if(READ_WRITE_1) begin
            READ_WRITE = 1;
            READ_WRITE_1 = 0;
          end
        end else begin
          if(READ_READ_1) begin
            READ_READ = 1;
            READ_READ_1 = 0;
          end
        end
      end
`endif
    end
  end
  always @ (negedge CLKA_) begin
      QA_update = 1'b0;
      DA_sh_update = 1'b0;
      XQA = 1'b0;
         read_mux_sel0_p2 = read_mux_sel0;
  end

  assign SIA_int = SEA_ ? SIA_ : {2{1'b0}};
  assign DA_int_bmux = TENA_ ? DA_ : TDA_;

  datapath_latch_sram_8_1024_dp uDQA0 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(SIA_int[0]), .D(DA_int_bmux[0]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[0]), .XQ(XQA), .Q(QA_int[0]));
  datapath_latch_sram_8_1024_dp uDQA1 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[0]), .D(DA_int_bmux[1]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[1]), .XQ(XQA), .Q(QA_int[1]));
  datapath_latch_sram_8_1024_dp uDQA2 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[1]), .D(DA_int_bmux[2]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[2]), .XQ(XQA), .Q(QA_int[2]));
  datapath_latch_sram_8_1024_dp uDQA3 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[2]), .D(DA_int_bmux[3]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[3]), .XQ(XQA), .Q(QA_int[3]));
  datapath_latch_sram_8_1024_dp uDQA4 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[5]), .D(DA_int_bmux[4]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[4]), .XQ(XQA), .Q(QA_int[4]));
  datapath_latch_sram_8_1024_dp uDQA5 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[6]), .D(DA_int_bmux[5]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[5]), .XQ(XQA), .Q(QA_int[5]));
  datapath_latch_sram_8_1024_dp uDQA6 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[7]), .D(DA_int_bmux[6]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[6]), .XQ(XQA), .Q(QA_int[6]));
  datapath_latch_sram_8_1024_dp uDQA7 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(SIA_int[1]), .D(DA_int_bmux[7]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[7]), .XQ(XQA), .Q(QA_int[7]));


`ifdef ARM_EMU_FAULT
   reg [AWIDTH-1:0] emu_AB_int;
   reg [MWIDTH-1:0] emu_rowb;

   always @(*) begin
     emu_AB_int = TENB_ ? AB_ : TAB_;
     emu_rowb = mem[emu_AB_int >> 4];
   end
`endif

  task readWriteB;
  begin
    if (RET1N_int === 1'b0 && (CENB_int === 1'b0 || DFTRAMBYP_int === 1'b1)) begin
    end else if (RET1N_int === 1'b0) begin
      // no cycle in retention mode
    end else if ((AB_int >= WORDS) && (CENB_int === 1'b0) && DFTRAMBYP_int === 1'b0) begin
    end else if (CENB_int === 1'b0 || DFTRAMBYP_int === 1'b1) begin
      mux_address = (AB_int & 4'b1111);
      row_address = (AB_int >> 4);
`ifdef ARM_EMU_FAULT
	row = emu_rowb;
`else
      if (DFTRAMBYP_int !== 1'b1) begin
      if (row_address > 63)
        row = {128{1'bx}};
      else
        row = mem[row_address];
      end
`endif
        writeEnable = ~ ( {8{GWENB_int}} | {WENB_int[7], WENB_int[6], WENB_int[5],
          WENB_int[4], WENB_int[3], WENB_int[2], WENB_int[1], WENB_int[0]});
      if (GWENB_int !== 1'b1 || DFTRAMBYP_int === 1'b1) begin
        row_mask =  ( {15'b000000000000000, writeEnable[7], 15'b000000000000000, writeEnable[6],
          15'b000000000000000, writeEnable[5], 15'b000000000000000, writeEnable[4],
          15'b000000000000000, writeEnable[3], 15'b000000000000000, writeEnable[2],
          15'b000000000000000, writeEnable[1], 15'b000000000000000, writeEnable[0]} << mux_address);
        new_data =  ( {15'b000000000000000, DB_int[7], 15'b000000000000000, DB_int[6],
          15'b000000000000000, DB_int[5], 15'b000000000000000, DB_int[4], 15'b000000000000000, DB_int[3],
          15'b000000000000000, DB_int[2], 15'b000000000000000, DB_int[1], 15'b000000000000000, DB_int[0]} << mux_address);
        row = (row & ~row_mask) | (row_mask & (~row_mask | new_data));
        if (DFTRAMBYP_int === 1'b1 && SEB_int === 1'b0) begin
        end else begin
        mem[row_address] = row;
        end
      end else begin
        data_out = (row >> (mux_address%4));
        readLatch1 = {data_out[124], data_out[120], data_out[116], data_out[112], data_out[108],
          data_out[104], data_out[100], data_out[96], data_out[92], data_out[88], data_out[84],
          data_out[80], data_out[76], data_out[72], data_out[68], data_out[64], data_out[60],
          data_out[56], data_out[52], data_out[48], data_out[44], data_out[40], data_out[36],
          data_out[32], data_out[28], data_out[24], data_out[20], data_out[16], data_out[12],
          data_out[8], data_out[4], data_out[0]};
        shifted_readLatch1 = (readLatch1 >> AB_int[3:2]);
        mem_path_B = {shifted_readLatch1[28], shifted_readLatch1[24], shifted_readLatch1[20],
          shifted_readLatch1[16], shifted_readLatch1[12], shifted_readLatch1[8], shifted_readLatch1[4],
          shifted_readLatch1[0]};
        	XQB = 1'b0; QB_update = 1'b1;
      end
      if (DFTRAMBYP_int === 1'b1) begin
        	XQB = 1'b0; QB_update = 1'b1;
      end
    end
  end
  endtask



  always @ (posedge CLKB_) begin
`ifdef POWER_PINS
  if (RET1N_ == 1'b0) begin
`else     
  if (RET1N_ == 1'b0) begin
`endif
      // no cycle in retention mode
  end else begin
      DFTRAMBYP_int = DFTRAMBYP_;
      SEB_int = SEB_;
      CENB_int = TENB_ ? CENB_ : TCENB_;
      EMAB_int = EMAB_;
      EMAWB_int = EMAWB_;
      EMASB_int = EMASB_;
      TENB_int = TENB_;
      TWENB_int = TWENB_;
      RET1N_int = RET1N_;
      COLLDISN_int = COLLDISN_;
      if (DFTRAMBYP_=== 1'b1 || CENB_int != 1'b1) begin
        WENB_int = TENB_ ? WENB_ : TWENB_;
        AB_int = TENB_ ? AB_ : TAB_;
        DB_int = TENB_ ? DB_ : TDB_;
        TCENB_int = TCENB_;
        TAB_int = TAB_;
        TDB_int = TDB_;
        GWENB_int = TENB_ ? GWENB_ : TGWENB_;
        TGWENB_int = TGWENB_;
        if (GWENB_int === 1'b1 || DFTRAMBYP_ == 1'b1) begin
          read_mux_sel1 = (TENB_ ? AB_[3:2] : TAB_[3:2] );
          read_mux_sel1_p2 = ((^read_mux_sel1 === 1'bx) && DFTRAMBYP_p2) ? {2{1'b0}} : read_mux_sel1;
        end
      end
      if (DFTRAMBYP_=== 1'b1 && SEB_ === 1'b1) begin
         read_mux_sel1 = (TENB_ ? AB_[3:2] : TAB_[3:2] );
         read_mux_sel1_p2 = ((^read_mux_sel1 === 1'bx) && DFTRAMBYP_p2) ? {2{1'b0}} : read_mux_sel1;
        XQB = 1'b0; QB_update = 1'b1;
      end else begin
      CENB_int = TENB_ ? CENB_ : TCENB_;
      EMAB_int = EMAB_;
      EMAWB_int = EMAWB_;
      EMASB_int = EMASB_;
      TENB_int = TENB_;
      TWENB_int = TWENB_;
      RET1N_int = RET1N_;
      COLLDISN_int = COLLDISN_;
      if (DFTRAMBYP_=== 1'b1 || CENB_int != 1'b1) begin
        WENB_int = TENB_ ? WENB_ : TWENB_;
        AB_int = TENB_ ? AB_ : TAB_;
        DB_int = TENB_ ? DB_ : TDB_;
        TCENB_int = TCENB_;
        TAB_int = TAB_;
        TDB_int = TDB_;
        GWENB_int = TENB_ ? GWENB_ : TGWENB_;
        TGWENB_int = TGWENB_;
        if (GWENB_int === 1'b1 || DFTRAMBYP_ == 1'b1) begin
          read_mux_sel1 = (TENB_ ? AB_[3:2] : TAB_[3:2] );
          read_mux_sel1_p2 = ((^read_mux_sel1 === 1'bx) && DFTRAMBYP_p2) ? {2{1'b0}} : read_mux_sel1;
        end
      end
    readWriteB;
      end
    #0;
`ifdef NO_COLLISIONS
`else     
      if ((1) && (CENA_int !== 1'b1 && CENB_int !== 1'b1 && DFTRAMBYP_ !== 1'b1) && COLLDISN_int === 1'b1 && row_contention(AA_int, AB_int, ({8{GWENA_int}}|WENA_int),
        ({8{GWENB_int}}|WENB_int))) begin
        if (col_contention(AA_int, AB_int)) begin
          COL_CC = 1;
        end
          ROW_CC = 1;
          READ_READ_1 = 0;
          READ_WRITE_1 = 0;
          WRITE_WRITE_1 = 0;
        if (GWENA_int !== 1'b1 && (& WENA_int) !== 1'b1 && GWENB_int !== 1'b1 && (& WENB_int) !== 1'b1) begin
	      if (is_contention(AA_int, AB_int, ({8{GWENA_int}}|WENA_int), ({8{GWENB_int}}|WENB_int))) begin
          COL_CC = 1;
          WRITE_WRITE = 1;
	      end
        end else if (GWENA_int !== 1'b1 && (& WENA_int) !== 1'b1) begin
		if (is_contention(AA_int, AB_int, ({8{GWENA_int}}|WENA_int), ({8{GWENB_int}}|WENB_int))) begin
          COL_CC = 1;
          READ_WRITE = 1;
		end
        end else if (GWENB_int !== 1'b1 && (& WENB_int) !== 1'b1) begin
		if (is_contention(AA_int, AB_int, ({8{GWENA_int}}|WENA_int), ({8{GWENB_int}}|WENB_int))) begin
          COL_CC = 1;
          READ_WRITE = 1;
		end
        end else begin
          COL_CC = 1;
          READ_READ = 1;
        end
        if (!is_contention(AA_int, AB_int, ({8{GWENA_int}}|WENA_int), ({8{GWENB_int}}|WENB_int))) begin
        if (GWENA_int !== 1'b1 && (& WENA_int) !== 1'b1 && GWENB_int !== 1'b1 && (& WENB_int) !== 1'b1) begin
          WRITE_WRITE = 1;
        end else if (!(GWENA_int !== 1'b1 && (& WENA_int) !== 1'b1) && (GWENB_int !== 1'b1 && (& WENB_int) !== 1'b1)) begin
          READ_WRITE = 1;
        end else if ((GWENA_int !== 1'b1 && (& WENA_int) !== 1'b1) && !(GWENB_int !== 1'b1 && (& WENB_int) !== 1'b1)) begin
          READ_WRITE = 1;
        end else begin
        end
        end
      end else if ((1) && (CENA_int !== 1'b1 && CENB_int !== 1'b1 && DFTRAMBYP_ !== 1'b1) && (COLLDISN_int === 1'b0 || COLLDISN_int === 1'bx)  && row_contention(AA_int,
        AB_int, ({8{GWENA_int}}|WENA_int), ({8{GWENB_int}}|WENB_int))) begin
          ROW_CC = 1;
          READ_READ_1 = 0;
          READ_WRITE_1 = 0;
          WRITE_WRITE_1 = 0;
        if (col_contention(AA_int, AB_int)) begin
          COL_CC = 1;
        end
        if (GWENA_int !== 1'b1 && (& WENA_int) !== 1'b1) begin
          WRITE_WRITE_1 = 1;
        end else if (is_contention(AA_int, AB_int, ({8{GWENA_int}}|WENA_int), ({8{GWENB_int}}|WENB_int))) begin
          COL_CC = 1;
          READ_WRITE_1 = 1;
        end else begin
          READ_READ_1 = 1;
          READ_WRITE_1 = 1;
        end
        if (GWENB_int !== 1'b1 && (& WENB_int) !== 1'b1) begin
          if(WRITE_WRITE_1)
            WRITE_WRITE = 1;
          if(READ_WRITE_1) begin
            READ_WRITE = 1;
            READ_WRITE_1 = 0;
          end
        end else if (is_contention(AA_int, AB_int, ({8{GWENA_int}}|WENA_int), ({8{GWENB_int}}|WENB_int))) begin
          COL_CC = 1;
          if(READ_WRITE_1) begin
            READ_WRITE = 1;
            READ_WRITE_1 = 0;
          end
        end else begin
          if(READ_READ_1) begin
            READ_READ = 1;
            READ_READ_1 = 0;
          end
        end
      end
`endif
    end
  end
  always @ (negedge CLKB_) begin
      QB_update = 1'b0;
      DB_sh_update = 1'b0;
      XQB = 1'b0;
         read_mux_sel1_p2 = read_mux_sel1;
  end

  assign SIB_int = SEB_ ? SIB_ : {2{1'b0}};
  assign DB_int_bmux = TENB_ ? DB_ : TDB_;

  datapath_latch_sram_8_1024_dp uDQB0 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(SIB_int[0]), .D(DB_int_bmux[0]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[0]), .XQ(XQB), .Q(QB_int[0]));
  datapath_latch_sram_8_1024_dp uDQB1 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[0]), .D(DB_int_bmux[1]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[1]), .XQ(XQB), .Q(QB_int[1]));
  datapath_latch_sram_8_1024_dp uDQB2 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[1]), .D(DB_int_bmux[2]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[2]), .XQ(XQB), .Q(QB_int[2]));
  datapath_latch_sram_8_1024_dp uDQB3 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[2]), .D(DB_int_bmux[3]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[3]), .XQ(XQB), .Q(QB_int[3]));
  datapath_latch_sram_8_1024_dp uDQB4 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[5]), .D(DB_int_bmux[4]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[4]), .XQ(XQB), .Q(QB_int[4]));
  datapath_latch_sram_8_1024_dp uDQB5 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[6]), .D(DB_int_bmux[5]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[5]), .XQ(XQB), .Q(QB_int[5]));
  datapath_latch_sram_8_1024_dp uDQB6 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[7]), .D(DB_int_bmux[6]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[6]), .XQ(XQB), .Q(QB_int[6]));
  datapath_latch_sram_8_1024_dp uDQB7 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(SIB_int[1]), .D(DB_int_bmux[7]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[7]), .XQ(XQB), .Q(QB_int[7]));



  function row_contention;
    input [9:0] aa;
    input [9:0] ab;
    input [7:0] wena;
    input [7:0] wenb;
    reg result;
    reg sameRow;
    reg sameMux;
    reg anyWrite;
  begin
    anyWrite = ((& wena) === 1'b1 && (& wenb) === 1'b1) ? 1'b0 : 1'b1;
    sameMux = (aa[3:0] == ab[3:0]) ? 1'b1 : 1'b0;
    if (aa[9:4] == ab[9:4]) begin
      sameRow = 1'b1;
    end else begin
      sameRow = 1'b0;
    end
    if (sameRow == 1'b1 && anyWrite == 1'b1)
      row_contention = 1'b1;
    else if (sameRow == 1'b1 && sameMux == 1'b1)
      row_contention = 1'b1;
    else
      row_contention = 1'b0;
  end
  endfunction

  function col_contention;
    input [9:0] aa;
    input [9:0] ab;
  begin
    if (aa[3:0] == ab[3:0])
      col_contention = 1'b1;
    else
      col_contention = 1'b0;
  end
  endfunction

  function is_contention;
    input [9:0] aa;
    input [9:0] ab;
    input [7:0] wena;
    input [7:0] wenb;
    reg result;
  begin
    if ((& wena) === 1'b1 && (& wenb) === 1'b1) begin
      result = 1'b0;
    end else if (aa == ab) begin
      result = 1'b1;
    end else begin
      result = 1'b0;
    end
    is_contention = result;
  end
  endfunction


endmodule
