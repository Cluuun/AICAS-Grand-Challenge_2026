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
//       Instance Name:              sram_32_1024_dp
//       Words:                      1024
//       Bits:                       32
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
//       Creation Date:  Wed Dec 10 20:55:32 2025
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

module datapath_latch_sram_32_1024_dp (CLK,Q_update,D_update,SE,SI,D,DFTRAMBYP,mem_path,XQ,Q);
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
endmodule // datapath_latch_sram_32_1024_dp

// If POWER_PINS is defined at Simulator Command Line, it selects the module definition with Power Ports
`ifdef POWER_PINS
module sram_32_1024_dp (VDDCE, VDDPE, VSSE, CENYA, WENYA, AYA, CENYB, WENYB, AYB, GWENYA,
    GWENYB, QA, QB, SOA, SOB, CLKA, CENA, WENA, AA, DA, CLKB, CENB, WENB, AB, DB, EMAA,
    EMAWA, EMASA, EMAB, EMAWB, EMASB, TENA, TCENA, TWENA, TAA, TDA, TENB, TCENB, TWENB,
    TAB, TDB, GWENA, GWENB, TGWENA, TGWENB, RET1N, SIA, SEA, DFTRAMBYP, SIB, SEB, COLLDISN);
`else
module sram_32_1024_dp (CENYA, WENYA, AYA, CENYB, WENYB, AYB, GWENYA, GWENYB, QA, QB,
    SOA, SOB, CLKA, CENA, WENA, AA, DA, CLKB, CENB, WENB, AB, DB, EMAA, EMAWA, EMASA,
    EMAB, EMAWB, EMASB, TENA, TCENA, TWENA, TAA, TDA, TENB, TCENB, TWENB, TAB, TDB,
    GWENA, GWENB, TGWENA, TGWENB, RET1N, SIA, SEA, DFTRAMBYP, SIB, SEB, COLLDISN);
`endif

  parameter ASSERT_PREFIX = "";
  parameter BITS = 32;
  parameter WORDS = 1024;
  parameter MUX = 16;
  parameter MEM_WIDTH = 512; // redun block size 4, 256 on left, 256 on right
  parameter MEM_HEIGHT = 64;
  parameter WP_SIZE = 1 ;
  parameter UPM_WIDTH = 3;
  parameter UPMW_WIDTH = 2;
  parameter UPMS_WIDTH = 1;

  output  CENYA;
  output [31:0] WENYA;
  output [9:0] AYA;
  output  CENYB;
  output [31:0] WENYB;
  output [9:0] AYB;
  output  GWENYA;
  output  GWENYB;
  output [31:0] QA;
  output [31:0] QB;
  output [1:0] SOA;
  output [1:0] SOB;
  input  CLKA;
  input  CENA;
  input [31:0] WENA;
  input [9:0] AA;
  input [31:0] DA;
  input  CLKB;
  input  CENB;
  input [31:0] WENB;
  input [9:0] AB;
  input [31:0] DB;
  input [2:0] EMAA;
  input [1:0] EMAWA;
  input  EMASA;
  input [2:0] EMAB;
  input [1:0] EMAWB;
  input  EMASB;
  input  TENA;
  input  TCENA;
  input [31:0] TWENA;
  input [9:0] TAA;
  input [31:0] TDA;
  input  TENB;
  input  TCENB;
  input [31:0] TWENB;
  input [9:0] TAB;
  input [31:0] TDB;
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
  reg [511:0] rowa, row_ta;
  integer row_address;
  integer mux_address;
  reg [511:0] mem [0:63];
  reg [511:0] row, row_t;
  reg LAST_CLKA;
  reg [511:0] row_maska;
  reg [511:0] new_dataa;
  reg [511:0] data_outa;
  reg [511:0] row_mask;
  reg [511:0] new_data;
  reg [511:0] data_out;
  reg [127:0] readLatch0;
  reg [127:0] shifted_readLatch0;
  reg [1:0] read_mux_sel0;
  reg [1:0] read_mux_sel0_p2;
  reg [127:0] readLatch1;
  reg [127:0] shifted_readLatch1;
  reg [1:0] read_mux_sel1;
  reg [1:0] read_mux_sel1_p2;
  reg LAST_CLKB;
  wire [31:0] QA_int;
  reg XQA, QA_update;
  reg XDA_sh, DA_sh_update;
  wire [31:0] DA_int_bmux;
  reg [31:0] mem_path_A;
  wire [31:0] QB_int;
  reg XQB, QB_update;
  reg XDB_sh, DB_sh_update;
  wire [31:0] DB_int_bmux;
  reg [31:0] mem_path_B;
  reg [31:0] writeEnablea;
  reg [31:0] writeEnable;
  reg READ_WRITE, WRITE_WRITE, READ_READ, ROW_CC, COL_CC;
  reg READ_WRITE_1, WRITE_WRITE_1, READ_READ_1;

  wire  CENYA_;
  wire [31:0] WENYA_;
  wire [9:0] AYA_;
  wire  CENYB_;
  wire [31:0] WENYB_;
  wire [9:0] AYB_;
  wire  GWENYA_;
  wire  GWENYB_;
  wire [31:0] QA_;
  wire [31:0] QB_;
  wire [1:0] SOA_;
  wire [1:0] SOB_;
 wire  CLKA_;
  wire  CENA_;
  reg  CENA_int;
  reg  CENA_p2;
  wire [31:0] WENA_;
  reg [31:0] WENA_int;
  wire [9:0] AA_;
  reg [9:0] AA_int;
  wire [31:0] DA_;
  reg [31:0] DA_int;
 wire  CLKB_;
  wire  CENB_;
  reg  CENB_int;
  reg  CENB_p2;
  wire [31:0] WENB_;
  reg [31:0] WENB_int;
  wire [9:0] AB_;
  reg [9:0] AB_int;
  wire [31:0] DB_;
  reg [31:0] DB_int;
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
  wire [31:0] TWENA_;
  reg [31:0] TWENA_int;
  wire [9:0] TAA_;
  reg [9:0] TAA_int;
  wire [31:0] TDA_;
  reg [31:0] TDA_int;
  wire  TENB_;
  reg  TENB_int;
  wire  TCENB_;
  reg  TCENB_int;
  reg  TCENB_p2;
  wire [31:0] TWENB_;
  reg [31:0] TWENB_int;
  wire [9:0] TAB_;
  reg [9:0] TAB_int;
  wire [31:0] TDB_;
  reg [31:0] TDB_int;
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
  assign WENYA[8] = WENYA_[8]; 
  assign WENYA[9] = WENYA_[9]; 
  assign WENYA[10] = WENYA_[10]; 
  assign WENYA[11] = WENYA_[11]; 
  assign WENYA[12] = WENYA_[12]; 
  assign WENYA[13] = WENYA_[13]; 
  assign WENYA[14] = WENYA_[14]; 
  assign WENYA[15] = WENYA_[15]; 
  assign WENYA[16] = WENYA_[16]; 
  assign WENYA[17] = WENYA_[17]; 
  assign WENYA[18] = WENYA_[18]; 
  assign WENYA[19] = WENYA_[19]; 
  assign WENYA[20] = WENYA_[20]; 
  assign WENYA[21] = WENYA_[21]; 
  assign WENYA[22] = WENYA_[22]; 
  assign WENYA[23] = WENYA_[23]; 
  assign WENYA[24] = WENYA_[24]; 
  assign WENYA[25] = WENYA_[25]; 
  assign WENYA[26] = WENYA_[26]; 
  assign WENYA[27] = WENYA_[27]; 
  assign WENYA[28] = WENYA_[28]; 
  assign WENYA[29] = WENYA_[29]; 
  assign WENYA[30] = WENYA_[30]; 
  assign WENYA[31] = WENYA_[31]; 
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
  assign WENYB[8] = WENYB_[8]; 
  assign WENYB[9] = WENYB_[9]; 
  assign WENYB[10] = WENYB_[10]; 
  assign WENYB[11] = WENYB_[11]; 
  assign WENYB[12] = WENYB_[12]; 
  assign WENYB[13] = WENYB_[13]; 
  assign WENYB[14] = WENYB_[14]; 
  assign WENYB[15] = WENYB_[15]; 
  assign WENYB[16] = WENYB_[16]; 
  assign WENYB[17] = WENYB_[17]; 
  assign WENYB[18] = WENYB_[18]; 
  assign WENYB[19] = WENYB_[19]; 
  assign WENYB[20] = WENYB_[20]; 
  assign WENYB[21] = WENYB_[21]; 
  assign WENYB[22] = WENYB_[22]; 
  assign WENYB[23] = WENYB_[23]; 
  assign WENYB[24] = WENYB_[24]; 
  assign WENYB[25] = WENYB_[25]; 
  assign WENYB[26] = WENYB_[26]; 
  assign WENYB[27] = WENYB_[27]; 
  assign WENYB[28] = WENYB_[28]; 
  assign WENYB[29] = WENYB_[29]; 
  assign WENYB[30] = WENYB_[30]; 
  assign WENYB[31] = WENYB_[31]; 
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
  assign QA[8] = QA_[8]; 
  assign QA[9] = QA_[9]; 
  assign QA[10] = QA_[10]; 
  assign QA[11] = QA_[11]; 
  assign QA[12] = QA_[12]; 
  assign QA[13] = QA_[13]; 
  assign QA[14] = QA_[14]; 
  assign QA[15] = QA_[15]; 
  assign QA[16] = QA_[16]; 
  assign QA[17] = QA_[17]; 
  assign QA[18] = QA_[18]; 
  assign QA[19] = QA_[19]; 
  assign QA[20] = QA_[20]; 
  assign QA[21] = QA_[21]; 
  assign QA[22] = QA_[22]; 
  assign QA[23] = QA_[23]; 
  assign QA[24] = QA_[24]; 
  assign QA[25] = QA_[25]; 
  assign QA[26] = QA_[26]; 
  assign QA[27] = QA_[27]; 
  assign QA[28] = QA_[28]; 
  assign QA[29] = QA_[29]; 
  assign QA[30] = QA_[30]; 
  assign QA[31] = QA_[31]; 
  assign QB[0] = QB_[0]; 
  assign QB[1] = QB_[1]; 
  assign QB[2] = QB_[2]; 
  assign QB[3] = QB_[3]; 
  assign QB[4] = QB_[4]; 
  assign QB[5] = QB_[5]; 
  assign QB[6] = QB_[6]; 
  assign QB[7] = QB_[7]; 
  assign QB[8] = QB_[8]; 
  assign QB[9] = QB_[9]; 
  assign QB[10] = QB_[10]; 
  assign QB[11] = QB_[11]; 
  assign QB[12] = QB_[12]; 
  assign QB[13] = QB_[13]; 
  assign QB[14] = QB_[14]; 
  assign QB[15] = QB_[15]; 
  assign QB[16] = QB_[16]; 
  assign QB[17] = QB_[17]; 
  assign QB[18] = QB_[18]; 
  assign QB[19] = QB_[19]; 
  assign QB[20] = QB_[20]; 
  assign QB[21] = QB_[21]; 
  assign QB[22] = QB_[22]; 
  assign QB[23] = QB_[23]; 
  assign QB[24] = QB_[24]; 
  assign QB[25] = QB_[25]; 
  assign QB[26] = QB_[26]; 
  assign QB[27] = QB_[27]; 
  assign QB[28] = QB_[28]; 
  assign QB[29] = QB_[29]; 
  assign QB[30] = QB_[30]; 
  assign QB[31] = QB_[31]; 
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
  assign WENA_[8] = WENA[8];
  assign WENA_[9] = WENA[9];
  assign WENA_[10] = WENA[10];
  assign WENA_[11] = WENA[11];
  assign WENA_[12] = WENA[12];
  assign WENA_[13] = WENA[13];
  assign WENA_[14] = WENA[14];
  assign WENA_[15] = WENA[15];
  assign WENA_[16] = WENA[16];
  assign WENA_[17] = WENA[17];
  assign WENA_[18] = WENA[18];
  assign WENA_[19] = WENA[19];
  assign WENA_[20] = WENA[20];
  assign WENA_[21] = WENA[21];
  assign WENA_[22] = WENA[22];
  assign WENA_[23] = WENA[23];
  assign WENA_[24] = WENA[24];
  assign WENA_[25] = WENA[25];
  assign WENA_[26] = WENA[26];
  assign WENA_[27] = WENA[27];
  assign WENA_[28] = WENA[28];
  assign WENA_[29] = WENA[29];
  assign WENA_[30] = WENA[30];
  assign WENA_[31] = WENA[31];
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
  assign DA_[8] = DA[8];
  assign DA_[9] = DA[9];
  assign DA_[10] = DA[10];
  assign DA_[11] = DA[11];
  assign DA_[12] = DA[12];
  assign DA_[13] = DA[13];
  assign DA_[14] = DA[14];
  assign DA_[15] = DA[15];
  assign DA_[16] = DA[16];
  assign DA_[17] = DA[17];
  assign DA_[18] = DA[18];
  assign DA_[19] = DA[19];
  assign DA_[20] = DA[20];
  assign DA_[21] = DA[21];
  assign DA_[22] = DA[22];
  assign DA_[23] = DA[23];
  assign DA_[24] = DA[24];
  assign DA_[25] = DA[25];
  assign DA_[26] = DA[26];
  assign DA_[27] = DA[27];
  assign DA_[28] = DA[28];
  assign DA_[29] = DA[29];
  assign DA_[30] = DA[30];
  assign DA_[31] = DA[31];
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
  assign WENB_[8] = WENB[8];
  assign WENB_[9] = WENB[9];
  assign WENB_[10] = WENB[10];
  assign WENB_[11] = WENB[11];
  assign WENB_[12] = WENB[12];
  assign WENB_[13] = WENB[13];
  assign WENB_[14] = WENB[14];
  assign WENB_[15] = WENB[15];
  assign WENB_[16] = WENB[16];
  assign WENB_[17] = WENB[17];
  assign WENB_[18] = WENB[18];
  assign WENB_[19] = WENB[19];
  assign WENB_[20] = WENB[20];
  assign WENB_[21] = WENB[21];
  assign WENB_[22] = WENB[22];
  assign WENB_[23] = WENB[23];
  assign WENB_[24] = WENB[24];
  assign WENB_[25] = WENB[25];
  assign WENB_[26] = WENB[26];
  assign WENB_[27] = WENB[27];
  assign WENB_[28] = WENB[28];
  assign WENB_[29] = WENB[29];
  assign WENB_[30] = WENB[30];
  assign WENB_[31] = WENB[31];
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
  assign DB_[8] = DB[8];
  assign DB_[9] = DB[9];
  assign DB_[10] = DB[10];
  assign DB_[11] = DB[11];
  assign DB_[12] = DB[12];
  assign DB_[13] = DB[13];
  assign DB_[14] = DB[14];
  assign DB_[15] = DB[15];
  assign DB_[16] = DB[16];
  assign DB_[17] = DB[17];
  assign DB_[18] = DB[18];
  assign DB_[19] = DB[19];
  assign DB_[20] = DB[20];
  assign DB_[21] = DB[21];
  assign DB_[22] = DB[22];
  assign DB_[23] = DB[23];
  assign DB_[24] = DB[24];
  assign DB_[25] = DB[25];
  assign DB_[26] = DB[26];
  assign DB_[27] = DB[27];
  assign DB_[28] = DB[28];
  assign DB_[29] = DB[29];
  assign DB_[30] = DB[30];
  assign DB_[31] = DB[31];
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
  assign TWENA_[8] = TWENA[8];
  assign TWENA_[9] = TWENA[9];
  assign TWENA_[10] = TWENA[10];
  assign TWENA_[11] = TWENA[11];
  assign TWENA_[12] = TWENA[12];
  assign TWENA_[13] = TWENA[13];
  assign TWENA_[14] = TWENA[14];
  assign TWENA_[15] = TWENA[15];
  assign TWENA_[16] = TWENA[16];
  assign TWENA_[17] = TWENA[17];
  assign TWENA_[18] = TWENA[18];
  assign TWENA_[19] = TWENA[19];
  assign TWENA_[20] = TWENA[20];
  assign TWENA_[21] = TWENA[21];
  assign TWENA_[22] = TWENA[22];
  assign TWENA_[23] = TWENA[23];
  assign TWENA_[24] = TWENA[24];
  assign TWENA_[25] = TWENA[25];
  assign TWENA_[26] = TWENA[26];
  assign TWENA_[27] = TWENA[27];
  assign TWENA_[28] = TWENA[28];
  assign TWENA_[29] = TWENA[29];
  assign TWENA_[30] = TWENA[30];
  assign TWENA_[31] = TWENA[31];
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
  assign TDA_[8] = TDA[8];
  assign TDA_[9] = TDA[9];
  assign TDA_[10] = TDA[10];
  assign TDA_[11] = TDA[11];
  assign TDA_[12] = TDA[12];
  assign TDA_[13] = TDA[13];
  assign TDA_[14] = TDA[14];
  assign TDA_[15] = TDA[15];
  assign TDA_[16] = TDA[16];
  assign TDA_[17] = TDA[17];
  assign TDA_[18] = TDA[18];
  assign TDA_[19] = TDA[19];
  assign TDA_[20] = TDA[20];
  assign TDA_[21] = TDA[21];
  assign TDA_[22] = TDA[22];
  assign TDA_[23] = TDA[23];
  assign TDA_[24] = TDA[24];
  assign TDA_[25] = TDA[25];
  assign TDA_[26] = TDA[26];
  assign TDA_[27] = TDA[27];
  assign TDA_[28] = TDA[28];
  assign TDA_[29] = TDA[29];
  assign TDA_[30] = TDA[30];
  assign TDA_[31] = TDA[31];
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
  assign TWENB_[8] = TWENB[8];
  assign TWENB_[9] = TWENB[9];
  assign TWENB_[10] = TWENB[10];
  assign TWENB_[11] = TWENB[11];
  assign TWENB_[12] = TWENB[12];
  assign TWENB_[13] = TWENB[13];
  assign TWENB_[14] = TWENB[14];
  assign TWENB_[15] = TWENB[15];
  assign TWENB_[16] = TWENB[16];
  assign TWENB_[17] = TWENB[17];
  assign TWENB_[18] = TWENB[18];
  assign TWENB_[19] = TWENB[19];
  assign TWENB_[20] = TWENB[20];
  assign TWENB_[21] = TWENB[21];
  assign TWENB_[22] = TWENB[22];
  assign TWENB_[23] = TWENB[23];
  assign TWENB_[24] = TWENB[24];
  assign TWENB_[25] = TWENB[25];
  assign TWENB_[26] = TWENB[26];
  assign TWENB_[27] = TWENB[27];
  assign TWENB_[28] = TWENB[28];
  assign TWENB_[29] = TWENB[29];
  assign TWENB_[30] = TWENB[30];
  assign TWENB_[31] = TWENB[31];
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
  assign TDB_[8] = TDB[8];
  assign TDB_[9] = TDB[9];
  assign TDB_[10] = TDB[10];
  assign TDB_[11] = TDB[11];
  assign TDB_[12] = TDB[12];
  assign TDB_[13] = TDB[13];
  assign TDB_[14] = TDB[14];
  assign TDB_[15] = TDB[15];
  assign TDB_[16] = TDB[16];
  assign TDB_[17] = TDB[17];
  assign TDB_[18] = TDB[18];
  assign TDB_[19] = TDB[19];
  assign TDB_[20] = TDB[20];
  assign TDB_[21] = TDB[21];
  assign TDB_[22] = TDB[22];
  assign TDB_[23] = TDB[23];
  assign TDB_[24] = TDB[24];
  assign TDB_[25] = TDB[25];
  assign TDB_[26] = TDB[26];
  assign TDB_[27] = TDB[27];
  assign TDB_[28] = TDB[28];
  assign TDB_[29] = TDB[29];
  assign TDB_[30] = TDB[30];
  assign TDB_[31] = TDB[31];
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
  assign WENYA_ = RET1N_ ? ({32{DFTRAMBYP_}} & (TENA_ ? WENA_ : TWENA_)) : {32{1'bx}};
  assign AYA_ = RET1N_ ? ({10{DFTRAMBYP_}} & (TENA_ ? AA_ : TAA_)) : {10{1'bx}};
  assign CENYB_ = RET1N_ ? (DFTRAMBYP_ & (TENB_ ? CENB_ : TCENB_)) : 1'bx;
  assign WENYB_ = RET1N_ ? ({32{DFTRAMBYP_}} & (TENB_ ? WENB_ : TWENB_)) : {32{1'bx}};
  assign AYB_ = RET1N_ ? ({10{DFTRAMBYP_}} & (TENB_ ? AB_ : TAB_)) : {10{1'bx}};
  assign GWENYA_ = RET1N_ ? (DFTRAMBYP_ & (TENA_ ? GWENA_ : TGWENA_)) : 1'bx;
  assign GWENYB_ = RET1N_ ? (DFTRAMBYP_ & (TENB_ ? GWENB_ : TGWENB_)) : 1'bx;
  assign QA_ = RET1N_ ? ((QA_int)) : {32{1'bx}};
  assign QB_ = RET1N_ ? ((QB_int)) : {32{1'bx}};
  assign SOA_ = RET1N_ ? ({QA_[16], QA_[15]}) : {2{1'bx}};
  assign SOB_ = RET1N_ ? ({QB_[16], QB_[15]}) : {2{1'bx}};

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
        rowa = {512{1'bx}};
      else
        rowa = mem[row_addressa];
      end
`endif
        writeEnablea = ~ ( {32{GWENA_int}} | {WENA_int[31], WENA_int[30], WENA_int[29],
          WENA_int[28], WENA_int[27], WENA_int[26], WENA_int[25], WENA_int[24], WENA_int[23],
          WENA_int[22], WENA_int[21], WENA_int[20], WENA_int[19], WENA_int[18], WENA_int[17],
          WENA_int[16], WENA_int[15], WENA_int[14], WENA_int[13], WENA_int[12], WENA_int[11],
          WENA_int[10], WENA_int[9], WENA_int[8], WENA_int[7], WENA_int[6], WENA_int[5],
          WENA_int[4], WENA_int[3], WENA_int[2], WENA_int[1], WENA_int[0]});
      if (GWENA_int !== 1'b1 || DFTRAMBYP_inta === 1'b1) begin
        row_maska =  ( {15'b000000000000000, writeEnablea[31], 15'b000000000000000, writeEnablea[30],
          15'b000000000000000, writeEnablea[29], 15'b000000000000000, writeEnablea[28],
          15'b000000000000000, writeEnablea[27], 15'b000000000000000, writeEnablea[26],
          15'b000000000000000, writeEnablea[25], 15'b000000000000000, writeEnablea[24],
          15'b000000000000000, writeEnablea[23], 15'b000000000000000, writeEnablea[22],
          15'b000000000000000, writeEnablea[21], 15'b000000000000000, writeEnablea[20],
          15'b000000000000000, writeEnablea[19], 15'b000000000000000, writeEnablea[18],
          15'b000000000000000, writeEnablea[17], 15'b000000000000000, writeEnablea[16],
          15'b000000000000000, writeEnablea[15], 15'b000000000000000, writeEnablea[14],
          15'b000000000000000, writeEnablea[13], 15'b000000000000000, writeEnablea[12],
          15'b000000000000000, writeEnablea[11], 15'b000000000000000, writeEnablea[10],
          15'b000000000000000, writeEnablea[9], 15'b000000000000000, writeEnablea[8],
          15'b000000000000000, writeEnablea[7], 15'b000000000000000, writeEnablea[6],
          15'b000000000000000, writeEnablea[5], 15'b000000000000000, writeEnablea[4],
          15'b000000000000000, writeEnablea[3], 15'b000000000000000, writeEnablea[2],
          15'b000000000000000, writeEnablea[1], 15'b000000000000000, writeEnablea[0]} << mux_addressa);
        new_dataa =  ( {15'b000000000000000, DA_int[31], 15'b000000000000000, DA_int[30],
          15'b000000000000000, DA_int[29], 15'b000000000000000, DA_int[28], 15'b000000000000000, DA_int[27],
          15'b000000000000000, DA_int[26], 15'b000000000000000, DA_int[25], 15'b000000000000000, DA_int[24],
          15'b000000000000000, DA_int[23], 15'b000000000000000, DA_int[22], 15'b000000000000000, DA_int[21],
          15'b000000000000000, DA_int[20], 15'b000000000000000, DA_int[19], 15'b000000000000000, DA_int[18],
          15'b000000000000000, DA_int[17], 15'b000000000000000, DA_int[16], 15'b000000000000000, DA_int[15],
          15'b000000000000000, DA_int[14], 15'b000000000000000, DA_int[13], 15'b000000000000000, DA_int[12],
          15'b000000000000000, DA_int[11], 15'b000000000000000, DA_int[10], 15'b000000000000000, DA_int[9],
          15'b000000000000000, DA_int[8], 15'b000000000000000, DA_int[7], 15'b000000000000000, DA_int[6],
          15'b000000000000000, DA_int[5], 15'b000000000000000, DA_int[4], 15'b000000000000000, DA_int[3],
          15'b000000000000000, DA_int[2], 15'b000000000000000, DA_int[1], 15'b000000000000000, DA_int[0]} << mux_addressa);
        rowa = (rowa & ~row_maska) | (row_maska & (~row_maska | new_dataa));
        if (DFTRAMBYP_inta === 1'b1 && SEA_int === 1'b0) begin
        end else begin
        mem[row_addressa] = rowa;
        end
      end else begin
        data_outa = (rowa >> (mux_addressa%4));
        readLatch0 = {data_outa[508], data_outa[504], data_outa[500], data_outa[496],
          data_outa[492], data_outa[488], data_outa[484], data_outa[480], data_outa[476],
          data_outa[472], data_outa[468], data_outa[464], data_outa[460], data_outa[456],
          data_outa[452], data_outa[448], data_outa[444], data_outa[440], data_outa[436],
          data_outa[432], data_outa[428], data_outa[424], data_outa[420], data_outa[416],
          data_outa[412], data_outa[408], data_outa[404], data_outa[400], data_outa[396],
          data_outa[392], data_outa[388], data_outa[384], data_outa[380], data_outa[376],
          data_outa[372], data_outa[368], data_outa[364], data_outa[360], data_outa[356],
          data_outa[352], data_outa[348], data_outa[344], data_outa[340], data_outa[336],
          data_outa[332], data_outa[328], data_outa[324], data_outa[320], data_outa[316],
          data_outa[312], data_outa[308], data_outa[304], data_outa[300], data_outa[296],
          data_outa[292], data_outa[288], data_outa[284], data_outa[280], data_outa[276],
          data_outa[272], data_outa[268], data_outa[264], data_outa[260], data_outa[256],
          data_outa[252], data_outa[248], data_outa[244], data_outa[240], data_outa[236],
          data_outa[232], data_outa[228], data_outa[224], data_outa[220], data_outa[216],
          data_outa[212], data_outa[208], data_outa[204], data_outa[200], data_outa[196],
          data_outa[192], data_outa[188], data_outa[184], data_outa[180], data_outa[176],
          data_outa[172], data_outa[168], data_outa[164], data_outa[160], data_outa[156],
          data_outa[152], data_outa[148], data_outa[144], data_outa[140], data_outa[136],
          data_outa[132], data_outa[128], data_outa[124], data_outa[120], data_outa[116],
          data_outa[112], data_outa[108], data_outa[104], data_outa[100], data_outa[96],
          data_outa[92], data_outa[88], data_outa[84], data_outa[80], data_outa[76],
          data_outa[72], data_outa[68], data_outa[64], data_outa[60], data_outa[56],
          data_outa[52], data_outa[48], data_outa[44], data_outa[40], data_outa[36],
          data_outa[32], data_outa[28], data_outa[24], data_outa[20], data_outa[16],
          data_outa[12], data_outa[8], data_outa[4], data_outa[0]};
        shifted_readLatch0 = (readLatch0 >> AA_int[3:2]);
        mem_path_A = {shifted_readLatch0[124], shifted_readLatch0[120], shifted_readLatch0[116],
          shifted_readLatch0[112], shifted_readLatch0[108], shifted_readLatch0[104],
          shifted_readLatch0[100], shifted_readLatch0[96], shifted_readLatch0[92],
          shifted_readLatch0[88], shifted_readLatch0[84], shifted_readLatch0[80], shifted_readLatch0[76],
          shifted_readLatch0[72], shifted_readLatch0[68], shifted_readLatch0[64], shifted_readLatch0[60],
          shifted_readLatch0[56], shifted_readLatch0[52], shifted_readLatch0[48], shifted_readLatch0[44],
          shifted_readLatch0[40], shifted_readLatch0[36], shifted_readLatch0[32], shifted_readLatch0[28],
          shifted_readLatch0[24], shifted_readLatch0[20], shifted_readLatch0[16], shifted_readLatch0[12],
          shifted_readLatch0[8], shifted_readLatch0[4], shifted_readLatch0[0]};
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
      if ((1) && (CENA_int !== 1'b1 && CENB_int !== 1'b1 && DFTRAMBYP_ !== 1'b1) && COLLDISN_int === 1'b1 && row_contention(AA_int, AB_int, ({32{GWENA_int}}|WENA_int),
        ({32{GWENB_int}}|WENB_int))) begin
        if (col_contention(AA_int, AB_int)) begin
          COL_CC = 1;
        end
          ROW_CC = 1;
          READ_READ_1 = 0;
          READ_WRITE_1 = 0;
          WRITE_WRITE_1 = 0;
        if (GWENA_int !== 1'b1 && (& WENA_int) !== 1'b1 && GWENB_int !== 1'b1 && (& WENB_int) !== 1'b1) begin
	      if (is_contention(AA_int, AB_int, ({32{GWENA_int}}|WENA_int), ({32{GWENB_int}}|WENB_int))) begin
          COL_CC = 1;
          WRITE_WRITE = 1;
	      end
        end else if (GWENA_int !== 1'b1 && (& WENA_int) !== 1'b1) begin
		if (is_contention(AA_int, AB_int, ({32{GWENA_int}}|WENA_int), ({32{GWENB_int}}|WENB_int))) begin
          COL_CC = 1;
          READ_WRITE = 1;
		end
        end else if (GWENB_int !== 1'b1 && (& WENB_int) !== 1'b1) begin
		if (is_contention(AA_int, AB_int, ({32{GWENA_int}}|WENA_int), ({32{GWENB_int}}|WENB_int))) begin
          COL_CC = 1;
          READ_WRITE = 1;
		end
        end else begin
          COL_CC = 1;
          READ_READ = 1;
        end
        if (!is_contention(AA_int, AB_int, ({32{GWENA_int}}|WENA_int), ({32{GWENB_int}}|WENB_int))) begin
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
        AB_int, ({32{GWENA_int}}|WENA_int), ({32{GWENB_int}}|WENB_int))) begin
          ROW_CC = 1;
          READ_READ_1 = 0;
          READ_WRITE_1 = 0;
          WRITE_WRITE_1 = 0;
        if (col_contention(AA_int, AB_int)) begin
          COL_CC = 1;
        end
        if (GWENB_int !== 1'b1 && (& WENB_int) !== 1'b1) begin
          WRITE_WRITE_1 = 1;
        end else if (is_contention(AA_int, AB_int, ({32{GWENA_int}}|WENA_int), ({32{GWENB_int}}|WENB_int))) begin
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
        end else if (is_contention(AA_int, AB_int, ({32{GWENA_int}}|WENA_int), ({32{GWENB_int}}|WENB_int))) begin
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

  datapath_latch_sram_32_1024_dp uDQA0 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(SIA_int[0]), .D(DA_int_bmux[0]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[0]), .XQ(XQA), .Q(QA_int[0]));
  datapath_latch_sram_32_1024_dp uDQA1 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[0]), .D(DA_int_bmux[1]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[1]), .XQ(XQA), .Q(QA_int[1]));
  datapath_latch_sram_32_1024_dp uDQA2 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[1]), .D(DA_int_bmux[2]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[2]), .XQ(XQA), .Q(QA_int[2]));
  datapath_latch_sram_32_1024_dp uDQA3 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[2]), .D(DA_int_bmux[3]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[3]), .XQ(XQA), .Q(QA_int[3]));
  datapath_latch_sram_32_1024_dp uDQA4 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[3]), .D(DA_int_bmux[4]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[4]), .XQ(XQA), .Q(QA_int[4]));
  datapath_latch_sram_32_1024_dp uDQA5 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[4]), .D(DA_int_bmux[5]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[5]), .XQ(XQA), .Q(QA_int[5]));
  datapath_latch_sram_32_1024_dp uDQA6 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[5]), .D(DA_int_bmux[6]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[6]), .XQ(XQA), .Q(QA_int[6]));
  datapath_latch_sram_32_1024_dp uDQA7 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[6]), .D(DA_int_bmux[7]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[7]), .XQ(XQA), .Q(QA_int[7]));
  datapath_latch_sram_32_1024_dp uDQA8 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[7]), .D(DA_int_bmux[8]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[8]), .XQ(XQA), .Q(QA_int[8]));
  datapath_latch_sram_32_1024_dp uDQA9 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[8]), .D(DA_int_bmux[9]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[9]), .XQ(XQA), .Q(QA_int[9]));
  datapath_latch_sram_32_1024_dp uDQA10 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[9]), .D(DA_int_bmux[10]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[10]), .XQ(XQA), .Q(QA_int[10]));
  datapath_latch_sram_32_1024_dp uDQA11 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[10]), .D(DA_int_bmux[11]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[11]), .XQ(XQA), .Q(QA_int[11]));
  datapath_latch_sram_32_1024_dp uDQA12 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[11]), .D(DA_int_bmux[12]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[12]), .XQ(XQA), .Q(QA_int[12]));
  datapath_latch_sram_32_1024_dp uDQA13 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[12]), .D(DA_int_bmux[13]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[13]), .XQ(XQA), .Q(QA_int[13]));
  datapath_latch_sram_32_1024_dp uDQA14 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[13]), .D(DA_int_bmux[14]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[14]), .XQ(XQA), .Q(QA_int[14]));
  datapath_latch_sram_32_1024_dp uDQA15 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[14]), .D(DA_int_bmux[15]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[15]), .XQ(XQA), .Q(QA_int[15]));
  datapath_latch_sram_32_1024_dp uDQA16 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[17]), .D(DA_int_bmux[16]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[16]), .XQ(XQA), .Q(QA_int[16]));
  datapath_latch_sram_32_1024_dp uDQA17 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[18]), .D(DA_int_bmux[17]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[17]), .XQ(XQA), .Q(QA_int[17]));
  datapath_latch_sram_32_1024_dp uDQA18 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[19]), .D(DA_int_bmux[18]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[18]), .XQ(XQA), .Q(QA_int[18]));
  datapath_latch_sram_32_1024_dp uDQA19 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[20]), .D(DA_int_bmux[19]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[19]), .XQ(XQA), .Q(QA_int[19]));
  datapath_latch_sram_32_1024_dp uDQA20 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[21]), .D(DA_int_bmux[20]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[20]), .XQ(XQA), .Q(QA_int[20]));
  datapath_latch_sram_32_1024_dp uDQA21 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[22]), .D(DA_int_bmux[21]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[21]), .XQ(XQA), .Q(QA_int[21]));
  datapath_latch_sram_32_1024_dp uDQA22 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[23]), .D(DA_int_bmux[22]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[22]), .XQ(XQA), .Q(QA_int[22]));
  datapath_latch_sram_32_1024_dp uDQA23 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[24]), .D(DA_int_bmux[23]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[23]), .XQ(XQA), .Q(QA_int[23]));
  datapath_latch_sram_32_1024_dp uDQA24 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[25]), .D(DA_int_bmux[24]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[24]), .XQ(XQA), .Q(QA_int[24]));
  datapath_latch_sram_32_1024_dp uDQA25 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[26]), .D(DA_int_bmux[25]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[25]), .XQ(XQA), .Q(QA_int[25]));
  datapath_latch_sram_32_1024_dp uDQA26 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[27]), .D(DA_int_bmux[26]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[26]), .XQ(XQA), .Q(QA_int[26]));
  datapath_latch_sram_32_1024_dp uDQA27 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[28]), .D(DA_int_bmux[27]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[27]), .XQ(XQA), .Q(QA_int[27]));
  datapath_latch_sram_32_1024_dp uDQA28 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[29]), .D(DA_int_bmux[28]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[28]), .XQ(XQA), .Q(QA_int[28]));
  datapath_latch_sram_32_1024_dp uDQA29 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[30]), .D(DA_int_bmux[29]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[29]), .XQ(XQA), .Q(QA_int[29]));
  datapath_latch_sram_32_1024_dp uDQA30 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[31]), .D(DA_int_bmux[30]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[30]), .XQ(XQA), .Q(QA_int[30]));
  datapath_latch_sram_32_1024_dp uDQA31 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(SIA_int[1]), .D(DA_int_bmux[31]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[31]), .XQ(XQA), .Q(QA_int[31]));


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
        row = {512{1'bx}};
      else
        row = mem[row_address];
      end
`endif
        writeEnable = ~ ( {32{GWENB_int}} | {WENB_int[31], WENB_int[30], WENB_int[29],
          WENB_int[28], WENB_int[27], WENB_int[26], WENB_int[25], WENB_int[24], WENB_int[23],
          WENB_int[22], WENB_int[21], WENB_int[20], WENB_int[19], WENB_int[18], WENB_int[17],
          WENB_int[16], WENB_int[15], WENB_int[14], WENB_int[13], WENB_int[12], WENB_int[11],
          WENB_int[10], WENB_int[9], WENB_int[8], WENB_int[7], WENB_int[6], WENB_int[5],
          WENB_int[4], WENB_int[3], WENB_int[2], WENB_int[1], WENB_int[0]});
      if (GWENB_int !== 1'b1 || DFTRAMBYP_int === 1'b1) begin
        row_mask =  ( {15'b000000000000000, writeEnable[31], 15'b000000000000000, writeEnable[30],
          15'b000000000000000, writeEnable[29], 15'b000000000000000, writeEnable[28],
          15'b000000000000000, writeEnable[27], 15'b000000000000000, writeEnable[26],
          15'b000000000000000, writeEnable[25], 15'b000000000000000, writeEnable[24],
          15'b000000000000000, writeEnable[23], 15'b000000000000000, writeEnable[22],
          15'b000000000000000, writeEnable[21], 15'b000000000000000, writeEnable[20],
          15'b000000000000000, writeEnable[19], 15'b000000000000000, writeEnable[18],
          15'b000000000000000, writeEnable[17], 15'b000000000000000, writeEnable[16],
          15'b000000000000000, writeEnable[15], 15'b000000000000000, writeEnable[14],
          15'b000000000000000, writeEnable[13], 15'b000000000000000, writeEnable[12],
          15'b000000000000000, writeEnable[11], 15'b000000000000000, writeEnable[10],
          15'b000000000000000, writeEnable[9], 15'b000000000000000, writeEnable[8],
          15'b000000000000000, writeEnable[7], 15'b000000000000000, writeEnable[6],
          15'b000000000000000, writeEnable[5], 15'b000000000000000, writeEnable[4],
          15'b000000000000000, writeEnable[3], 15'b000000000000000, writeEnable[2],
          15'b000000000000000, writeEnable[1], 15'b000000000000000, writeEnable[0]} << mux_address);
        new_data =  ( {15'b000000000000000, DB_int[31], 15'b000000000000000, DB_int[30],
          15'b000000000000000, DB_int[29], 15'b000000000000000, DB_int[28], 15'b000000000000000, DB_int[27],
          15'b000000000000000, DB_int[26], 15'b000000000000000, DB_int[25], 15'b000000000000000, DB_int[24],
          15'b000000000000000, DB_int[23], 15'b000000000000000, DB_int[22], 15'b000000000000000, DB_int[21],
          15'b000000000000000, DB_int[20], 15'b000000000000000, DB_int[19], 15'b000000000000000, DB_int[18],
          15'b000000000000000, DB_int[17], 15'b000000000000000, DB_int[16], 15'b000000000000000, DB_int[15],
          15'b000000000000000, DB_int[14], 15'b000000000000000, DB_int[13], 15'b000000000000000, DB_int[12],
          15'b000000000000000, DB_int[11], 15'b000000000000000, DB_int[10], 15'b000000000000000, DB_int[9],
          15'b000000000000000, DB_int[8], 15'b000000000000000, DB_int[7], 15'b000000000000000, DB_int[6],
          15'b000000000000000, DB_int[5], 15'b000000000000000, DB_int[4], 15'b000000000000000, DB_int[3],
          15'b000000000000000, DB_int[2], 15'b000000000000000, DB_int[1], 15'b000000000000000, DB_int[0]} << mux_address);
        row = (row & ~row_mask) | (row_mask & (~row_mask | new_data));
        if (DFTRAMBYP_int === 1'b1 && SEB_int === 1'b0) begin
        end else begin
        mem[row_address] = row;
        end
      end else begin
        data_out = (row >> (mux_address%4));
        readLatch1 = {data_out[508], data_out[504], data_out[500], data_out[496], data_out[492],
          data_out[488], data_out[484], data_out[480], data_out[476], data_out[472],
          data_out[468], data_out[464], data_out[460], data_out[456], data_out[452],
          data_out[448], data_out[444], data_out[440], data_out[436], data_out[432],
          data_out[428], data_out[424], data_out[420], data_out[416], data_out[412],
          data_out[408], data_out[404], data_out[400], data_out[396], data_out[392],
          data_out[388], data_out[384], data_out[380], data_out[376], data_out[372],
          data_out[368], data_out[364], data_out[360], data_out[356], data_out[352],
          data_out[348], data_out[344], data_out[340], data_out[336], data_out[332],
          data_out[328], data_out[324], data_out[320], data_out[316], data_out[312],
          data_out[308], data_out[304], data_out[300], data_out[296], data_out[292],
          data_out[288], data_out[284], data_out[280], data_out[276], data_out[272],
          data_out[268], data_out[264], data_out[260], data_out[256], data_out[252],
          data_out[248], data_out[244], data_out[240], data_out[236], data_out[232],
          data_out[228], data_out[224], data_out[220], data_out[216], data_out[212],
          data_out[208], data_out[204], data_out[200], data_out[196], data_out[192],
          data_out[188], data_out[184], data_out[180], data_out[176], data_out[172],
          data_out[168], data_out[164], data_out[160], data_out[156], data_out[152],
          data_out[148], data_out[144], data_out[140], data_out[136], data_out[132],
          data_out[128], data_out[124], data_out[120], data_out[116], data_out[112],
          data_out[108], data_out[104], data_out[100], data_out[96], data_out[92],
          data_out[88], data_out[84], data_out[80], data_out[76], data_out[72], data_out[68],
          data_out[64], data_out[60], data_out[56], data_out[52], data_out[48], data_out[44],
          data_out[40], data_out[36], data_out[32], data_out[28], data_out[24], data_out[20],
          data_out[16], data_out[12], data_out[8], data_out[4], data_out[0]};
        shifted_readLatch1 = (readLatch1 >> AB_int[3:2]);
        mem_path_B = {shifted_readLatch1[124], shifted_readLatch1[120], shifted_readLatch1[116],
          shifted_readLatch1[112], shifted_readLatch1[108], shifted_readLatch1[104],
          shifted_readLatch1[100], shifted_readLatch1[96], shifted_readLatch1[92],
          shifted_readLatch1[88], shifted_readLatch1[84], shifted_readLatch1[80], shifted_readLatch1[76],
          shifted_readLatch1[72], shifted_readLatch1[68], shifted_readLatch1[64], shifted_readLatch1[60],
          shifted_readLatch1[56], shifted_readLatch1[52], shifted_readLatch1[48], shifted_readLatch1[44],
          shifted_readLatch1[40], shifted_readLatch1[36], shifted_readLatch1[32], shifted_readLatch1[28],
          shifted_readLatch1[24], shifted_readLatch1[20], shifted_readLatch1[16], shifted_readLatch1[12],
          shifted_readLatch1[8], shifted_readLatch1[4], shifted_readLatch1[0]};
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
      if ((1) && (CENA_int !== 1'b1 && CENB_int !== 1'b1 && DFTRAMBYP_ !== 1'b1) && COLLDISN_int === 1'b1 && row_contention(AA_int, AB_int, ({32{GWENA_int}}|WENA_int),
        ({32{GWENB_int}}|WENB_int))) begin
        if (col_contention(AA_int, AB_int)) begin
          COL_CC = 1;
        end
          ROW_CC = 1;
          READ_READ_1 = 0;
          READ_WRITE_1 = 0;
          WRITE_WRITE_1 = 0;
        if (GWENA_int !== 1'b1 && (& WENA_int) !== 1'b1 && GWENB_int !== 1'b1 && (& WENB_int) !== 1'b1) begin
	      if (is_contention(AA_int, AB_int, ({32{GWENA_int}}|WENA_int), ({32{GWENB_int}}|WENB_int))) begin
          COL_CC = 1;
          WRITE_WRITE = 1;
	      end
        end else if (GWENA_int !== 1'b1 && (& WENA_int) !== 1'b1) begin
		if (is_contention(AA_int, AB_int, ({32{GWENA_int}}|WENA_int), ({32{GWENB_int}}|WENB_int))) begin
          COL_CC = 1;
          READ_WRITE = 1;
		end
        end else if (GWENB_int !== 1'b1 && (& WENB_int) !== 1'b1) begin
		if (is_contention(AA_int, AB_int, ({32{GWENA_int}}|WENA_int), ({32{GWENB_int}}|WENB_int))) begin
          COL_CC = 1;
          READ_WRITE = 1;
		end
        end else begin
          COL_CC = 1;
          READ_READ = 1;
        end
        if (!is_contention(AA_int, AB_int, ({32{GWENA_int}}|WENA_int), ({32{GWENB_int}}|WENB_int))) begin
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
        AB_int, ({32{GWENA_int}}|WENA_int), ({32{GWENB_int}}|WENB_int))) begin
          ROW_CC = 1;
          READ_READ_1 = 0;
          READ_WRITE_1 = 0;
          WRITE_WRITE_1 = 0;
        if (col_contention(AA_int, AB_int)) begin
          COL_CC = 1;
        end
        if (GWENA_int !== 1'b1 && (& WENA_int) !== 1'b1) begin
          WRITE_WRITE_1 = 1;
        end else if (is_contention(AA_int, AB_int, ({32{GWENA_int}}|WENA_int), ({32{GWENB_int}}|WENB_int))) begin
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
        end else if (is_contention(AA_int, AB_int, ({32{GWENA_int}}|WENA_int), ({32{GWENB_int}}|WENB_int))) begin
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

  datapath_latch_sram_32_1024_dp uDQB0 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(SIB_int[0]), .D(DB_int_bmux[0]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[0]), .XQ(XQB), .Q(QB_int[0]));
  datapath_latch_sram_32_1024_dp uDQB1 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[0]), .D(DB_int_bmux[1]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[1]), .XQ(XQB), .Q(QB_int[1]));
  datapath_latch_sram_32_1024_dp uDQB2 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[1]), .D(DB_int_bmux[2]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[2]), .XQ(XQB), .Q(QB_int[2]));
  datapath_latch_sram_32_1024_dp uDQB3 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[2]), .D(DB_int_bmux[3]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[3]), .XQ(XQB), .Q(QB_int[3]));
  datapath_latch_sram_32_1024_dp uDQB4 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[3]), .D(DB_int_bmux[4]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[4]), .XQ(XQB), .Q(QB_int[4]));
  datapath_latch_sram_32_1024_dp uDQB5 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[4]), .D(DB_int_bmux[5]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[5]), .XQ(XQB), .Q(QB_int[5]));
  datapath_latch_sram_32_1024_dp uDQB6 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[5]), .D(DB_int_bmux[6]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[6]), .XQ(XQB), .Q(QB_int[6]));
  datapath_latch_sram_32_1024_dp uDQB7 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[6]), .D(DB_int_bmux[7]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[7]), .XQ(XQB), .Q(QB_int[7]));
  datapath_latch_sram_32_1024_dp uDQB8 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[7]), .D(DB_int_bmux[8]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[8]), .XQ(XQB), .Q(QB_int[8]));
  datapath_latch_sram_32_1024_dp uDQB9 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[8]), .D(DB_int_bmux[9]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[9]), .XQ(XQB), .Q(QB_int[9]));
  datapath_latch_sram_32_1024_dp uDQB10 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[9]), .D(DB_int_bmux[10]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[10]), .XQ(XQB), .Q(QB_int[10]));
  datapath_latch_sram_32_1024_dp uDQB11 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[10]), .D(DB_int_bmux[11]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[11]), .XQ(XQB), .Q(QB_int[11]));
  datapath_latch_sram_32_1024_dp uDQB12 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[11]), .D(DB_int_bmux[12]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[12]), .XQ(XQB), .Q(QB_int[12]));
  datapath_latch_sram_32_1024_dp uDQB13 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[12]), .D(DB_int_bmux[13]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[13]), .XQ(XQB), .Q(QB_int[13]));
  datapath_latch_sram_32_1024_dp uDQB14 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[13]), .D(DB_int_bmux[14]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[14]), .XQ(XQB), .Q(QB_int[14]));
  datapath_latch_sram_32_1024_dp uDQB15 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[14]), .D(DB_int_bmux[15]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[15]), .XQ(XQB), .Q(QB_int[15]));
  datapath_latch_sram_32_1024_dp uDQB16 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[17]), .D(DB_int_bmux[16]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[16]), .XQ(XQB), .Q(QB_int[16]));
  datapath_latch_sram_32_1024_dp uDQB17 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[18]), .D(DB_int_bmux[17]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[17]), .XQ(XQB), .Q(QB_int[17]));
  datapath_latch_sram_32_1024_dp uDQB18 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[19]), .D(DB_int_bmux[18]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[18]), .XQ(XQB), .Q(QB_int[18]));
  datapath_latch_sram_32_1024_dp uDQB19 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[20]), .D(DB_int_bmux[19]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[19]), .XQ(XQB), .Q(QB_int[19]));
  datapath_latch_sram_32_1024_dp uDQB20 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[21]), .D(DB_int_bmux[20]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[20]), .XQ(XQB), .Q(QB_int[20]));
  datapath_latch_sram_32_1024_dp uDQB21 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[22]), .D(DB_int_bmux[21]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[21]), .XQ(XQB), .Q(QB_int[21]));
  datapath_latch_sram_32_1024_dp uDQB22 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[23]), .D(DB_int_bmux[22]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[22]), .XQ(XQB), .Q(QB_int[22]));
  datapath_latch_sram_32_1024_dp uDQB23 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[24]), .D(DB_int_bmux[23]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[23]), .XQ(XQB), .Q(QB_int[23]));
  datapath_latch_sram_32_1024_dp uDQB24 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[25]), .D(DB_int_bmux[24]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[24]), .XQ(XQB), .Q(QB_int[24]));
  datapath_latch_sram_32_1024_dp uDQB25 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[26]), .D(DB_int_bmux[25]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[25]), .XQ(XQB), .Q(QB_int[25]));
  datapath_latch_sram_32_1024_dp uDQB26 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[27]), .D(DB_int_bmux[26]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[26]), .XQ(XQB), .Q(QB_int[26]));
  datapath_latch_sram_32_1024_dp uDQB27 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[28]), .D(DB_int_bmux[27]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[27]), .XQ(XQB), .Q(QB_int[27]));
  datapath_latch_sram_32_1024_dp uDQB28 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[29]), .D(DB_int_bmux[28]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[28]), .XQ(XQB), .Q(QB_int[28]));
  datapath_latch_sram_32_1024_dp uDQB29 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[30]), .D(DB_int_bmux[29]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[29]), .XQ(XQB), .Q(QB_int[29]));
  datapath_latch_sram_32_1024_dp uDQB30 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[31]), .D(DB_int_bmux[30]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[30]), .XQ(XQB), .Q(QB_int[30]));
  datapath_latch_sram_32_1024_dp uDQB31 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(SIB_int[1]), .D(DB_int_bmux[31]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[31]), .XQ(XQB), .Q(QB_int[31]));



  function row_contention;
    input [9:0] aa;
    input [9:0] ab;
    input [31:0] wena;
    input [31:0] wenb;
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
    input [31:0] wena;
    input [31:0] wenb;
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
