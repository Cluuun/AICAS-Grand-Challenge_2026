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
//       Instance Name:              sram_128_1024_dp
//       Words:                      1024
//       Bits:                       128
//       Mux:                        4
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
//       Creation Date:  Thu Dec  4 21:29:50 2025
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

module datapath_latch_sram_128_1024_dp (CLK,Q_update,D_update,SE,SI,D,DFTRAMBYP,mem_path,XQ,Q);
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
endmodule // datapath_latch_sram_128_1024_dp

// If POWER_PINS is defined at Simulator Command Line, it selects the module definition with Power Ports
`ifdef POWER_PINS
module sram_128_1024_dp (VDDCE, VDDPE, VSSE, CENYA, WENYA, AYA, CENYB, WENYB, AYB,
    GWENYA, GWENYB, QA, QB, SOA, SOB, CLKA, CENA, WENA, AA, DA, CLKB, CENB, WENB, AB,
    DB, EMAA, EMAWA, EMASA, EMAB, EMAWB, EMASB, TENA, TCENA, TWENA, TAA, TDA, TENB,
    TCENB, TWENB, TAB, TDB, GWENA, GWENB, TGWENA, TGWENB, RET1N, SIA, SEA, DFTRAMBYP,
    SIB, SEB, COLLDISN);
`else
module sram_128_1024_dp (CENYA, WENYA, AYA, CENYB, WENYB, AYB, GWENYA, GWENYB, QA,
    QB, SOA, SOB, CLKA, CENA, WENA, AA, DA, CLKB, CENB, WENB, AB, DB, EMAA, EMAWA,
    EMASA, EMAB, EMAWB, EMASB, TENA, TCENA, TWENA, TAA, TDA, TENB, TCENB, TWENB, TAB,
    TDB, GWENA, GWENB, TGWENA, TGWENB, RET1N, SIA, SEA, DFTRAMBYP, SIB, SEB, COLLDISN);
`endif

  parameter ASSERT_PREFIX = "";
  parameter BITS = 128;
  parameter WORDS = 1024;
  parameter MUX = 4;
  parameter MEM_WIDTH = 512; // redun block size 4, 256 on left, 256 on right
  parameter MEM_HEIGHT = 256;
  parameter WP_SIZE = 1 ;
  parameter UPM_WIDTH = 3;
  parameter UPMW_WIDTH = 2;
  parameter UPMS_WIDTH = 1;

  output  CENYA;
  output [127:0] WENYA;
  output [9:0] AYA;
  output  CENYB;
  output [127:0] WENYB;
  output [9:0] AYB;
  output  GWENYA;
  output  GWENYB;
  output [127:0] QA;
  output [127:0] QB;
  output [1:0] SOA;
  output [1:0] SOB;
  input  CLKA;
  input  CENA;
  input [127:0] WENA;
  input [9:0] AA;
  input [127:0] DA;
  input  CLKB;
  input  CENB;
  input [127:0] WENB;
  input [9:0] AB;
  input [127:0] DB;
  input [2:0] EMAA;
  input [1:0] EMAWA;
  input  EMASA;
  input [2:0] EMAB;
  input [1:0] EMAWB;
  input  EMASB;
  input  TENA;
  input  TCENA;
  input [127:0] TWENA;
  input [9:0] TAA;
  input [127:0] TDA;
  input  TENB;
  input  TCENB;
  input [127:0] TWENB;
  input [9:0] TAB;
  input [127:0] TDB;
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
  reg [511:0] mem [0:255];
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
  reg  read_mux_sel0_p2;
  reg [127:0] readLatch1;
  reg [127:0] shifted_readLatch1;
  reg  read_mux_sel1_p2;
  reg LAST_CLKB;
  wire [127:0] QA_int;
  reg XQA, QA_update;
  reg XDA_sh, DA_sh_update;
  wire [127:0] DA_int_bmux;
  reg [127:0] mem_path_A;
  wire [127:0] QB_int;
  reg XQB, QB_update;
  reg XDB_sh, DB_sh_update;
  wire [127:0] DB_int_bmux;
  reg [127:0] mem_path_B;
  reg [127:0] writeEnablea;
  reg [127:0] writeEnable;
  reg READ_WRITE, WRITE_WRITE, READ_READ, ROW_CC, COL_CC;
  reg READ_WRITE_1, WRITE_WRITE_1, READ_READ_1;

  wire  CENYA_;
  wire [127:0] WENYA_;
  wire [9:0] AYA_;
  wire  CENYB_;
  wire [127:0] WENYB_;
  wire [9:0] AYB_;
  wire  GWENYA_;
  wire  GWENYB_;
  wire [127:0] QA_;
  wire [127:0] QB_;
  wire [1:0] SOA_;
  wire [1:0] SOB_;
 wire  CLKA_;
  wire  CENA_;
  reg  CENA_int;
  reg  CENA_p2;
  wire [127:0] WENA_;
  reg [127:0] WENA_int;
  wire [9:0] AA_;
  reg [9:0] AA_int;
  wire [127:0] DA_;
  reg [127:0] DA_int;
 wire  CLKB_;
  wire  CENB_;
  reg  CENB_int;
  reg  CENB_p2;
  wire [127:0] WENB_;
  reg [127:0] WENB_int;
  wire [9:0] AB_;
  reg [9:0] AB_int;
  wire [127:0] DB_;
  reg [127:0] DB_int;
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
  wire [127:0] TWENA_;
  reg [127:0] TWENA_int;
  wire [9:0] TAA_;
  reg [9:0] TAA_int;
  wire [127:0] TDA_;
  reg [127:0] TDA_int;
  wire  TENB_;
  reg  TENB_int;
  wire  TCENB_;
  reg  TCENB_int;
  reg  TCENB_p2;
  wire [127:0] TWENB_;
  reg [127:0] TWENB_int;
  wire [9:0] TAB_;
  reg [9:0] TAB_int;
  wire [127:0] TDB_;
  reg [127:0] TDB_int;
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
  assign WENYA[32] = WENYA_[32]; 
  assign WENYA[33] = WENYA_[33]; 
  assign WENYA[34] = WENYA_[34]; 
  assign WENYA[35] = WENYA_[35]; 
  assign WENYA[36] = WENYA_[36]; 
  assign WENYA[37] = WENYA_[37]; 
  assign WENYA[38] = WENYA_[38]; 
  assign WENYA[39] = WENYA_[39]; 
  assign WENYA[40] = WENYA_[40]; 
  assign WENYA[41] = WENYA_[41]; 
  assign WENYA[42] = WENYA_[42]; 
  assign WENYA[43] = WENYA_[43]; 
  assign WENYA[44] = WENYA_[44]; 
  assign WENYA[45] = WENYA_[45]; 
  assign WENYA[46] = WENYA_[46]; 
  assign WENYA[47] = WENYA_[47]; 
  assign WENYA[48] = WENYA_[48]; 
  assign WENYA[49] = WENYA_[49]; 
  assign WENYA[50] = WENYA_[50]; 
  assign WENYA[51] = WENYA_[51]; 
  assign WENYA[52] = WENYA_[52]; 
  assign WENYA[53] = WENYA_[53]; 
  assign WENYA[54] = WENYA_[54]; 
  assign WENYA[55] = WENYA_[55]; 
  assign WENYA[56] = WENYA_[56]; 
  assign WENYA[57] = WENYA_[57]; 
  assign WENYA[58] = WENYA_[58]; 
  assign WENYA[59] = WENYA_[59]; 
  assign WENYA[60] = WENYA_[60]; 
  assign WENYA[61] = WENYA_[61]; 
  assign WENYA[62] = WENYA_[62]; 
  assign WENYA[63] = WENYA_[63]; 
  assign WENYA[64] = WENYA_[64]; 
  assign WENYA[65] = WENYA_[65]; 
  assign WENYA[66] = WENYA_[66]; 
  assign WENYA[67] = WENYA_[67]; 
  assign WENYA[68] = WENYA_[68]; 
  assign WENYA[69] = WENYA_[69]; 
  assign WENYA[70] = WENYA_[70]; 
  assign WENYA[71] = WENYA_[71]; 
  assign WENYA[72] = WENYA_[72]; 
  assign WENYA[73] = WENYA_[73]; 
  assign WENYA[74] = WENYA_[74]; 
  assign WENYA[75] = WENYA_[75]; 
  assign WENYA[76] = WENYA_[76]; 
  assign WENYA[77] = WENYA_[77]; 
  assign WENYA[78] = WENYA_[78]; 
  assign WENYA[79] = WENYA_[79]; 
  assign WENYA[80] = WENYA_[80]; 
  assign WENYA[81] = WENYA_[81]; 
  assign WENYA[82] = WENYA_[82]; 
  assign WENYA[83] = WENYA_[83]; 
  assign WENYA[84] = WENYA_[84]; 
  assign WENYA[85] = WENYA_[85]; 
  assign WENYA[86] = WENYA_[86]; 
  assign WENYA[87] = WENYA_[87]; 
  assign WENYA[88] = WENYA_[88]; 
  assign WENYA[89] = WENYA_[89]; 
  assign WENYA[90] = WENYA_[90]; 
  assign WENYA[91] = WENYA_[91]; 
  assign WENYA[92] = WENYA_[92]; 
  assign WENYA[93] = WENYA_[93]; 
  assign WENYA[94] = WENYA_[94]; 
  assign WENYA[95] = WENYA_[95]; 
  assign WENYA[96] = WENYA_[96]; 
  assign WENYA[97] = WENYA_[97]; 
  assign WENYA[98] = WENYA_[98]; 
  assign WENYA[99] = WENYA_[99]; 
  assign WENYA[100] = WENYA_[100]; 
  assign WENYA[101] = WENYA_[101]; 
  assign WENYA[102] = WENYA_[102]; 
  assign WENYA[103] = WENYA_[103]; 
  assign WENYA[104] = WENYA_[104]; 
  assign WENYA[105] = WENYA_[105]; 
  assign WENYA[106] = WENYA_[106]; 
  assign WENYA[107] = WENYA_[107]; 
  assign WENYA[108] = WENYA_[108]; 
  assign WENYA[109] = WENYA_[109]; 
  assign WENYA[110] = WENYA_[110]; 
  assign WENYA[111] = WENYA_[111]; 
  assign WENYA[112] = WENYA_[112]; 
  assign WENYA[113] = WENYA_[113]; 
  assign WENYA[114] = WENYA_[114]; 
  assign WENYA[115] = WENYA_[115]; 
  assign WENYA[116] = WENYA_[116]; 
  assign WENYA[117] = WENYA_[117]; 
  assign WENYA[118] = WENYA_[118]; 
  assign WENYA[119] = WENYA_[119]; 
  assign WENYA[120] = WENYA_[120]; 
  assign WENYA[121] = WENYA_[121]; 
  assign WENYA[122] = WENYA_[122]; 
  assign WENYA[123] = WENYA_[123]; 
  assign WENYA[124] = WENYA_[124]; 
  assign WENYA[125] = WENYA_[125]; 
  assign WENYA[126] = WENYA_[126]; 
  assign WENYA[127] = WENYA_[127]; 
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
  assign WENYB[32] = WENYB_[32]; 
  assign WENYB[33] = WENYB_[33]; 
  assign WENYB[34] = WENYB_[34]; 
  assign WENYB[35] = WENYB_[35]; 
  assign WENYB[36] = WENYB_[36]; 
  assign WENYB[37] = WENYB_[37]; 
  assign WENYB[38] = WENYB_[38]; 
  assign WENYB[39] = WENYB_[39]; 
  assign WENYB[40] = WENYB_[40]; 
  assign WENYB[41] = WENYB_[41]; 
  assign WENYB[42] = WENYB_[42]; 
  assign WENYB[43] = WENYB_[43]; 
  assign WENYB[44] = WENYB_[44]; 
  assign WENYB[45] = WENYB_[45]; 
  assign WENYB[46] = WENYB_[46]; 
  assign WENYB[47] = WENYB_[47]; 
  assign WENYB[48] = WENYB_[48]; 
  assign WENYB[49] = WENYB_[49]; 
  assign WENYB[50] = WENYB_[50]; 
  assign WENYB[51] = WENYB_[51]; 
  assign WENYB[52] = WENYB_[52]; 
  assign WENYB[53] = WENYB_[53]; 
  assign WENYB[54] = WENYB_[54]; 
  assign WENYB[55] = WENYB_[55]; 
  assign WENYB[56] = WENYB_[56]; 
  assign WENYB[57] = WENYB_[57]; 
  assign WENYB[58] = WENYB_[58]; 
  assign WENYB[59] = WENYB_[59]; 
  assign WENYB[60] = WENYB_[60]; 
  assign WENYB[61] = WENYB_[61]; 
  assign WENYB[62] = WENYB_[62]; 
  assign WENYB[63] = WENYB_[63]; 
  assign WENYB[64] = WENYB_[64]; 
  assign WENYB[65] = WENYB_[65]; 
  assign WENYB[66] = WENYB_[66]; 
  assign WENYB[67] = WENYB_[67]; 
  assign WENYB[68] = WENYB_[68]; 
  assign WENYB[69] = WENYB_[69]; 
  assign WENYB[70] = WENYB_[70]; 
  assign WENYB[71] = WENYB_[71]; 
  assign WENYB[72] = WENYB_[72]; 
  assign WENYB[73] = WENYB_[73]; 
  assign WENYB[74] = WENYB_[74]; 
  assign WENYB[75] = WENYB_[75]; 
  assign WENYB[76] = WENYB_[76]; 
  assign WENYB[77] = WENYB_[77]; 
  assign WENYB[78] = WENYB_[78]; 
  assign WENYB[79] = WENYB_[79]; 
  assign WENYB[80] = WENYB_[80]; 
  assign WENYB[81] = WENYB_[81]; 
  assign WENYB[82] = WENYB_[82]; 
  assign WENYB[83] = WENYB_[83]; 
  assign WENYB[84] = WENYB_[84]; 
  assign WENYB[85] = WENYB_[85]; 
  assign WENYB[86] = WENYB_[86]; 
  assign WENYB[87] = WENYB_[87]; 
  assign WENYB[88] = WENYB_[88]; 
  assign WENYB[89] = WENYB_[89]; 
  assign WENYB[90] = WENYB_[90]; 
  assign WENYB[91] = WENYB_[91]; 
  assign WENYB[92] = WENYB_[92]; 
  assign WENYB[93] = WENYB_[93]; 
  assign WENYB[94] = WENYB_[94]; 
  assign WENYB[95] = WENYB_[95]; 
  assign WENYB[96] = WENYB_[96]; 
  assign WENYB[97] = WENYB_[97]; 
  assign WENYB[98] = WENYB_[98]; 
  assign WENYB[99] = WENYB_[99]; 
  assign WENYB[100] = WENYB_[100]; 
  assign WENYB[101] = WENYB_[101]; 
  assign WENYB[102] = WENYB_[102]; 
  assign WENYB[103] = WENYB_[103]; 
  assign WENYB[104] = WENYB_[104]; 
  assign WENYB[105] = WENYB_[105]; 
  assign WENYB[106] = WENYB_[106]; 
  assign WENYB[107] = WENYB_[107]; 
  assign WENYB[108] = WENYB_[108]; 
  assign WENYB[109] = WENYB_[109]; 
  assign WENYB[110] = WENYB_[110]; 
  assign WENYB[111] = WENYB_[111]; 
  assign WENYB[112] = WENYB_[112]; 
  assign WENYB[113] = WENYB_[113]; 
  assign WENYB[114] = WENYB_[114]; 
  assign WENYB[115] = WENYB_[115]; 
  assign WENYB[116] = WENYB_[116]; 
  assign WENYB[117] = WENYB_[117]; 
  assign WENYB[118] = WENYB_[118]; 
  assign WENYB[119] = WENYB_[119]; 
  assign WENYB[120] = WENYB_[120]; 
  assign WENYB[121] = WENYB_[121]; 
  assign WENYB[122] = WENYB_[122]; 
  assign WENYB[123] = WENYB_[123]; 
  assign WENYB[124] = WENYB_[124]; 
  assign WENYB[125] = WENYB_[125]; 
  assign WENYB[126] = WENYB_[126]; 
  assign WENYB[127] = WENYB_[127]; 
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
  assign QA[32] = QA_[32]; 
  assign QA[33] = QA_[33]; 
  assign QA[34] = QA_[34]; 
  assign QA[35] = QA_[35]; 
  assign QA[36] = QA_[36]; 
  assign QA[37] = QA_[37]; 
  assign QA[38] = QA_[38]; 
  assign QA[39] = QA_[39]; 
  assign QA[40] = QA_[40]; 
  assign QA[41] = QA_[41]; 
  assign QA[42] = QA_[42]; 
  assign QA[43] = QA_[43]; 
  assign QA[44] = QA_[44]; 
  assign QA[45] = QA_[45]; 
  assign QA[46] = QA_[46]; 
  assign QA[47] = QA_[47]; 
  assign QA[48] = QA_[48]; 
  assign QA[49] = QA_[49]; 
  assign QA[50] = QA_[50]; 
  assign QA[51] = QA_[51]; 
  assign QA[52] = QA_[52]; 
  assign QA[53] = QA_[53]; 
  assign QA[54] = QA_[54]; 
  assign QA[55] = QA_[55]; 
  assign QA[56] = QA_[56]; 
  assign QA[57] = QA_[57]; 
  assign QA[58] = QA_[58]; 
  assign QA[59] = QA_[59]; 
  assign QA[60] = QA_[60]; 
  assign QA[61] = QA_[61]; 
  assign QA[62] = QA_[62]; 
  assign QA[63] = QA_[63]; 
  assign QA[64] = QA_[64]; 
  assign QA[65] = QA_[65]; 
  assign QA[66] = QA_[66]; 
  assign QA[67] = QA_[67]; 
  assign QA[68] = QA_[68]; 
  assign QA[69] = QA_[69]; 
  assign QA[70] = QA_[70]; 
  assign QA[71] = QA_[71]; 
  assign QA[72] = QA_[72]; 
  assign QA[73] = QA_[73]; 
  assign QA[74] = QA_[74]; 
  assign QA[75] = QA_[75]; 
  assign QA[76] = QA_[76]; 
  assign QA[77] = QA_[77]; 
  assign QA[78] = QA_[78]; 
  assign QA[79] = QA_[79]; 
  assign QA[80] = QA_[80]; 
  assign QA[81] = QA_[81]; 
  assign QA[82] = QA_[82]; 
  assign QA[83] = QA_[83]; 
  assign QA[84] = QA_[84]; 
  assign QA[85] = QA_[85]; 
  assign QA[86] = QA_[86]; 
  assign QA[87] = QA_[87]; 
  assign QA[88] = QA_[88]; 
  assign QA[89] = QA_[89]; 
  assign QA[90] = QA_[90]; 
  assign QA[91] = QA_[91]; 
  assign QA[92] = QA_[92]; 
  assign QA[93] = QA_[93]; 
  assign QA[94] = QA_[94]; 
  assign QA[95] = QA_[95]; 
  assign QA[96] = QA_[96]; 
  assign QA[97] = QA_[97]; 
  assign QA[98] = QA_[98]; 
  assign QA[99] = QA_[99]; 
  assign QA[100] = QA_[100]; 
  assign QA[101] = QA_[101]; 
  assign QA[102] = QA_[102]; 
  assign QA[103] = QA_[103]; 
  assign QA[104] = QA_[104]; 
  assign QA[105] = QA_[105]; 
  assign QA[106] = QA_[106]; 
  assign QA[107] = QA_[107]; 
  assign QA[108] = QA_[108]; 
  assign QA[109] = QA_[109]; 
  assign QA[110] = QA_[110]; 
  assign QA[111] = QA_[111]; 
  assign QA[112] = QA_[112]; 
  assign QA[113] = QA_[113]; 
  assign QA[114] = QA_[114]; 
  assign QA[115] = QA_[115]; 
  assign QA[116] = QA_[116]; 
  assign QA[117] = QA_[117]; 
  assign QA[118] = QA_[118]; 
  assign QA[119] = QA_[119]; 
  assign QA[120] = QA_[120]; 
  assign QA[121] = QA_[121]; 
  assign QA[122] = QA_[122]; 
  assign QA[123] = QA_[123]; 
  assign QA[124] = QA_[124]; 
  assign QA[125] = QA_[125]; 
  assign QA[126] = QA_[126]; 
  assign QA[127] = QA_[127]; 
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
  assign QB[32] = QB_[32]; 
  assign QB[33] = QB_[33]; 
  assign QB[34] = QB_[34]; 
  assign QB[35] = QB_[35]; 
  assign QB[36] = QB_[36]; 
  assign QB[37] = QB_[37]; 
  assign QB[38] = QB_[38]; 
  assign QB[39] = QB_[39]; 
  assign QB[40] = QB_[40]; 
  assign QB[41] = QB_[41]; 
  assign QB[42] = QB_[42]; 
  assign QB[43] = QB_[43]; 
  assign QB[44] = QB_[44]; 
  assign QB[45] = QB_[45]; 
  assign QB[46] = QB_[46]; 
  assign QB[47] = QB_[47]; 
  assign QB[48] = QB_[48]; 
  assign QB[49] = QB_[49]; 
  assign QB[50] = QB_[50]; 
  assign QB[51] = QB_[51]; 
  assign QB[52] = QB_[52]; 
  assign QB[53] = QB_[53]; 
  assign QB[54] = QB_[54]; 
  assign QB[55] = QB_[55]; 
  assign QB[56] = QB_[56]; 
  assign QB[57] = QB_[57]; 
  assign QB[58] = QB_[58]; 
  assign QB[59] = QB_[59]; 
  assign QB[60] = QB_[60]; 
  assign QB[61] = QB_[61]; 
  assign QB[62] = QB_[62]; 
  assign QB[63] = QB_[63]; 
  assign QB[64] = QB_[64]; 
  assign QB[65] = QB_[65]; 
  assign QB[66] = QB_[66]; 
  assign QB[67] = QB_[67]; 
  assign QB[68] = QB_[68]; 
  assign QB[69] = QB_[69]; 
  assign QB[70] = QB_[70]; 
  assign QB[71] = QB_[71]; 
  assign QB[72] = QB_[72]; 
  assign QB[73] = QB_[73]; 
  assign QB[74] = QB_[74]; 
  assign QB[75] = QB_[75]; 
  assign QB[76] = QB_[76]; 
  assign QB[77] = QB_[77]; 
  assign QB[78] = QB_[78]; 
  assign QB[79] = QB_[79]; 
  assign QB[80] = QB_[80]; 
  assign QB[81] = QB_[81]; 
  assign QB[82] = QB_[82]; 
  assign QB[83] = QB_[83]; 
  assign QB[84] = QB_[84]; 
  assign QB[85] = QB_[85]; 
  assign QB[86] = QB_[86]; 
  assign QB[87] = QB_[87]; 
  assign QB[88] = QB_[88]; 
  assign QB[89] = QB_[89]; 
  assign QB[90] = QB_[90]; 
  assign QB[91] = QB_[91]; 
  assign QB[92] = QB_[92]; 
  assign QB[93] = QB_[93]; 
  assign QB[94] = QB_[94]; 
  assign QB[95] = QB_[95]; 
  assign QB[96] = QB_[96]; 
  assign QB[97] = QB_[97]; 
  assign QB[98] = QB_[98]; 
  assign QB[99] = QB_[99]; 
  assign QB[100] = QB_[100]; 
  assign QB[101] = QB_[101]; 
  assign QB[102] = QB_[102]; 
  assign QB[103] = QB_[103]; 
  assign QB[104] = QB_[104]; 
  assign QB[105] = QB_[105]; 
  assign QB[106] = QB_[106]; 
  assign QB[107] = QB_[107]; 
  assign QB[108] = QB_[108]; 
  assign QB[109] = QB_[109]; 
  assign QB[110] = QB_[110]; 
  assign QB[111] = QB_[111]; 
  assign QB[112] = QB_[112]; 
  assign QB[113] = QB_[113]; 
  assign QB[114] = QB_[114]; 
  assign QB[115] = QB_[115]; 
  assign QB[116] = QB_[116]; 
  assign QB[117] = QB_[117]; 
  assign QB[118] = QB_[118]; 
  assign QB[119] = QB_[119]; 
  assign QB[120] = QB_[120]; 
  assign QB[121] = QB_[121]; 
  assign QB[122] = QB_[122]; 
  assign QB[123] = QB_[123]; 
  assign QB[124] = QB_[124]; 
  assign QB[125] = QB_[125]; 
  assign QB[126] = QB_[126]; 
  assign QB[127] = QB_[127]; 
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
  assign WENA_[32] = WENA[32];
  assign WENA_[33] = WENA[33];
  assign WENA_[34] = WENA[34];
  assign WENA_[35] = WENA[35];
  assign WENA_[36] = WENA[36];
  assign WENA_[37] = WENA[37];
  assign WENA_[38] = WENA[38];
  assign WENA_[39] = WENA[39];
  assign WENA_[40] = WENA[40];
  assign WENA_[41] = WENA[41];
  assign WENA_[42] = WENA[42];
  assign WENA_[43] = WENA[43];
  assign WENA_[44] = WENA[44];
  assign WENA_[45] = WENA[45];
  assign WENA_[46] = WENA[46];
  assign WENA_[47] = WENA[47];
  assign WENA_[48] = WENA[48];
  assign WENA_[49] = WENA[49];
  assign WENA_[50] = WENA[50];
  assign WENA_[51] = WENA[51];
  assign WENA_[52] = WENA[52];
  assign WENA_[53] = WENA[53];
  assign WENA_[54] = WENA[54];
  assign WENA_[55] = WENA[55];
  assign WENA_[56] = WENA[56];
  assign WENA_[57] = WENA[57];
  assign WENA_[58] = WENA[58];
  assign WENA_[59] = WENA[59];
  assign WENA_[60] = WENA[60];
  assign WENA_[61] = WENA[61];
  assign WENA_[62] = WENA[62];
  assign WENA_[63] = WENA[63];
  assign WENA_[64] = WENA[64];
  assign WENA_[65] = WENA[65];
  assign WENA_[66] = WENA[66];
  assign WENA_[67] = WENA[67];
  assign WENA_[68] = WENA[68];
  assign WENA_[69] = WENA[69];
  assign WENA_[70] = WENA[70];
  assign WENA_[71] = WENA[71];
  assign WENA_[72] = WENA[72];
  assign WENA_[73] = WENA[73];
  assign WENA_[74] = WENA[74];
  assign WENA_[75] = WENA[75];
  assign WENA_[76] = WENA[76];
  assign WENA_[77] = WENA[77];
  assign WENA_[78] = WENA[78];
  assign WENA_[79] = WENA[79];
  assign WENA_[80] = WENA[80];
  assign WENA_[81] = WENA[81];
  assign WENA_[82] = WENA[82];
  assign WENA_[83] = WENA[83];
  assign WENA_[84] = WENA[84];
  assign WENA_[85] = WENA[85];
  assign WENA_[86] = WENA[86];
  assign WENA_[87] = WENA[87];
  assign WENA_[88] = WENA[88];
  assign WENA_[89] = WENA[89];
  assign WENA_[90] = WENA[90];
  assign WENA_[91] = WENA[91];
  assign WENA_[92] = WENA[92];
  assign WENA_[93] = WENA[93];
  assign WENA_[94] = WENA[94];
  assign WENA_[95] = WENA[95];
  assign WENA_[96] = WENA[96];
  assign WENA_[97] = WENA[97];
  assign WENA_[98] = WENA[98];
  assign WENA_[99] = WENA[99];
  assign WENA_[100] = WENA[100];
  assign WENA_[101] = WENA[101];
  assign WENA_[102] = WENA[102];
  assign WENA_[103] = WENA[103];
  assign WENA_[104] = WENA[104];
  assign WENA_[105] = WENA[105];
  assign WENA_[106] = WENA[106];
  assign WENA_[107] = WENA[107];
  assign WENA_[108] = WENA[108];
  assign WENA_[109] = WENA[109];
  assign WENA_[110] = WENA[110];
  assign WENA_[111] = WENA[111];
  assign WENA_[112] = WENA[112];
  assign WENA_[113] = WENA[113];
  assign WENA_[114] = WENA[114];
  assign WENA_[115] = WENA[115];
  assign WENA_[116] = WENA[116];
  assign WENA_[117] = WENA[117];
  assign WENA_[118] = WENA[118];
  assign WENA_[119] = WENA[119];
  assign WENA_[120] = WENA[120];
  assign WENA_[121] = WENA[121];
  assign WENA_[122] = WENA[122];
  assign WENA_[123] = WENA[123];
  assign WENA_[124] = WENA[124];
  assign WENA_[125] = WENA[125];
  assign WENA_[126] = WENA[126];
  assign WENA_[127] = WENA[127];
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
  assign DA_[32] = DA[32];
  assign DA_[33] = DA[33];
  assign DA_[34] = DA[34];
  assign DA_[35] = DA[35];
  assign DA_[36] = DA[36];
  assign DA_[37] = DA[37];
  assign DA_[38] = DA[38];
  assign DA_[39] = DA[39];
  assign DA_[40] = DA[40];
  assign DA_[41] = DA[41];
  assign DA_[42] = DA[42];
  assign DA_[43] = DA[43];
  assign DA_[44] = DA[44];
  assign DA_[45] = DA[45];
  assign DA_[46] = DA[46];
  assign DA_[47] = DA[47];
  assign DA_[48] = DA[48];
  assign DA_[49] = DA[49];
  assign DA_[50] = DA[50];
  assign DA_[51] = DA[51];
  assign DA_[52] = DA[52];
  assign DA_[53] = DA[53];
  assign DA_[54] = DA[54];
  assign DA_[55] = DA[55];
  assign DA_[56] = DA[56];
  assign DA_[57] = DA[57];
  assign DA_[58] = DA[58];
  assign DA_[59] = DA[59];
  assign DA_[60] = DA[60];
  assign DA_[61] = DA[61];
  assign DA_[62] = DA[62];
  assign DA_[63] = DA[63];
  assign DA_[64] = DA[64];
  assign DA_[65] = DA[65];
  assign DA_[66] = DA[66];
  assign DA_[67] = DA[67];
  assign DA_[68] = DA[68];
  assign DA_[69] = DA[69];
  assign DA_[70] = DA[70];
  assign DA_[71] = DA[71];
  assign DA_[72] = DA[72];
  assign DA_[73] = DA[73];
  assign DA_[74] = DA[74];
  assign DA_[75] = DA[75];
  assign DA_[76] = DA[76];
  assign DA_[77] = DA[77];
  assign DA_[78] = DA[78];
  assign DA_[79] = DA[79];
  assign DA_[80] = DA[80];
  assign DA_[81] = DA[81];
  assign DA_[82] = DA[82];
  assign DA_[83] = DA[83];
  assign DA_[84] = DA[84];
  assign DA_[85] = DA[85];
  assign DA_[86] = DA[86];
  assign DA_[87] = DA[87];
  assign DA_[88] = DA[88];
  assign DA_[89] = DA[89];
  assign DA_[90] = DA[90];
  assign DA_[91] = DA[91];
  assign DA_[92] = DA[92];
  assign DA_[93] = DA[93];
  assign DA_[94] = DA[94];
  assign DA_[95] = DA[95];
  assign DA_[96] = DA[96];
  assign DA_[97] = DA[97];
  assign DA_[98] = DA[98];
  assign DA_[99] = DA[99];
  assign DA_[100] = DA[100];
  assign DA_[101] = DA[101];
  assign DA_[102] = DA[102];
  assign DA_[103] = DA[103];
  assign DA_[104] = DA[104];
  assign DA_[105] = DA[105];
  assign DA_[106] = DA[106];
  assign DA_[107] = DA[107];
  assign DA_[108] = DA[108];
  assign DA_[109] = DA[109];
  assign DA_[110] = DA[110];
  assign DA_[111] = DA[111];
  assign DA_[112] = DA[112];
  assign DA_[113] = DA[113];
  assign DA_[114] = DA[114];
  assign DA_[115] = DA[115];
  assign DA_[116] = DA[116];
  assign DA_[117] = DA[117];
  assign DA_[118] = DA[118];
  assign DA_[119] = DA[119];
  assign DA_[120] = DA[120];
  assign DA_[121] = DA[121];
  assign DA_[122] = DA[122];
  assign DA_[123] = DA[123];
  assign DA_[124] = DA[124];
  assign DA_[125] = DA[125];
  assign DA_[126] = DA[126];
  assign DA_[127] = DA[127];
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
  assign WENB_[32] = WENB[32];
  assign WENB_[33] = WENB[33];
  assign WENB_[34] = WENB[34];
  assign WENB_[35] = WENB[35];
  assign WENB_[36] = WENB[36];
  assign WENB_[37] = WENB[37];
  assign WENB_[38] = WENB[38];
  assign WENB_[39] = WENB[39];
  assign WENB_[40] = WENB[40];
  assign WENB_[41] = WENB[41];
  assign WENB_[42] = WENB[42];
  assign WENB_[43] = WENB[43];
  assign WENB_[44] = WENB[44];
  assign WENB_[45] = WENB[45];
  assign WENB_[46] = WENB[46];
  assign WENB_[47] = WENB[47];
  assign WENB_[48] = WENB[48];
  assign WENB_[49] = WENB[49];
  assign WENB_[50] = WENB[50];
  assign WENB_[51] = WENB[51];
  assign WENB_[52] = WENB[52];
  assign WENB_[53] = WENB[53];
  assign WENB_[54] = WENB[54];
  assign WENB_[55] = WENB[55];
  assign WENB_[56] = WENB[56];
  assign WENB_[57] = WENB[57];
  assign WENB_[58] = WENB[58];
  assign WENB_[59] = WENB[59];
  assign WENB_[60] = WENB[60];
  assign WENB_[61] = WENB[61];
  assign WENB_[62] = WENB[62];
  assign WENB_[63] = WENB[63];
  assign WENB_[64] = WENB[64];
  assign WENB_[65] = WENB[65];
  assign WENB_[66] = WENB[66];
  assign WENB_[67] = WENB[67];
  assign WENB_[68] = WENB[68];
  assign WENB_[69] = WENB[69];
  assign WENB_[70] = WENB[70];
  assign WENB_[71] = WENB[71];
  assign WENB_[72] = WENB[72];
  assign WENB_[73] = WENB[73];
  assign WENB_[74] = WENB[74];
  assign WENB_[75] = WENB[75];
  assign WENB_[76] = WENB[76];
  assign WENB_[77] = WENB[77];
  assign WENB_[78] = WENB[78];
  assign WENB_[79] = WENB[79];
  assign WENB_[80] = WENB[80];
  assign WENB_[81] = WENB[81];
  assign WENB_[82] = WENB[82];
  assign WENB_[83] = WENB[83];
  assign WENB_[84] = WENB[84];
  assign WENB_[85] = WENB[85];
  assign WENB_[86] = WENB[86];
  assign WENB_[87] = WENB[87];
  assign WENB_[88] = WENB[88];
  assign WENB_[89] = WENB[89];
  assign WENB_[90] = WENB[90];
  assign WENB_[91] = WENB[91];
  assign WENB_[92] = WENB[92];
  assign WENB_[93] = WENB[93];
  assign WENB_[94] = WENB[94];
  assign WENB_[95] = WENB[95];
  assign WENB_[96] = WENB[96];
  assign WENB_[97] = WENB[97];
  assign WENB_[98] = WENB[98];
  assign WENB_[99] = WENB[99];
  assign WENB_[100] = WENB[100];
  assign WENB_[101] = WENB[101];
  assign WENB_[102] = WENB[102];
  assign WENB_[103] = WENB[103];
  assign WENB_[104] = WENB[104];
  assign WENB_[105] = WENB[105];
  assign WENB_[106] = WENB[106];
  assign WENB_[107] = WENB[107];
  assign WENB_[108] = WENB[108];
  assign WENB_[109] = WENB[109];
  assign WENB_[110] = WENB[110];
  assign WENB_[111] = WENB[111];
  assign WENB_[112] = WENB[112];
  assign WENB_[113] = WENB[113];
  assign WENB_[114] = WENB[114];
  assign WENB_[115] = WENB[115];
  assign WENB_[116] = WENB[116];
  assign WENB_[117] = WENB[117];
  assign WENB_[118] = WENB[118];
  assign WENB_[119] = WENB[119];
  assign WENB_[120] = WENB[120];
  assign WENB_[121] = WENB[121];
  assign WENB_[122] = WENB[122];
  assign WENB_[123] = WENB[123];
  assign WENB_[124] = WENB[124];
  assign WENB_[125] = WENB[125];
  assign WENB_[126] = WENB[126];
  assign WENB_[127] = WENB[127];
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
  assign DB_[32] = DB[32];
  assign DB_[33] = DB[33];
  assign DB_[34] = DB[34];
  assign DB_[35] = DB[35];
  assign DB_[36] = DB[36];
  assign DB_[37] = DB[37];
  assign DB_[38] = DB[38];
  assign DB_[39] = DB[39];
  assign DB_[40] = DB[40];
  assign DB_[41] = DB[41];
  assign DB_[42] = DB[42];
  assign DB_[43] = DB[43];
  assign DB_[44] = DB[44];
  assign DB_[45] = DB[45];
  assign DB_[46] = DB[46];
  assign DB_[47] = DB[47];
  assign DB_[48] = DB[48];
  assign DB_[49] = DB[49];
  assign DB_[50] = DB[50];
  assign DB_[51] = DB[51];
  assign DB_[52] = DB[52];
  assign DB_[53] = DB[53];
  assign DB_[54] = DB[54];
  assign DB_[55] = DB[55];
  assign DB_[56] = DB[56];
  assign DB_[57] = DB[57];
  assign DB_[58] = DB[58];
  assign DB_[59] = DB[59];
  assign DB_[60] = DB[60];
  assign DB_[61] = DB[61];
  assign DB_[62] = DB[62];
  assign DB_[63] = DB[63];
  assign DB_[64] = DB[64];
  assign DB_[65] = DB[65];
  assign DB_[66] = DB[66];
  assign DB_[67] = DB[67];
  assign DB_[68] = DB[68];
  assign DB_[69] = DB[69];
  assign DB_[70] = DB[70];
  assign DB_[71] = DB[71];
  assign DB_[72] = DB[72];
  assign DB_[73] = DB[73];
  assign DB_[74] = DB[74];
  assign DB_[75] = DB[75];
  assign DB_[76] = DB[76];
  assign DB_[77] = DB[77];
  assign DB_[78] = DB[78];
  assign DB_[79] = DB[79];
  assign DB_[80] = DB[80];
  assign DB_[81] = DB[81];
  assign DB_[82] = DB[82];
  assign DB_[83] = DB[83];
  assign DB_[84] = DB[84];
  assign DB_[85] = DB[85];
  assign DB_[86] = DB[86];
  assign DB_[87] = DB[87];
  assign DB_[88] = DB[88];
  assign DB_[89] = DB[89];
  assign DB_[90] = DB[90];
  assign DB_[91] = DB[91];
  assign DB_[92] = DB[92];
  assign DB_[93] = DB[93];
  assign DB_[94] = DB[94];
  assign DB_[95] = DB[95];
  assign DB_[96] = DB[96];
  assign DB_[97] = DB[97];
  assign DB_[98] = DB[98];
  assign DB_[99] = DB[99];
  assign DB_[100] = DB[100];
  assign DB_[101] = DB[101];
  assign DB_[102] = DB[102];
  assign DB_[103] = DB[103];
  assign DB_[104] = DB[104];
  assign DB_[105] = DB[105];
  assign DB_[106] = DB[106];
  assign DB_[107] = DB[107];
  assign DB_[108] = DB[108];
  assign DB_[109] = DB[109];
  assign DB_[110] = DB[110];
  assign DB_[111] = DB[111];
  assign DB_[112] = DB[112];
  assign DB_[113] = DB[113];
  assign DB_[114] = DB[114];
  assign DB_[115] = DB[115];
  assign DB_[116] = DB[116];
  assign DB_[117] = DB[117];
  assign DB_[118] = DB[118];
  assign DB_[119] = DB[119];
  assign DB_[120] = DB[120];
  assign DB_[121] = DB[121];
  assign DB_[122] = DB[122];
  assign DB_[123] = DB[123];
  assign DB_[124] = DB[124];
  assign DB_[125] = DB[125];
  assign DB_[126] = DB[126];
  assign DB_[127] = DB[127];
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
  assign TWENA_[32] = TWENA[32];
  assign TWENA_[33] = TWENA[33];
  assign TWENA_[34] = TWENA[34];
  assign TWENA_[35] = TWENA[35];
  assign TWENA_[36] = TWENA[36];
  assign TWENA_[37] = TWENA[37];
  assign TWENA_[38] = TWENA[38];
  assign TWENA_[39] = TWENA[39];
  assign TWENA_[40] = TWENA[40];
  assign TWENA_[41] = TWENA[41];
  assign TWENA_[42] = TWENA[42];
  assign TWENA_[43] = TWENA[43];
  assign TWENA_[44] = TWENA[44];
  assign TWENA_[45] = TWENA[45];
  assign TWENA_[46] = TWENA[46];
  assign TWENA_[47] = TWENA[47];
  assign TWENA_[48] = TWENA[48];
  assign TWENA_[49] = TWENA[49];
  assign TWENA_[50] = TWENA[50];
  assign TWENA_[51] = TWENA[51];
  assign TWENA_[52] = TWENA[52];
  assign TWENA_[53] = TWENA[53];
  assign TWENA_[54] = TWENA[54];
  assign TWENA_[55] = TWENA[55];
  assign TWENA_[56] = TWENA[56];
  assign TWENA_[57] = TWENA[57];
  assign TWENA_[58] = TWENA[58];
  assign TWENA_[59] = TWENA[59];
  assign TWENA_[60] = TWENA[60];
  assign TWENA_[61] = TWENA[61];
  assign TWENA_[62] = TWENA[62];
  assign TWENA_[63] = TWENA[63];
  assign TWENA_[64] = TWENA[64];
  assign TWENA_[65] = TWENA[65];
  assign TWENA_[66] = TWENA[66];
  assign TWENA_[67] = TWENA[67];
  assign TWENA_[68] = TWENA[68];
  assign TWENA_[69] = TWENA[69];
  assign TWENA_[70] = TWENA[70];
  assign TWENA_[71] = TWENA[71];
  assign TWENA_[72] = TWENA[72];
  assign TWENA_[73] = TWENA[73];
  assign TWENA_[74] = TWENA[74];
  assign TWENA_[75] = TWENA[75];
  assign TWENA_[76] = TWENA[76];
  assign TWENA_[77] = TWENA[77];
  assign TWENA_[78] = TWENA[78];
  assign TWENA_[79] = TWENA[79];
  assign TWENA_[80] = TWENA[80];
  assign TWENA_[81] = TWENA[81];
  assign TWENA_[82] = TWENA[82];
  assign TWENA_[83] = TWENA[83];
  assign TWENA_[84] = TWENA[84];
  assign TWENA_[85] = TWENA[85];
  assign TWENA_[86] = TWENA[86];
  assign TWENA_[87] = TWENA[87];
  assign TWENA_[88] = TWENA[88];
  assign TWENA_[89] = TWENA[89];
  assign TWENA_[90] = TWENA[90];
  assign TWENA_[91] = TWENA[91];
  assign TWENA_[92] = TWENA[92];
  assign TWENA_[93] = TWENA[93];
  assign TWENA_[94] = TWENA[94];
  assign TWENA_[95] = TWENA[95];
  assign TWENA_[96] = TWENA[96];
  assign TWENA_[97] = TWENA[97];
  assign TWENA_[98] = TWENA[98];
  assign TWENA_[99] = TWENA[99];
  assign TWENA_[100] = TWENA[100];
  assign TWENA_[101] = TWENA[101];
  assign TWENA_[102] = TWENA[102];
  assign TWENA_[103] = TWENA[103];
  assign TWENA_[104] = TWENA[104];
  assign TWENA_[105] = TWENA[105];
  assign TWENA_[106] = TWENA[106];
  assign TWENA_[107] = TWENA[107];
  assign TWENA_[108] = TWENA[108];
  assign TWENA_[109] = TWENA[109];
  assign TWENA_[110] = TWENA[110];
  assign TWENA_[111] = TWENA[111];
  assign TWENA_[112] = TWENA[112];
  assign TWENA_[113] = TWENA[113];
  assign TWENA_[114] = TWENA[114];
  assign TWENA_[115] = TWENA[115];
  assign TWENA_[116] = TWENA[116];
  assign TWENA_[117] = TWENA[117];
  assign TWENA_[118] = TWENA[118];
  assign TWENA_[119] = TWENA[119];
  assign TWENA_[120] = TWENA[120];
  assign TWENA_[121] = TWENA[121];
  assign TWENA_[122] = TWENA[122];
  assign TWENA_[123] = TWENA[123];
  assign TWENA_[124] = TWENA[124];
  assign TWENA_[125] = TWENA[125];
  assign TWENA_[126] = TWENA[126];
  assign TWENA_[127] = TWENA[127];
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
  assign TDA_[32] = TDA[32];
  assign TDA_[33] = TDA[33];
  assign TDA_[34] = TDA[34];
  assign TDA_[35] = TDA[35];
  assign TDA_[36] = TDA[36];
  assign TDA_[37] = TDA[37];
  assign TDA_[38] = TDA[38];
  assign TDA_[39] = TDA[39];
  assign TDA_[40] = TDA[40];
  assign TDA_[41] = TDA[41];
  assign TDA_[42] = TDA[42];
  assign TDA_[43] = TDA[43];
  assign TDA_[44] = TDA[44];
  assign TDA_[45] = TDA[45];
  assign TDA_[46] = TDA[46];
  assign TDA_[47] = TDA[47];
  assign TDA_[48] = TDA[48];
  assign TDA_[49] = TDA[49];
  assign TDA_[50] = TDA[50];
  assign TDA_[51] = TDA[51];
  assign TDA_[52] = TDA[52];
  assign TDA_[53] = TDA[53];
  assign TDA_[54] = TDA[54];
  assign TDA_[55] = TDA[55];
  assign TDA_[56] = TDA[56];
  assign TDA_[57] = TDA[57];
  assign TDA_[58] = TDA[58];
  assign TDA_[59] = TDA[59];
  assign TDA_[60] = TDA[60];
  assign TDA_[61] = TDA[61];
  assign TDA_[62] = TDA[62];
  assign TDA_[63] = TDA[63];
  assign TDA_[64] = TDA[64];
  assign TDA_[65] = TDA[65];
  assign TDA_[66] = TDA[66];
  assign TDA_[67] = TDA[67];
  assign TDA_[68] = TDA[68];
  assign TDA_[69] = TDA[69];
  assign TDA_[70] = TDA[70];
  assign TDA_[71] = TDA[71];
  assign TDA_[72] = TDA[72];
  assign TDA_[73] = TDA[73];
  assign TDA_[74] = TDA[74];
  assign TDA_[75] = TDA[75];
  assign TDA_[76] = TDA[76];
  assign TDA_[77] = TDA[77];
  assign TDA_[78] = TDA[78];
  assign TDA_[79] = TDA[79];
  assign TDA_[80] = TDA[80];
  assign TDA_[81] = TDA[81];
  assign TDA_[82] = TDA[82];
  assign TDA_[83] = TDA[83];
  assign TDA_[84] = TDA[84];
  assign TDA_[85] = TDA[85];
  assign TDA_[86] = TDA[86];
  assign TDA_[87] = TDA[87];
  assign TDA_[88] = TDA[88];
  assign TDA_[89] = TDA[89];
  assign TDA_[90] = TDA[90];
  assign TDA_[91] = TDA[91];
  assign TDA_[92] = TDA[92];
  assign TDA_[93] = TDA[93];
  assign TDA_[94] = TDA[94];
  assign TDA_[95] = TDA[95];
  assign TDA_[96] = TDA[96];
  assign TDA_[97] = TDA[97];
  assign TDA_[98] = TDA[98];
  assign TDA_[99] = TDA[99];
  assign TDA_[100] = TDA[100];
  assign TDA_[101] = TDA[101];
  assign TDA_[102] = TDA[102];
  assign TDA_[103] = TDA[103];
  assign TDA_[104] = TDA[104];
  assign TDA_[105] = TDA[105];
  assign TDA_[106] = TDA[106];
  assign TDA_[107] = TDA[107];
  assign TDA_[108] = TDA[108];
  assign TDA_[109] = TDA[109];
  assign TDA_[110] = TDA[110];
  assign TDA_[111] = TDA[111];
  assign TDA_[112] = TDA[112];
  assign TDA_[113] = TDA[113];
  assign TDA_[114] = TDA[114];
  assign TDA_[115] = TDA[115];
  assign TDA_[116] = TDA[116];
  assign TDA_[117] = TDA[117];
  assign TDA_[118] = TDA[118];
  assign TDA_[119] = TDA[119];
  assign TDA_[120] = TDA[120];
  assign TDA_[121] = TDA[121];
  assign TDA_[122] = TDA[122];
  assign TDA_[123] = TDA[123];
  assign TDA_[124] = TDA[124];
  assign TDA_[125] = TDA[125];
  assign TDA_[126] = TDA[126];
  assign TDA_[127] = TDA[127];
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
  assign TWENB_[32] = TWENB[32];
  assign TWENB_[33] = TWENB[33];
  assign TWENB_[34] = TWENB[34];
  assign TWENB_[35] = TWENB[35];
  assign TWENB_[36] = TWENB[36];
  assign TWENB_[37] = TWENB[37];
  assign TWENB_[38] = TWENB[38];
  assign TWENB_[39] = TWENB[39];
  assign TWENB_[40] = TWENB[40];
  assign TWENB_[41] = TWENB[41];
  assign TWENB_[42] = TWENB[42];
  assign TWENB_[43] = TWENB[43];
  assign TWENB_[44] = TWENB[44];
  assign TWENB_[45] = TWENB[45];
  assign TWENB_[46] = TWENB[46];
  assign TWENB_[47] = TWENB[47];
  assign TWENB_[48] = TWENB[48];
  assign TWENB_[49] = TWENB[49];
  assign TWENB_[50] = TWENB[50];
  assign TWENB_[51] = TWENB[51];
  assign TWENB_[52] = TWENB[52];
  assign TWENB_[53] = TWENB[53];
  assign TWENB_[54] = TWENB[54];
  assign TWENB_[55] = TWENB[55];
  assign TWENB_[56] = TWENB[56];
  assign TWENB_[57] = TWENB[57];
  assign TWENB_[58] = TWENB[58];
  assign TWENB_[59] = TWENB[59];
  assign TWENB_[60] = TWENB[60];
  assign TWENB_[61] = TWENB[61];
  assign TWENB_[62] = TWENB[62];
  assign TWENB_[63] = TWENB[63];
  assign TWENB_[64] = TWENB[64];
  assign TWENB_[65] = TWENB[65];
  assign TWENB_[66] = TWENB[66];
  assign TWENB_[67] = TWENB[67];
  assign TWENB_[68] = TWENB[68];
  assign TWENB_[69] = TWENB[69];
  assign TWENB_[70] = TWENB[70];
  assign TWENB_[71] = TWENB[71];
  assign TWENB_[72] = TWENB[72];
  assign TWENB_[73] = TWENB[73];
  assign TWENB_[74] = TWENB[74];
  assign TWENB_[75] = TWENB[75];
  assign TWENB_[76] = TWENB[76];
  assign TWENB_[77] = TWENB[77];
  assign TWENB_[78] = TWENB[78];
  assign TWENB_[79] = TWENB[79];
  assign TWENB_[80] = TWENB[80];
  assign TWENB_[81] = TWENB[81];
  assign TWENB_[82] = TWENB[82];
  assign TWENB_[83] = TWENB[83];
  assign TWENB_[84] = TWENB[84];
  assign TWENB_[85] = TWENB[85];
  assign TWENB_[86] = TWENB[86];
  assign TWENB_[87] = TWENB[87];
  assign TWENB_[88] = TWENB[88];
  assign TWENB_[89] = TWENB[89];
  assign TWENB_[90] = TWENB[90];
  assign TWENB_[91] = TWENB[91];
  assign TWENB_[92] = TWENB[92];
  assign TWENB_[93] = TWENB[93];
  assign TWENB_[94] = TWENB[94];
  assign TWENB_[95] = TWENB[95];
  assign TWENB_[96] = TWENB[96];
  assign TWENB_[97] = TWENB[97];
  assign TWENB_[98] = TWENB[98];
  assign TWENB_[99] = TWENB[99];
  assign TWENB_[100] = TWENB[100];
  assign TWENB_[101] = TWENB[101];
  assign TWENB_[102] = TWENB[102];
  assign TWENB_[103] = TWENB[103];
  assign TWENB_[104] = TWENB[104];
  assign TWENB_[105] = TWENB[105];
  assign TWENB_[106] = TWENB[106];
  assign TWENB_[107] = TWENB[107];
  assign TWENB_[108] = TWENB[108];
  assign TWENB_[109] = TWENB[109];
  assign TWENB_[110] = TWENB[110];
  assign TWENB_[111] = TWENB[111];
  assign TWENB_[112] = TWENB[112];
  assign TWENB_[113] = TWENB[113];
  assign TWENB_[114] = TWENB[114];
  assign TWENB_[115] = TWENB[115];
  assign TWENB_[116] = TWENB[116];
  assign TWENB_[117] = TWENB[117];
  assign TWENB_[118] = TWENB[118];
  assign TWENB_[119] = TWENB[119];
  assign TWENB_[120] = TWENB[120];
  assign TWENB_[121] = TWENB[121];
  assign TWENB_[122] = TWENB[122];
  assign TWENB_[123] = TWENB[123];
  assign TWENB_[124] = TWENB[124];
  assign TWENB_[125] = TWENB[125];
  assign TWENB_[126] = TWENB[126];
  assign TWENB_[127] = TWENB[127];
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
  assign TDB_[32] = TDB[32];
  assign TDB_[33] = TDB[33];
  assign TDB_[34] = TDB[34];
  assign TDB_[35] = TDB[35];
  assign TDB_[36] = TDB[36];
  assign TDB_[37] = TDB[37];
  assign TDB_[38] = TDB[38];
  assign TDB_[39] = TDB[39];
  assign TDB_[40] = TDB[40];
  assign TDB_[41] = TDB[41];
  assign TDB_[42] = TDB[42];
  assign TDB_[43] = TDB[43];
  assign TDB_[44] = TDB[44];
  assign TDB_[45] = TDB[45];
  assign TDB_[46] = TDB[46];
  assign TDB_[47] = TDB[47];
  assign TDB_[48] = TDB[48];
  assign TDB_[49] = TDB[49];
  assign TDB_[50] = TDB[50];
  assign TDB_[51] = TDB[51];
  assign TDB_[52] = TDB[52];
  assign TDB_[53] = TDB[53];
  assign TDB_[54] = TDB[54];
  assign TDB_[55] = TDB[55];
  assign TDB_[56] = TDB[56];
  assign TDB_[57] = TDB[57];
  assign TDB_[58] = TDB[58];
  assign TDB_[59] = TDB[59];
  assign TDB_[60] = TDB[60];
  assign TDB_[61] = TDB[61];
  assign TDB_[62] = TDB[62];
  assign TDB_[63] = TDB[63];
  assign TDB_[64] = TDB[64];
  assign TDB_[65] = TDB[65];
  assign TDB_[66] = TDB[66];
  assign TDB_[67] = TDB[67];
  assign TDB_[68] = TDB[68];
  assign TDB_[69] = TDB[69];
  assign TDB_[70] = TDB[70];
  assign TDB_[71] = TDB[71];
  assign TDB_[72] = TDB[72];
  assign TDB_[73] = TDB[73];
  assign TDB_[74] = TDB[74];
  assign TDB_[75] = TDB[75];
  assign TDB_[76] = TDB[76];
  assign TDB_[77] = TDB[77];
  assign TDB_[78] = TDB[78];
  assign TDB_[79] = TDB[79];
  assign TDB_[80] = TDB[80];
  assign TDB_[81] = TDB[81];
  assign TDB_[82] = TDB[82];
  assign TDB_[83] = TDB[83];
  assign TDB_[84] = TDB[84];
  assign TDB_[85] = TDB[85];
  assign TDB_[86] = TDB[86];
  assign TDB_[87] = TDB[87];
  assign TDB_[88] = TDB[88];
  assign TDB_[89] = TDB[89];
  assign TDB_[90] = TDB[90];
  assign TDB_[91] = TDB[91];
  assign TDB_[92] = TDB[92];
  assign TDB_[93] = TDB[93];
  assign TDB_[94] = TDB[94];
  assign TDB_[95] = TDB[95];
  assign TDB_[96] = TDB[96];
  assign TDB_[97] = TDB[97];
  assign TDB_[98] = TDB[98];
  assign TDB_[99] = TDB[99];
  assign TDB_[100] = TDB[100];
  assign TDB_[101] = TDB[101];
  assign TDB_[102] = TDB[102];
  assign TDB_[103] = TDB[103];
  assign TDB_[104] = TDB[104];
  assign TDB_[105] = TDB[105];
  assign TDB_[106] = TDB[106];
  assign TDB_[107] = TDB[107];
  assign TDB_[108] = TDB[108];
  assign TDB_[109] = TDB[109];
  assign TDB_[110] = TDB[110];
  assign TDB_[111] = TDB[111];
  assign TDB_[112] = TDB[112];
  assign TDB_[113] = TDB[113];
  assign TDB_[114] = TDB[114];
  assign TDB_[115] = TDB[115];
  assign TDB_[116] = TDB[116];
  assign TDB_[117] = TDB[117];
  assign TDB_[118] = TDB[118];
  assign TDB_[119] = TDB[119];
  assign TDB_[120] = TDB[120];
  assign TDB_[121] = TDB[121];
  assign TDB_[122] = TDB[122];
  assign TDB_[123] = TDB[123];
  assign TDB_[124] = TDB[124];
  assign TDB_[125] = TDB[125];
  assign TDB_[126] = TDB[126];
  assign TDB_[127] = TDB[127];
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
  assign WENYA_ = RET1N_ ? ({128{DFTRAMBYP_}} & (TENA_ ? WENA_ : TWENA_)) : {128{1'bx}};
  assign AYA_ = RET1N_ ? ({10{DFTRAMBYP_}} & (TENA_ ? AA_ : TAA_)) : {10{1'bx}};
  assign CENYB_ = RET1N_ ? (DFTRAMBYP_ & (TENB_ ? CENB_ : TCENB_)) : 1'bx;
  assign WENYB_ = RET1N_ ? ({128{DFTRAMBYP_}} & (TENB_ ? WENB_ : TWENB_)) : {128{1'bx}};
  assign AYB_ = RET1N_ ? ({10{DFTRAMBYP_}} & (TENB_ ? AB_ : TAB_)) : {10{1'bx}};
  assign GWENYA_ = RET1N_ ? (DFTRAMBYP_ & (TENA_ ? GWENA_ : TGWENA_)) : 1'bx;
  assign GWENYB_ = RET1N_ ? (DFTRAMBYP_ & (TENB_ ? GWENB_ : TGWENB_)) : 1'bx;
  assign QA_ = RET1N_ ? ((QA_int)) : {128{1'bx}};
  assign QB_ = RET1N_ ? ((QB_int)) : {128{1'bx}};
  assign SOA_ = RET1N_ ? ({QA_[64], QA_[63]}) : {2{1'bx}};
  assign SOB_ = RET1N_ ? ({QB_[64], QB_[63]}) : {2{1'bx}};

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
     emu_rowa = mem[emu_AA_int >> 2];
   end
`endif

  task readWriteA;
  begin
    if (RET1N_int === 1'b0 && (CENA_int === 1'b0 || DFTRAMBYP_inta === 1'b1)) begin
    end else if (RET1N_int === 1'b0) begin
      // no cycle in retention mode
    end else if ((AA_int >= WORDS) && (CENA_int === 1'b0) && DFTRAMBYP_inta === 1'b0) begin
    end else if (CENA_int === 1'b0 || DFTRAMBYP_inta === 1'b1) begin
      mux_addressa = (AA_int & 2'b11);
      row_addressa = (AA_int >> 2);
`ifdef ARM_EMU_FAULT
	rowa = emu_rowa;
`else
      if (DFTRAMBYP_inta !== 1'b1) begin
      if (row_addressa > 255)
        rowa = {512{1'bx}};
      else
        rowa = mem[row_addressa];
      end
`endif
        writeEnablea = ~ ( {128{GWENA_int}} | {WENA_int[127], WENA_int[126], WENA_int[125],
          WENA_int[124], WENA_int[123], WENA_int[122], WENA_int[121], WENA_int[120],
          WENA_int[119], WENA_int[118], WENA_int[117], WENA_int[116], WENA_int[115],
          WENA_int[114], WENA_int[113], WENA_int[112], WENA_int[111], WENA_int[110],
          WENA_int[109], WENA_int[108], WENA_int[107], WENA_int[106], WENA_int[105],
          WENA_int[104], WENA_int[103], WENA_int[102], WENA_int[101], WENA_int[100],
          WENA_int[99], WENA_int[98], WENA_int[97], WENA_int[96], WENA_int[95], WENA_int[94],
          WENA_int[93], WENA_int[92], WENA_int[91], WENA_int[90], WENA_int[89], WENA_int[88],
          WENA_int[87], WENA_int[86], WENA_int[85], WENA_int[84], WENA_int[83], WENA_int[82],
          WENA_int[81], WENA_int[80], WENA_int[79], WENA_int[78], WENA_int[77], WENA_int[76],
          WENA_int[75], WENA_int[74], WENA_int[73], WENA_int[72], WENA_int[71], WENA_int[70],
          WENA_int[69], WENA_int[68], WENA_int[67], WENA_int[66], WENA_int[65], WENA_int[64],
          WENA_int[63], WENA_int[62], WENA_int[61], WENA_int[60], WENA_int[59], WENA_int[58],
          WENA_int[57], WENA_int[56], WENA_int[55], WENA_int[54], WENA_int[53], WENA_int[52],
          WENA_int[51], WENA_int[50], WENA_int[49], WENA_int[48], WENA_int[47], WENA_int[46],
          WENA_int[45], WENA_int[44], WENA_int[43], WENA_int[42], WENA_int[41], WENA_int[40],
          WENA_int[39], WENA_int[38], WENA_int[37], WENA_int[36], WENA_int[35], WENA_int[34],
          WENA_int[33], WENA_int[32], WENA_int[31], WENA_int[30], WENA_int[29], WENA_int[28],
          WENA_int[27], WENA_int[26], WENA_int[25], WENA_int[24], WENA_int[23], WENA_int[22],
          WENA_int[21], WENA_int[20], WENA_int[19], WENA_int[18], WENA_int[17], WENA_int[16],
          WENA_int[15], WENA_int[14], WENA_int[13], WENA_int[12], WENA_int[11], WENA_int[10],
          WENA_int[9], WENA_int[8], WENA_int[7], WENA_int[6], WENA_int[5], WENA_int[4],
          WENA_int[3], WENA_int[2], WENA_int[1], WENA_int[0]});
      if (GWENA_int !== 1'b1 || DFTRAMBYP_inta === 1'b1) begin
        row_maska =  ( {3'b000, writeEnablea[127], 3'b000, writeEnablea[126], 3'b000, writeEnablea[125],
          3'b000, writeEnablea[124], 3'b000, writeEnablea[123], 3'b000, writeEnablea[122],
          3'b000, writeEnablea[121], 3'b000, writeEnablea[120], 3'b000, writeEnablea[119],
          3'b000, writeEnablea[118], 3'b000, writeEnablea[117], 3'b000, writeEnablea[116],
          3'b000, writeEnablea[115], 3'b000, writeEnablea[114], 3'b000, writeEnablea[113],
          3'b000, writeEnablea[112], 3'b000, writeEnablea[111], 3'b000, writeEnablea[110],
          3'b000, writeEnablea[109], 3'b000, writeEnablea[108], 3'b000, writeEnablea[107],
          3'b000, writeEnablea[106], 3'b000, writeEnablea[105], 3'b000, writeEnablea[104],
          3'b000, writeEnablea[103], 3'b000, writeEnablea[102], 3'b000, writeEnablea[101],
          3'b000, writeEnablea[100], 3'b000, writeEnablea[99], 3'b000, writeEnablea[98],
          3'b000, writeEnablea[97], 3'b000, writeEnablea[96], 3'b000, writeEnablea[95],
          3'b000, writeEnablea[94], 3'b000, writeEnablea[93], 3'b000, writeEnablea[92],
          3'b000, writeEnablea[91], 3'b000, writeEnablea[90], 3'b000, writeEnablea[89],
          3'b000, writeEnablea[88], 3'b000, writeEnablea[87], 3'b000, writeEnablea[86],
          3'b000, writeEnablea[85], 3'b000, writeEnablea[84], 3'b000, writeEnablea[83],
          3'b000, writeEnablea[82], 3'b000, writeEnablea[81], 3'b000, writeEnablea[80],
          3'b000, writeEnablea[79], 3'b000, writeEnablea[78], 3'b000, writeEnablea[77],
          3'b000, writeEnablea[76], 3'b000, writeEnablea[75], 3'b000, writeEnablea[74],
          3'b000, writeEnablea[73], 3'b000, writeEnablea[72], 3'b000, writeEnablea[71],
          3'b000, writeEnablea[70], 3'b000, writeEnablea[69], 3'b000, writeEnablea[68],
          3'b000, writeEnablea[67], 3'b000, writeEnablea[66], 3'b000, writeEnablea[65],
          3'b000, writeEnablea[64], 3'b000, writeEnablea[63], 3'b000, writeEnablea[62],
          3'b000, writeEnablea[61], 3'b000, writeEnablea[60], 3'b000, writeEnablea[59],
          3'b000, writeEnablea[58], 3'b000, writeEnablea[57], 3'b000, writeEnablea[56],
          3'b000, writeEnablea[55], 3'b000, writeEnablea[54], 3'b000, writeEnablea[53],
          3'b000, writeEnablea[52], 3'b000, writeEnablea[51], 3'b000, writeEnablea[50],
          3'b000, writeEnablea[49], 3'b000, writeEnablea[48], 3'b000, writeEnablea[47],
          3'b000, writeEnablea[46], 3'b000, writeEnablea[45], 3'b000, writeEnablea[44],
          3'b000, writeEnablea[43], 3'b000, writeEnablea[42], 3'b000, writeEnablea[41],
          3'b000, writeEnablea[40], 3'b000, writeEnablea[39], 3'b000, writeEnablea[38],
          3'b000, writeEnablea[37], 3'b000, writeEnablea[36], 3'b000, writeEnablea[35],
          3'b000, writeEnablea[34], 3'b000, writeEnablea[33], 3'b000, writeEnablea[32],
          3'b000, writeEnablea[31], 3'b000, writeEnablea[30], 3'b000, writeEnablea[29],
          3'b000, writeEnablea[28], 3'b000, writeEnablea[27], 3'b000, writeEnablea[26],
          3'b000, writeEnablea[25], 3'b000, writeEnablea[24], 3'b000, writeEnablea[23],
          3'b000, writeEnablea[22], 3'b000, writeEnablea[21], 3'b000, writeEnablea[20],
          3'b000, writeEnablea[19], 3'b000, writeEnablea[18], 3'b000, writeEnablea[17],
          3'b000, writeEnablea[16], 3'b000, writeEnablea[15], 3'b000, writeEnablea[14],
          3'b000, writeEnablea[13], 3'b000, writeEnablea[12], 3'b000, writeEnablea[11],
          3'b000, writeEnablea[10], 3'b000, writeEnablea[9], 3'b000, writeEnablea[8],
          3'b000, writeEnablea[7], 3'b000, writeEnablea[6], 3'b000, writeEnablea[5],
          3'b000, writeEnablea[4], 3'b000, writeEnablea[3], 3'b000, writeEnablea[2],
          3'b000, writeEnablea[1], 3'b000, writeEnablea[0]} << mux_addressa);
        new_dataa =  ( {3'b000, DA_int[127], 3'b000, DA_int[126], 3'b000, DA_int[125],
          3'b000, DA_int[124], 3'b000, DA_int[123], 3'b000, DA_int[122], 3'b000, DA_int[121],
          3'b000, DA_int[120], 3'b000, DA_int[119], 3'b000, DA_int[118], 3'b000, DA_int[117],
          3'b000, DA_int[116], 3'b000, DA_int[115], 3'b000, DA_int[114], 3'b000, DA_int[113],
          3'b000, DA_int[112], 3'b000, DA_int[111], 3'b000, DA_int[110], 3'b000, DA_int[109],
          3'b000, DA_int[108], 3'b000, DA_int[107], 3'b000, DA_int[106], 3'b000, DA_int[105],
          3'b000, DA_int[104], 3'b000, DA_int[103], 3'b000, DA_int[102], 3'b000, DA_int[101],
          3'b000, DA_int[100], 3'b000, DA_int[99], 3'b000, DA_int[98], 3'b000, DA_int[97],
          3'b000, DA_int[96], 3'b000, DA_int[95], 3'b000, DA_int[94], 3'b000, DA_int[93],
          3'b000, DA_int[92], 3'b000, DA_int[91], 3'b000, DA_int[90], 3'b000, DA_int[89],
          3'b000, DA_int[88], 3'b000, DA_int[87], 3'b000, DA_int[86], 3'b000, DA_int[85],
          3'b000, DA_int[84], 3'b000, DA_int[83], 3'b000, DA_int[82], 3'b000, DA_int[81],
          3'b000, DA_int[80], 3'b000, DA_int[79], 3'b000, DA_int[78], 3'b000, DA_int[77],
          3'b000, DA_int[76], 3'b000, DA_int[75], 3'b000, DA_int[74], 3'b000, DA_int[73],
          3'b000, DA_int[72], 3'b000, DA_int[71], 3'b000, DA_int[70], 3'b000, DA_int[69],
          3'b000, DA_int[68], 3'b000, DA_int[67], 3'b000, DA_int[66], 3'b000, DA_int[65],
          3'b000, DA_int[64], 3'b000, DA_int[63], 3'b000, DA_int[62], 3'b000, DA_int[61],
          3'b000, DA_int[60], 3'b000, DA_int[59], 3'b000, DA_int[58], 3'b000, DA_int[57],
          3'b000, DA_int[56], 3'b000, DA_int[55], 3'b000, DA_int[54], 3'b000, DA_int[53],
          3'b000, DA_int[52], 3'b000, DA_int[51], 3'b000, DA_int[50], 3'b000, DA_int[49],
          3'b000, DA_int[48], 3'b000, DA_int[47], 3'b000, DA_int[46], 3'b000, DA_int[45],
          3'b000, DA_int[44], 3'b000, DA_int[43], 3'b000, DA_int[42], 3'b000, DA_int[41],
          3'b000, DA_int[40], 3'b000, DA_int[39], 3'b000, DA_int[38], 3'b000, DA_int[37],
          3'b000, DA_int[36], 3'b000, DA_int[35], 3'b000, DA_int[34], 3'b000, DA_int[33],
          3'b000, DA_int[32], 3'b000, DA_int[31], 3'b000, DA_int[30], 3'b000, DA_int[29],
          3'b000, DA_int[28], 3'b000, DA_int[27], 3'b000, DA_int[26], 3'b000, DA_int[25],
          3'b000, DA_int[24], 3'b000, DA_int[23], 3'b000, DA_int[22], 3'b000, DA_int[21],
          3'b000, DA_int[20], 3'b000, DA_int[19], 3'b000, DA_int[18], 3'b000, DA_int[17],
          3'b000, DA_int[16], 3'b000, DA_int[15], 3'b000, DA_int[14], 3'b000, DA_int[13],
          3'b000, DA_int[12], 3'b000, DA_int[11], 3'b000, DA_int[10], 3'b000, DA_int[9],
          3'b000, DA_int[8], 3'b000, DA_int[7], 3'b000, DA_int[6], 3'b000, DA_int[5],
          3'b000, DA_int[4], 3'b000, DA_int[3], 3'b000, DA_int[2], 3'b000, DA_int[1],
          3'b000, DA_int[0]} << mux_addressa);
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
        shifted_readLatch0 = readLatch0;
        mem_path_A = {shifted_readLatch0[127], shifted_readLatch0[126], shifted_readLatch0[125],
          shifted_readLatch0[124], shifted_readLatch0[123], shifted_readLatch0[122],
          shifted_readLatch0[121], shifted_readLatch0[120], shifted_readLatch0[119],
          shifted_readLatch0[118], shifted_readLatch0[117], shifted_readLatch0[116],
          shifted_readLatch0[115], shifted_readLatch0[114], shifted_readLatch0[113],
          shifted_readLatch0[112], shifted_readLatch0[111], shifted_readLatch0[110],
          shifted_readLatch0[109], shifted_readLatch0[108], shifted_readLatch0[107],
          shifted_readLatch0[106], shifted_readLatch0[105], shifted_readLatch0[104],
          shifted_readLatch0[103], shifted_readLatch0[102], shifted_readLatch0[101],
          shifted_readLatch0[100], shifted_readLatch0[99], shifted_readLatch0[98],
          shifted_readLatch0[97], shifted_readLatch0[96], shifted_readLatch0[95], shifted_readLatch0[94],
          shifted_readLatch0[93], shifted_readLatch0[92], shifted_readLatch0[91], shifted_readLatch0[90],
          shifted_readLatch0[89], shifted_readLatch0[88], shifted_readLatch0[87], shifted_readLatch0[86],
          shifted_readLatch0[85], shifted_readLatch0[84], shifted_readLatch0[83], shifted_readLatch0[82],
          shifted_readLatch0[81], shifted_readLatch0[80], shifted_readLatch0[79], shifted_readLatch0[78],
          shifted_readLatch0[77], shifted_readLatch0[76], shifted_readLatch0[75], shifted_readLatch0[74],
          shifted_readLatch0[73], shifted_readLatch0[72], shifted_readLatch0[71], shifted_readLatch0[70],
          shifted_readLatch0[69], shifted_readLatch0[68], shifted_readLatch0[67], shifted_readLatch0[66],
          shifted_readLatch0[65], shifted_readLatch0[64], shifted_readLatch0[63], shifted_readLatch0[62],
          shifted_readLatch0[61], shifted_readLatch0[60], shifted_readLatch0[59], shifted_readLatch0[58],
          shifted_readLatch0[57], shifted_readLatch0[56], shifted_readLatch0[55], shifted_readLatch0[54],
          shifted_readLatch0[53], shifted_readLatch0[52], shifted_readLatch0[51], shifted_readLatch0[50],
          shifted_readLatch0[49], shifted_readLatch0[48], shifted_readLatch0[47], shifted_readLatch0[46],
          shifted_readLatch0[45], shifted_readLatch0[44], shifted_readLatch0[43], shifted_readLatch0[42],
          shifted_readLatch0[41], shifted_readLatch0[40], shifted_readLatch0[39], shifted_readLatch0[38],
          shifted_readLatch0[37], shifted_readLatch0[36], shifted_readLatch0[35], shifted_readLatch0[34],
          shifted_readLatch0[33], shifted_readLatch0[32], shifted_readLatch0[31], shifted_readLatch0[30],
          shifted_readLatch0[29], shifted_readLatch0[28], shifted_readLatch0[27], shifted_readLatch0[26],
          shifted_readLatch0[25], shifted_readLatch0[24], shifted_readLatch0[23], shifted_readLatch0[22],
          shifted_readLatch0[21], shifted_readLatch0[20], shifted_readLatch0[19], shifted_readLatch0[18],
          shifted_readLatch0[17], shifted_readLatch0[16], shifted_readLatch0[15], shifted_readLatch0[14],
          shifted_readLatch0[13], shifted_readLatch0[12], shifted_readLatch0[11], shifted_readLatch0[10],
          shifted_readLatch0[9], shifted_readLatch0[8], shifted_readLatch0[7], shifted_readLatch0[6],
          shifted_readLatch0[5], shifted_readLatch0[4], shifted_readLatch0[3], shifted_readLatch0[2],
          shifted_readLatch0[1], shifted_readLatch0[0]};
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
      end
      if (DFTRAMBYP_=== 1'b1 && SEA_ === 1'b1) begin
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
      end
    readWriteA;
      end
    #0;
`ifdef NO_COLLISIONS
`else     
      if ((1) && (CENA_int !== 1'b1 && CENB_int !== 1'b1 && DFTRAMBYP_ !== 1'b1) && COLLDISN_int === 1'b1 && row_contention(AA_int, AB_int, ({128{GWENA_int}}|WENA_int),
        ({128{GWENB_int}}|WENB_int))) begin
        if (col_contention(AA_int, AB_int)) begin
          COL_CC = 1;
        end
          ROW_CC = 1;
          READ_READ_1 = 0;
          READ_WRITE_1 = 0;
          WRITE_WRITE_1 = 0;
        if (GWENA_int !== 1'b1 && (& WENA_int) !== 1'b1 && GWENB_int !== 1'b1 && (& WENB_int) !== 1'b1) begin
	      if (is_contention(AA_int, AB_int, ({128{GWENA_int}}|WENA_int), ({128{GWENB_int}}|WENB_int))) begin
          COL_CC = 1;
          WRITE_WRITE = 1;
	      end
        end else if (GWENA_int !== 1'b1 && (& WENA_int) !== 1'b1) begin
		if (is_contention(AA_int, AB_int, ({128{GWENA_int}}|WENA_int), ({128{GWENB_int}}|WENB_int))) begin
          COL_CC = 1;
          READ_WRITE = 1;
		end
        end else if (GWENB_int !== 1'b1 && (& WENB_int) !== 1'b1) begin
		if (is_contention(AA_int, AB_int, ({128{GWENA_int}}|WENA_int), ({128{GWENB_int}}|WENB_int))) begin
          COL_CC = 1;
          READ_WRITE = 1;
		end
        end else begin
          COL_CC = 1;
          READ_READ = 1;
        end
        if (!is_contention(AA_int, AB_int, ({128{GWENA_int}}|WENA_int), ({128{GWENB_int}}|WENB_int))) begin
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
        AB_int, ({128{GWENA_int}}|WENA_int), ({128{GWENB_int}}|WENB_int))) begin
          ROW_CC = 1;
          READ_READ_1 = 0;
          READ_WRITE_1 = 0;
          WRITE_WRITE_1 = 0;
        if (col_contention(AA_int, AB_int)) begin
          COL_CC = 1;
        end
        if (GWENB_int !== 1'b1 && (& WENB_int) !== 1'b1) begin
          WRITE_WRITE_1 = 1;
        end else if (is_contention(AA_int, AB_int, ({128{GWENA_int}}|WENA_int), ({128{GWENB_int}}|WENB_int))) begin
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
        end else if (is_contention(AA_int, AB_int, ({128{GWENA_int}}|WENA_int), ({128{GWENB_int}}|WENB_int))) begin
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
  end

  assign SIA_int = SEA_ ? SIA_ : {2{1'b0}};
  assign DA_int_bmux = TENA_ ? DA_ : TDA_;

  datapath_latch_sram_128_1024_dp uDQA0 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(SIA_int[0]), .D(DA_int_bmux[0]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[0]), .XQ(XQA), .Q(QA_int[0]));
  datapath_latch_sram_128_1024_dp uDQA1 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[0]), .D(DA_int_bmux[1]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[1]), .XQ(XQA), .Q(QA_int[1]));
  datapath_latch_sram_128_1024_dp uDQA2 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[1]), .D(DA_int_bmux[2]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[2]), .XQ(XQA), .Q(QA_int[2]));
  datapath_latch_sram_128_1024_dp uDQA3 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[2]), .D(DA_int_bmux[3]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[3]), .XQ(XQA), .Q(QA_int[3]));
  datapath_latch_sram_128_1024_dp uDQA4 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[3]), .D(DA_int_bmux[4]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[4]), .XQ(XQA), .Q(QA_int[4]));
  datapath_latch_sram_128_1024_dp uDQA5 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[4]), .D(DA_int_bmux[5]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[5]), .XQ(XQA), .Q(QA_int[5]));
  datapath_latch_sram_128_1024_dp uDQA6 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[5]), .D(DA_int_bmux[6]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[6]), .XQ(XQA), .Q(QA_int[6]));
  datapath_latch_sram_128_1024_dp uDQA7 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[6]), .D(DA_int_bmux[7]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[7]), .XQ(XQA), .Q(QA_int[7]));
  datapath_latch_sram_128_1024_dp uDQA8 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[7]), .D(DA_int_bmux[8]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[8]), .XQ(XQA), .Q(QA_int[8]));
  datapath_latch_sram_128_1024_dp uDQA9 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[8]), .D(DA_int_bmux[9]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[9]), .XQ(XQA), .Q(QA_int[9]));
  datapath_latch_sram_128_1024_dp uDQA10 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[9]), .D(DA_int_bmux[10]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[10]), .XQ(XQA), .Q(QA_int[10]));
  datapath_latch_sram_128_1024_dp uDQA11 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[10]), .D(DA_int_bmux[11]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[11]), .XQ(XQA), .Q(QA_int[11]));
  datapath_latch_sram_128_1024_dp uDQA12 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[11]), .D(DA_int_bmux[12]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[12]), .XQ(XQA), .Q(QA_int[12]));
  datapath_latch_sram_128_1024_dp uDQA13 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[12]), .D(DA_int_bmux[13]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[13]), .XQ(XQA), .Q(QA_int[13]));
  datapath_latch_sram_128_1024_dp uDQA14 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[13]), .D(DA_int_bmux[14]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[14]), .XQ(XQA), .Q(QA_int[14]));
  datapath_latch_sram_128_1024_dp uDQA15 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[14]), .D(DA_int_bmux[15]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[15]), .XQ(XQA), .Q(QA_int[15]));
  datapath_latch_sram_128_1024_dp uDQA16 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[15]), .D(DA_int_bmux[16]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[16]), .XQ(XQA), .Q(QA_int[16]));
  datapath_latch_sram_128_1024_dp uDQA17 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[16]), .D(DA_int_bmux[17]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[17]), .XQ(XQA), .Q(QA_int[17]));
  datapath_latch_sram_128_1024_dp uDQA18 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[17]), .D(DA_int_bmux[18]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[18]), .XQ(XQA), .Q(QA_int[18]));
  datapath_latch_sram_128_1024_dp uDQA19 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[18]), .D(DA_int_bmux[19]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[19]), .XQ(XQA), .Q(QA_int[19]));
  datapath_latch_sram_128_1024_dp uDQA20 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[19]), .D(DA_int_bmux[20]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[20]), .XQ(XQA), .Q(QA_int[20]));
  datapath_latch_sram_128_1024_dp uDQA21 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[20]), .D(DA_int_bmux[21]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[21]), .XQ(XQA), .Q(QA_int[21]));
  datapath_latch_sram_128_1024_dp uDQA22 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[21]), .D(DA_int_bmux[22]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[22]), .XQ(XQA), .Q(QA_int[22]));
  datapath_latch_sram_128_1024_dp uDQA23 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[22]), .D(DA_int_bmux[23]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[23]), .XQ(XQA), .Q(QA_int[23]));
  datapath_latch_sram_128_1024_dp uDQA24 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[23]), .D(DA_int_bmux[24]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[24]), .XQ(XQA), .Q(QA_int[24]));
  datapath_latch_sram_128_1024_dp uDQA25 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[24]), .D(DA_int_bmux[25]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[25]), .XQ(XQA), .Q(QA_int[25]));
  datapath_latch_sram_128_1024_dp uDQA26 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[25]), .D(DA_int_bmux[26]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[26]), .XQ(XQA), .Q(QA_int[26]));
  datapath_latch_sram_128_1024_dp uDQA27 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[26]), .D(DA_int_bmux[27]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[27]), .XQ(XQA), .Q(QA_int[27]));
  datapath_latch_sram_128_1024_dp uDQA28 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[27]), .D(DA_int_bmux[28]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[28]), .XQ(XQA), .Q(QA_int[28]));
  datapath_latch_sram_128_1024_dp uDQA29 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[28]), .D(DA_int_bmux[29]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[29]), .XQ(XQA), .Q(QA_int[29]));
  datapath_latch_sram_128_1024_dp uDQA30 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[29]), .D(DA_int_bmux[30]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[30]), .XQ(XQA), .Q(QA_int[30]));
  datapath_latch_sram_128_1024_dp uDQA31 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[30]), .D(DA_int_bmux[31]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[31]), .XQ(XQA), .Q(QA_int[31]));
  datapath_latch_sram_128_1024_dp uDQA32 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[31]), .D(DA_int_bmux[32]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[32]), .XQ(XQA), .Q(QA_int[32]));
  datapath_latch_sram_128_1024_dp uDQA33 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[32]), .D(DA_int_bmux[33]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[33]), .XQ(XQA), .Q(QA_int[33]));
  datapath_latch_sram_128_1024_dp uDQA34 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[33]), .D(DA_int_bmux[34]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[34]), .XQ(XQA), .Q(QA_int[34]));
  datapath_latch_sram_128_1024_dp uDQA35 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[34]), .D(DA_int_bmux[35]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[35]), .XQ(XQA), .Q(QA_int[35]));
  datapath_latch_sram_128_1024_dp uDQA36 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[35]), .D(DA_int_bmux[36]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[36]), .XQ(XQA), .Q(QA_int[36]));
  datapath_latch_sram_128_1024_dp uDQA37 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[36]), .D(DA_int_bmux[37]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[37]), .XQ(XQA), .Q(QA_int[37]));
  datapath_latch_sram_128_1024_dp uDQA38 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[37]), .D(DA_int_bmux[38]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[38]), .XQ(XQA), .Q(QA_int[38]));
  datapath_latch_sram_128_1024_dp uDQA39 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[38]), .D(DA_int_bmux[39]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[39]), .XQ(XQA), .Q(QA_int[39]));
  datapath_latch_sram_128_1024_dp uDQA40 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[39]), .D(DA_int_bmux[40]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[40]), .XQ(XQA), .Q(QA_int[40]));
  datapath_latch_sram_128_1024_dp uDQA41 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[40]), .D(DA_int_bmux[41]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[41]), .XQ(XQA), .Q(QA_int[41]));
  datapath_latch_sram_128_1024_dp uDQA42 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[41]), .D(DA_int_bmux[42]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[42]), .XQ(XQA), .Q(QA_int[42]));
  datapath_latch_sram_128_1024_dp uDQA43 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[42]), .D(DA_int_bmux[43]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[43]), .XQ(XQA), .Q(QA_int[43]));
  datapath_latch_sram_128_1024_dp uDQA44 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[43]), .D(DA_int_bmux[44]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[44]), .XQ(XQA), .Q(QA_int[44]));
  datapath_latch_sram_128_1024_dp uDQA45 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[44]), .D(DA_int_bmux[45]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[45]), .XQ(XQA), .Q(QA_int[45]));
  datapath_latch_sram_128_1024_dp uDQA46 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[45]), .D(DA_int_bmux[46]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[46]), .XQ(XQA), .Q(QA_int[46]));
  datapath_latch_sram_128_1024_dp uDQA47 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[46]), .D(DA_int_bmux[47]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[47]), .XQ(XQA), .Q(QA_int[47]));
  datapath_latch_sram_128_1024_dp uDQA48 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[47]), .D(DA_int_bmux[48]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[48]), .XQ(XQA), .Q(QA_int[48]));
  datapath_latch_sram_128_1024_dp uDQA49 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[48]), .D(DA_int_bmux[49]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[49]), .XQ(XQA), .Q(QA_int[49]));
  datapath_latch_sram_128_1024_dp uDQA50 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[49]), .D(DA_int_bmux[50]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[50]), .XQ(XQA), .Q(QA_int[50]));
  datapath_latch_sram_128_1024_dp uDQA51 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[50]), .D(DA_int_bmux[51]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[51]), .XQ(XQA), .Q(QA_int[51]));
  datapath_latch_sram_128_1024_dp uDQA52 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[51]), .D(DA_int_bmux[52]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[52]), .XQ(XQA), .Q(QA_int[52]));
  datapath_latch_sram_128_1024_dp uDQA53 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[52]), .D(DA_int_bmux[53]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[53]), .XQ(XQA), .Q(QA_int[53]));
  datapath_latch_sram_128_1024_dp uDQA54 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[53]), .D(DA_int_bmux[54]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[54]), .XQ(XQA), .Q(QA_int[54]));
  datapath_latch_sram_128_1024_dp uDQA55 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[54]), .D(DA_int_bmux[55]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[55]), .XQ(XQA), .Q(QA_int[55]));
  datapath_latch_sram_128_1024_dp uDQA56 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[55]), .D(DA_int_bmux[56]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[56]), .XQ(XQA), .Q(QA_int[56]));
  datapath_latch_sram_128_1024_dp uDQA57 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[56]), .D(DA_int_bmux[57]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[57]), .XQ(XQA), .Q(QA_int[57]));
  datapath_latch_sram_128_1024_dp uDQA58 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[57]), .D(DA_int_bmux[58]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[58]), .XQ(XQA), .Q(QA_int[58]));
  datapath_latch_sram_128_1024_dp uDQA59 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[58]), .D(DA_int_bmux[59]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[59]), .XQ(XQA), .Q(QA_int[59]));
  datapath_latch_sram_128_1024_dp uDQA60 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[59]), .D(DA_int_bmux[60]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[60]), .XQ(XQA), .Q(QA_int[60]));
  datapath_latch_sram_128_1024_dp uDQA61 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[60]), .D(DA_int_bmux[61]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[61]), .XQ(XQA), .Q(QA_int[61]));
  datapath_latch_sram_128_1024_dp uDQA62 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[61]), .D(DA_int_bmux[62]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[62]), .XQ(XQA), .Q(QA_int[62]));
  datapath_latch_sram_128_1024_dp uDQA63 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[62]), .D(DA_int_bmux[63]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[63]), .XQ(XQA), .Q(QA_int[63]));
  datapath_latch_sram_128_1024_dp uDQA64 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[65]), .D(DA_int_bmux[64]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[64]), .XQ(XQA), .Q(QA_int[64]));
  datapath_latch_sram_128_1024_dp uDQA65 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[66]), .D(DA_int_bmux[65]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[65]), .XQ(XQA), .Q(QA_int[65]));
  datapath_latch_sram_128_1024_dp uDQA66 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[67]), .D(DA_int_bmux[66]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[66]), .XQ(XQA), .Q(QA_int[66]));
  datapath_latch_sram_128_1024_dp uDQA67 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[68]), .D(DA_int_bmux[67]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[67]), .XQ(XQA), .Q(QA_int[67]));
  datapath_latch_sram_128_1024_dp uDQA68 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[69]), .D(DA_int_bmux[68]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[68]), .XQ(XQA), .Q(QA_int[68]));
  datapath_latch_sram_128_1024_dp uDQA69 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[70]), .D(DA_int_bmux[69]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[69]), .XQ(XQA), .Q(QA_int[69]));
  datapath_latch_sram_128_1024_dp uDQA70 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[71]), .D(DA_int_bmux[70]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[70]), .XQ(XQA), .Q(QA_int[70]));
  datapath_latch_sram_128_1024_dp uDQA71 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[72]), .D(DA_int_bmux[71]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[71]), .XQ(XQA), .Q(QA_int[71]));
  datapath_latch_sram_128_1024_dp uDQA72 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[73]), .D(DA_int_bmux[72]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[72]), .XQ(XQA), .Q(QA_int[72]));
  datapath_latch_sram_128_1024_dp uDQA73 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[74]), .D(DA_int_bmux[73]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[73]), .XQ(XQA), .Q(QA_int[73]));
  datapath_latch_sram_128_1024_dp uDQA74 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[75]), .D(DA_int_bmux[74]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[74]), .XQ(XQA), .Q(QA_int[74]));
  datapath_latch_sram_128_1024_dp uDQA75 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[76]), .D(DA_int_bmux[75]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[75]), .XQ(XQA), .Q(QA_int[75]));
  datapath_latch_sram_128_1024_dp uDQA76 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[77]), .D(DA_int_bmux[76]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[76]), .XQ(XQA), .Q(QA_int[76]));
  datapath_latch_sram_128_1024_dp uDQA77 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[78]), .D(DA_int_bmux[77]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[77]), .XQ(XQA), .Q(QA_int[77]));
  datapath_latch_sram_128_1024_dp uDQA78 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[79]), .D(DA_int_bmux[78]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[78]), .XQ(XQA), .Q(QA_int[78]));
  datapath_latch_sram_128_1024_dp uDQA79 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[80]), .D(DA_int_bmux[79]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[79]), .XQ(XQA), .Q(QA_int[79]));
  datapath_latch_sram_128_1024_dp uDQA80 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[81]), .D(DA_int_bmux[80]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[80]), .XQ(XQA), .Q(QA_int[80]));
  datapath_latch_sram_128_1024_dp uDQA81 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[82]), .D(DA_int_bmux[81]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[81]), .XQ(XQA), .Q(QA_int[81]));
  datapath_latch_sram_128_1024_dp uDQA82 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[83]), .D(DA_int_bmux[82]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[82]), .XQ(XQA), .Q(QA_int[82]));
  datapath_latch_sram_128_1024_dp uDQA83 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[84]), .D(DA_int_bmux[83]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[83]), .XQ(XQA), .Q(QA_int[83]));
  datapath_latch_sram_128_1024_dp uDQA84 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[85]), .D(DA_int_bmux[84]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[84]), .XQ(XQA), .Q(QA_int[84]));
  datapath_latch_sram_128_1024_dp uDQA85 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[86]), .D(DA_int_bmux[85]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[85]), .XQ(XQA), .Q(QA_int[85]));
  datapath_latch_sram_128_1024_dp uDQA86 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[87]), .D(DA_int_bmux[86]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[86]), .XQ(XQA), .Q(QA_int[86]));
  datapath_latch_sram_128_1024_dp uDQA87 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[88]), .D(DA_int_bmux[87]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[87]), .XQ(XQA), .Q(QA_int[87]));
  datapath_latch_sram_128_1024_dp uDQA88 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[89]), .D(DA_int_bmux[88]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[88]), .XQ(XQA), .Q(QA_int[88]));
  datapath_latch_sram_128_1024_dp uDQA89 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[90]), .D(DA_int_bmux[89]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[89]), .XQ(XQA), .Q(QA_int[89]));
  datapath_latch_sram_128_1024_dp uDQA90 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[91]), .D(DA_int_bmux[90]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[90]), .XQ(XQA), .Q(QA_int[90]));
  datapath_latch_sram_128_1024_dp uDQA91 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[92]), .D(DA_int_bmux[91]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[91]), .XQ(XQA), .Q(QA_int[91]));
  datapath_latch_sram_128_1024_dp uDQA92 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[93]), .D(DA_int_bmux[92]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[92]), .XQ(XQA), .Q(QA_int[92]));
  datapath_latch_sram_128_1024_dp uDQA93 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[94]), .D(DA_int_bmux[93]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[93]), .XQ(XQA), .Q(QA_int[93]));
  datapath_latch_sram_128_1024_dp uDQA94 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[95]), .D(DA_int_bmux[94]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[94]), .XQ(XQA), .Q(QA_int[94]));
  datapath_latch_sram_128_1024_dp uDQA95 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[96]), .D(DA_int_bmux[95]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[95]), .XQ(XQA), .Q(QA_int[95]));
  datapath_latch_sram_128_1024_dp uDQA96 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[97]), .D(DA_int_bmux[96]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[96]), .XQ(XQA), .Q(QA_int[96]));
  datapath_latch_sram_128_1024_dp uDQA97 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[98]), .D(DA_int_bmux[97]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[97]), .XQ(XQA), .Q(QA_int[97]));
  datapath_latch_sram_128_1024_dp uDQA98 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[99]), .D(DA_int_bmux[98]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[98]), .XQ(XQA), .Q(QA_int[98]));
  datapath_latch_sram_128_1024_dp uDQA99 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[100]), .D(DA_int_bmux[99]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[99]), .XQ(XQA), .Q(QA_int[99]));
  datapath_latch_sram_128_1024_dp uDQA100 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[101]), .D(DA_int_bmux[100]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[100]), .XQ(XQA), .Q(QA_int[100]));
  datapath_latch_sram_128_1024_dp uDQA101 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[102]), .D(DA_int_bmux[101]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[101]), .XQ(XQA), .Q(QA_int[101]));
  datapath_latch_sram_128_1024_dp uDQA102 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[103]), .D(DA_int_bmux[102]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[102]), .XQ(XQA), .Q(QA_int[102]));
  datapath_latch_sram_128_1024_dp uDQA103 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[104]), .D(DA_int_bmux[103]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[103]), .XQ(XQA), .Q(QA_int[103]));
  datapath_latch_sram_128_1024_dp uDQA104 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[105]), .D(DA_int_bmux[104]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[104]), .XQ(XQA), .Q(QA_int[104]));
  datapath_latch_sram_128_1024_dp uDQA105 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[106]), .D(DA_int_bmux[105]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[105]), .XQ(XQA), .Q(QA_int[105]));
  datapath_latch_sram_128_1024_dp uDQA106 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[107]), .D(DA_int_bmux[106]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[106]), .XQ(XQA), .Q(QA_int[106]));
  datapath_latch_sram_128_1024_dp uDQA107 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[108]), .D(DA_int_bmux[107]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[107]), .XQ(XQA), .Q(QA_int[107]));
  datapath_latch_sram_128_1024_dp uDQA108 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[109]), .D(DA_int_bmux[108]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[108]), .XQ(XQA), .Q(QA_int[108]));
  datapath_latch_sram_128_1024_dp uDQA109 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[110]), .D(DA_int_bmux[109]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[109]), .XQ(XQA), .Q(QA_int[109]));
  datapath_latch_sram_128_1024_dp uDQA110 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[111]), .D(DA_int_bmux[110]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[110]), .XQ(XQA), .Q(QA_int[110]));
  datapath_latch_sram_128_1024_dp uDQA111 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[112]), .D(DA_int_bmux[111]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[111]), .XQ(XQA), .Q(QA_int[111]));
  datapath_latch_sram_128_1024_dp uDQA112 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[113]), .D(DA_int_bmux[112]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[112]), .XQ(XQA), .Q(QA_int[112]));
  datapath_latch_sram_128_1024_dp uDQA113 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[114]), .D(DA_int_bmux[113]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[113]), .XQ(XQA), .Q(QA_int[113]));
  datapath_latch_sram_128_1024_dp uDQA114 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[115]), .D(DA_int_bmux[114]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[114]), .XQ(XQA), .Q(QA_int[114]));
  datapath_latch_sram_128_1024_dp uDQA115 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[116]), .D(DA_int_bmux[115]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[115]), .XQ(XQA), .Q(QA_int[115]));
  datapath_latch_sram_128_1024_dp uDQA116 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[117]), .D(DA_int_bmux[116]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[116]), .XQ(XQA), .Q(QA_int[116]));
  datapath_latch_sram_128_1024_dp uDQA117 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[118]), .D(DA_int_bmux[117]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[117]), .XQ(XQA), .Q(QA_int[117]));
  datapath_latch_sram_128_1024_dp uDQA118 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[119]), .D(DA_int_bmux[118]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[118]), .XQ(XQA), .Q(QA_int[118]));
  datapath_latch_sram_128_1024_dp uDQA119 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[120]), .D(DA_int_bmux[119]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[119]), .XQ(XQA), .Q(QA_int[119]));
  datapath_latch_sram_128_1024_dp uDQA120 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[121]), .D(DA_int_bmux[120]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[120]), .XQ(XQA), .Q(QA_int[120]));
  datapath_latch_sram_128_1024_dp uDQA121 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[122]), .D(DA_int_bmux[121]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[121]), .XQ(XQA), .Q(QA_int[121]));
  datapath_latch_sram_128_1024_dp uDQA122 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[123]), .D(DA_int_bmux[122]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[122]), .XQ(XQA), .Q(QA_int[122]));
  datapath_latch_sram_128_1024_dp uDQA123 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[124]), .D(DA_int_bmux[123]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[123]), .XQ(XQA), .Q(QA_int[123]));
  datapath_latch_sram_128_1024_dp uDQA124 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[125]), .D(DA_int_bmux[124]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[124]), .XQ(XQA), .Q(QA_int[124]));
  datapath_latch_sram_128_1024_dp uDQA125 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[126]), .D(DA_int_bmux[125]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[125]), .XQ(XQA), .Q(QA_int[125]));
  datapath_latch_sram_128_1024_dp uDQA126 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(QA_int[127]), .D(DA_int_bmux[126]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[126]), .XQ(XQA), .Q(QA_int[126]));
  datapath_latch_sram_128_1024_dp uDQA127 (.CLK(CLKA), .Q_update(QA_update), .D_update(DA_sh_update), .SE(SEA_), .SI(SIA_int[1]), .D(DA_int_bmux[127]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_A[127]), .XQ(XQA), .Q(QA_int[127]));


`ifdef ARM_EMU_FAULT
   reg [AWIDTH-1:0] emu_AB_int;
   reg [MWIDTH-1:0] emu_rowb;

   always @(*) begin
     emu_AB_int = TENB_ ? AB_ : TAB_;
     emu_rowb = mem[emu_AB_int >> 2];
   end
`endif

  task readWriteB;
  begin
    if (RET1N_int === 1'b0 && (CENB_int === 1'b0 || DFTRAMBYP_int === 1'b1)) begin
    end else if (RET1N_int === 1'b0) begin
      // no cycle in retention mode
    end else if ((AB_int >= WORDS) && (CENB_int === 1'b0) && DFTRAMBYP_int === 1'b0) begin
    end else if (CENB_int === 1'b0 || DFTRAMBYP_int === 1'b1) begin
      mux_address = (AB_int & 2'b11);
      row_address = (AB_int >> 2);
`ifdef ARM_EMU_FAULT
	row = emu_rowb;
`else
      if (DFTRAMBYP_int !== 1'b1) begin
      if (row_address > 255)
        row = {512{1'bx}};
      else
        row = mem[row_address];
      end
`endif
        writeEnable = ~ ( {128{GWENB_int}} | {WENB_int[127], WENB_int[126], WENB_int[125],
          WENB_int[124], WENB_int[123], WENB_int[122], WENB_int[121], WENB_int[120],
          WENB_int[119], WENB_int[118], WENB_int[117], WENB_int[116], WENB_int[115],
          WENB_int[114], WENB_int[113], WENB_int[112], WENB_int[111], WENB_int[110],
          WENB_int[109], WENB_int[108], WENB_int[107], WENB_int[106], WENB_int[105],
          WENB_int[104], WENB_int[103], WENB_int[102], WENB_int[101], WENB_int[100],
          WENB_int[99], WENB_int[98], WENB_int[97], WENB_int[96], WENB_int[95], WENB_int[94],
          WENB_int[93], WENB_int[92], WENB_int[91], WENB_int[90], WENB_int[89], WENB_int[88],
          WENB_int[87], WENB_int[86], WENB_int[85], WENB_int[84], WENB_int[83], WENB_int[82],
          WENB_int[81], WENB_int[80], WENB_int[79], WENB_int[78], WENB_int[77], WENB_int[76],
          WENB_int[75], WENB_int[74], WENB_int[73], WENB_int[72], WENB_int[71], WENB_int[70],
          WENB_int[69], WENB_int[68], WENB_int[67], WENB_int[66], WENB_int[65], WENB_int[64],
          WENB_int[63], WENB_int[62], WENB_int[61], WENB_int[60], WENB_int[59], WENB_int[58],
          WENB_int[57], WENB_int[56], WENB_int[55], WENB_int[54], WENB_int[53], WENB_int[52],
          WENB_int[51], WENB_int[50], WENB_int[49], WENB_int[48], WENB_int[47], WENB_int[46],
          WENB_int[45], WENB_int[44], WENB_int[43], WENB_int[42], WENB_int[41], WENB_int[40],
          WENB_int[39], WENB_int[38], WENB_int[37], WENB_int[36], WENB_int[35], WENB_int[34],
          WENB_int[33], WENB_int[32], WENB_int[31], WENB_int[30], WENB_int[29], WENB_int[28],
          WENB_int[27], WENB_int[26], WENB_int[25], WENB_int[24], WENB_int[23], WENB_int[22],
          WENB_int[21], WENB_int[20], WENB_int[19], WENB_int[18], WENB_int[17], WENB_int[16],
          WENB_int[15], WENB_int[14], WENB_int[13], WENB_int[12], WENB_int[11], WENB_int[10],
          WENB_int[9], WENB_int[8], WENB_int[7], WENB_int[6], WENB_int[5], WENB_int[4],
          WENB_int[3], WENB_int[2], WENB_int[1], WENB_int[0]});
      if (GWENB_int !== 1'b1 || DFTRAMBYP_int === 1'b1) begin
        row_mask =  ( {3'b000, writeEnable[127], 3'b000, writeEnable[126], 3'b000, writeEnable[125],
          3'b000, writeEnable[124], 3'b000, writeEnable[123], 3'b000, writeEnable[122],
          3'b000, writeEnable[121], 3'b000, writeEnable[120], 3'b000, writeEnable[119],
          3'b000, writeEnable[118], 3'b000, writeEnable[117], 3'b000, writeEnable[116],
          3'b000, writeEnable[115], 3'b000, writeEnable[114], 3'b000, writeEnable[113],
          3'b000, writeEnable[112], 3'b000, writeEnable[111], 3'b000, writeEnable[110],
          3'b000, writeEnable[109], 3'b000, writeEnable[108], 3'b000, writeEnable[107],
          3'b000, writeEnable[106], 3'b000, writeEnable[105], 3'b000, writeEnable[104],
          3'b000, writeEnable[103], 3'b000, writeEnable[102], 3'b000, writeEnable[101],
          3'b000, writeEnable[100], 3'b000, writeEnable[99], 3'b000, writeEnable[98],
          3'b000, writeEnable[97], 3'b000, writeEnable[96], 3'b000, writeEnable[95],
          3'b000, writeEnable[94], 3'b000, writeEnable[93], 3'b000, writeEnable[92],
          3'b000, writeEnable[91], 3'b000, writeEnable[90], 3'b000, writeEnable[89],
          3'b000, writeEnable[88], 3'b000, writeEnable[87], 3'b000, writeEnable[86],
          3'b000, writeEnable[85], 3'b000, writeEnable[84], 3'b000, writeEnable[83],
          3'b000, writeEnable[82], 3'b000, writeEnable[81], 3'b000, writeEnable[80],
          3'b000, writeEnable[79], 3'b000, writeEnable[78], 3'b000, writeEnable[77],
          3'b000, writeEnable[76], 3'b000, writeEnable[75], 3'b000, writeEnable[74],
          3'b000, writeEnable[73], 3'b000, writeEnable[72], 3'b000, writeEnable[71],
          3'b000, writeEnable[70], 3'b000, writeEnable[69], 3'b000, writeEnable[68],
          3'b000, writeEnable[67], 3'b000, writeEnable[66], 3'b000, writeEnable[65],
          3'b000, writeEnable[64], 3'b000, writeEnable[63], 3'b000, writeEnable[62],
          3'b000, writeEnable[61], 3'b000, writeEnable[60], 3'b000, writeEnable[59],
          3'b000, writeEnable[58], 3'b000, writeEnable[57], 3'b000, writeEnable[56],
          3'b000, writeEnable[55], 3'b000, writeEnable[54], 3'b000, writeEnable[53],
          3'b000, writeEnable[52], 3'b000, writeEnable[51], 3'b000, writeEnable[50],
          3'b000, writeEnable[49], 3'b000, writeEnable[48], 3'b000, writeEnable[47],
          3'b000, writeEnable[46], 3'b000, writeEnable[45], 3'b000, writeEnable[44],
          3'b000, writeEnable[43], 3'b000, writeEnable[42], 3'b000, writeEnable[41],
          3'b000, writeEnable[40], 3'b000, writeEnable[39], 3'b000, writeEnable[38],
          3'b000, writeEnable[37], 3'b000, writeEnable[36], 3'b000, writeEnable[35],
          3'b000, writeEnable[34], 3'b000, writeEnable[33], 3'b000, writeEnable[32],
          3'b000, writeEnable[31], 3'b000, writeEnable[30], 3'b000, writeEnable[29],
          3'b000, writeEnable[28], 3'b000, writeEnable[27], 3'b000, writeEnable[26],
          3'b000, writeEnable[25], 3'b000, writeEnable[24], 3'b000, writeEnable[23],
          3'b000, writeEnable[22], 3'b000, writeEnable[21], 3'b000, writeEnable[20],
          3'b000, writeEnable[19], 3'b000, writeEnable[18], 3'b000, writeEnable[17],
          3'b000, writeEnable[16], 3'b000, writeEnable[15], 3'b000, writeEnable[14],
          3'b000, writeEnable[13], 3'b000, writeEnable[12], 3'b000, writeEnable[11],
          3'b000, writeEnable[10], 3'b000, writeEnable[9], 3'b000, writeEnable[8],
          3'b000, writeEnable[7], 3'b000, writeEnable[6], 3'b000, writeEnable[5], 3'b000, writeEnable[4],
          3'b000, writeEnable[3], 3'b000, writeEnable[2], 3'b000, writeEnable[1], 3'b000, writeEnable[0]} << mux_address);
        new_data =  ( {3'b000, DB_int[127], 3'b000, DB_int[126], 3'b000, DB_int[125],
          3'b000, DB_int[124], 3'b000, DB_int[123], 3'b000, DB_int[122], 3'b000, DB_int[121],
          3'b000, DB_int[120], 3'b000, DB_int[119], 3'b000, DB_int[118], 3'b000, DB_int[117],
          3'b000, DB_int[116], 3'b000, DB_int[115], 3'b000, DB_int[114], 3'b000, DB_int[113],
          3'b000, DB_int[112], 3'b000, DB_int[111], 3'b000, DB_int[110], 3'b000, DB_int[109],
          3'b000, DB_int[108], 3'b000, DB_int[107], 3'b000, DB_int[106], 3'b000, DB_int[105],
          3'b000, DB_int[104], 3'b000, DB_int[103], 3'b000, DB_int[102], 3'b000, DB_int[101],
          3'b000, DB_int[100], 3'b000, DB_int[99], 3'b000, DB_int[98], 3'b000, DB_int[97],
          3'b000, DB_int[96], 3'b000, DB_int[95], 3'b000, DB_int[94], 3'b000, DB_int[93],
          3'b000, DB_int[92], 3'b000, DB_int[91], 3'b000, DB_int[90], 3'b000, DB_int[89],
          3'b000, DB_int[88], 3'b000, DB_int[87], 3'b000, DB_int[86], 3'b000, DB_int[85],
          3'b000, DB_int[84], 3'b000, DB_int[83], 3'b000, DB_int[82], 3'b000, DB_int[81],
          3'b000, DB_int[80], 3'b000, DB_int[79], 3'b000, DB_int[78], 3'b000, DB_int[77],
          3'b000, DB_int[76], 3'b000, DB_int[75], 3'b000, DB_int[74], 3'b000, DB_int[73],
          3'b000, DB_int[72], 3'b000, DB_int[71], 3'b000, DB_int[70], 3'b000, DB_int[69],
          3'b000, DB_int[68], 3'b000, DB_int[67], 3'b000, DB_int[66], 3'b000, DB_int[65],
          3'b000, DB_int[64], 3'b000, DB_int[63], 3'b000, DB_int[62], 3'b000, DB_int[61],
          3'b000, DB_int[60], 3'b000, DB_int[59], 3'b000, DB_int[58], 3'b000, DB_int[57],
          3'b000, DB_int[56], 3'b000, DB_int[55], 3'b000, DB_int[54], 3'b000, DB_int[53],
          3'b000, DB_int[52], 3'b000, DB_int[51], 3'b000, DB_int[50], 3'b000, DB_int[49],
          3'b000, DB_int[48], 3'b000, DB_int[47], 3'b000, DB_int[46], 3'b000, DB_int[45],
          3'b000, DB_int[44], 3'b000, DB_int[43], 3'b000, DB_int[42], 3'b000, DB_int[41],
          3'b000, DB_int[40], 3'b000, DB_int[39], 3'b000, DB_int[38], 3'b000, DB_int[37],
          3'b000, DB_int[36], 3'b000, DB_int[35], 3'b000, DB_int[34], 3'b000, DB_int[33],
          3'b000, DB_int[32], 3'b000, DB_int[31], 3'b000, DB_int[30], 3'b000, DB_int[29],
          3'b000, DB_int[28], 3'b000, DB_int[27], 3'b000, DB_int[26], 3'b000, DB_int[25],
          3'b000, DB_int[24], 3'b000, DB_int[23], 3'b000, DB_int[22], 3'b000, DB_int[21],
          3'b000, DB_int[20], 3'b000, DB_int[19], 3'b000, DB_int[18], 3'b000, DB_int[17],
          3'b000, DB_int[16], 3'b000, DB_int[15], 3'b000, DB_int[14], 3'b000, DB_int[13],
          3'b000, DB_int[12], 3'b000, DB_int[11], 3'b000, DB_int[10], 3'b000, DB_int[9],
          3'b000, DB_int[8], 3'b000, DB_int[7], 3'b000, DB_int[6], 3'b000, DB_int[5],
          3'b000, DB_int[4], 3'b000, DB_int[3], 3'b000, DB_int[2], 3'b000, DB_int[1],
          3'b000, DB_int[0]} << mux_address);
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
        shifted_readLatch1 = readLatch1;
        mem_path_B = {shifted_readLatch1[127], shifted_readLatch1[126], shifted_readLatch1[125],
          shifted_readLatch1[124], shifted_readLatch1[123], shifted_readLatch1[122],
          shifted_readLatch1[121], shifted_readLatch1[120], shifted_readLatch1[119],
          shifted_readLatch1[118], shifted_readLatch1[117], shifted_readLatch1[116],
          shifted_readLatch1[115], shifted_readLatch1[114], shifted_readLatch1[113],
          shifted_readLatch1[112], shifted_readLatch1[111], shifted_readLatch1[110],
          shifted_readLatch1[109], shifted_readLatch1[108], shifted_readLatch1[107],
          shifted_readLatch1[106], shifted_readLatch1[105], shifted_readLatch1[104],
          shifted_readLatch1[103], shifted_readLatch1[102], shifted_readLatch1[101],
          shifted_readLatch1[100], shifted_readLatch1[99], shifted_readLatch1[98],
          shifted_readLatch1[97], shifted_readLatch1[96], shifted_readLatch1[95], shifted_readLatch1[94],
          shifted_readLatch1[93], shifted_readLatch1[92], shifted_readLatch1[91], shifted_readLatch1[90],
          shifted_readLatch1[89], shifted_readLatch1[88], shifted_readLatch1[87], shifted_readLatch1[86],
          shifted_readLatch1[85], shifted_readLatch1[84], shifted_readLatch1[83], shifted_readLatch1[82],
          shifted_readLatch1[81], shifted_readLatch1[80], shifted_readLatch1[79], shifted_readLatch1[78],
          shifted_readLatch1[77], shifted_readLatch1[76], shifted_readLatch1[75], shifted_readLatch1[74],
          shifted_readLatch1[73], shifted_readLatch1[72], shifted_readLatch1[71], shifted_readLatch1[70],
          shifted_readLatch1[69], shifted_readLatch1[68], shifted_readLatch1[67], shifted_readLatch1[66],
          shifted_readLatch1[65], shifted_readLatch1[64], shifted_readLatch1[63], shifted_readLatch1[62],
          shifted_readLatch1[61], shifted_readLatch1[60], shifted_readLatch1[59], shifted_readLatch1[58],
          shifted_readLatch1[57], shifted_readLatch1[56], shifted_readLatch1[55], shifted_readLatch1[54],
          shifted_readLatch1[53], shifted_readLatch1[52], shifted_readLatch1[51], shifted_readLatch1[50],
          shifted_readLatch1[49], shifted_readLatch1[48], shifted_readLatch1[47], shifted_readLatch1[46],
          shifted_readLatch1[45], shifted_readLatch1[44], shifted_readLatch1[43], shifted_readLatch1[42],
          shifted_readLatch1[41], shifted_readLatch1[40], shifted_readLatch1[39], shifted_readLatch1[38],
          shifted_readLatch1[37], shifted_readLatch1[36], shifted_readLatch1[35], shifted_readLatch1[34],
          shifted_readLatch1[33], shifted_readLatch1[32], shifted_readLatch1[31], shifted_readLatch1[30],
          shifted_readLatch1[29], shifted_readLatch1[28], shifted_readLatch1[27], shifted_readLatch1[26],
          shifted_readLatch1[25], shifted_readLatch1[24], shifted_readLatch1[23], shifted_readLatch1[22],
          shifted_readLatch1[21], shifted_readLatch1[20], shifted_readLatch1[19], shifted_readLatch1[18],
          shifted_readLatch1[17], shifted_readLatch1[16], shifted_readLatch1[15], shifted_readLatch1[14],
          shifted_readLatch1[13], shifted_readLatch1[12], shifted_readLatch1[11], shifted_readLatch1[10],
          shifted_readLatch1[9], shifted_readLatch1[8], shifted_readLatch1[7], shifted_readLatch1[6],
          shifted_readLatch1[5], shifted_readLatch1[4], shifted_readLatch1[3], shifted_readLatch1[2],
          shifted_readLatch1[1], shifted_readLatch1[0]};
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
      end
      if (DFTRAMBYP_=== 1'b1 && SEB_ === 1'b1) begin
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
      end
    readWriteB;
      end
    #0;
`ifdef NO_COLLISIONS
`else     
      if ((1) && (CENA_int !== 1'b1 && CENB_int !== 1'b1 && DFTRAMBYP_ !== 1'b1) && COLLDISN_int === 1'b1 && row_contention(AA_int, AB_int, ({128{GWENA_int}}|WENA_int),
        ({128{GWENB_int}}|WENB_int))) begin
        if (col_contention(AA_int, AB_int)) begin
          COL_CC = 1;
        end
          ROW_CC = 1;
          READ_READ_1 = 0;
          READ_WRITE_1 = 0;
          WRITE_WRITE_1 = 0;
        if (GWENA_int !== 1'b1 && (& WENA_int) !== 1'b1 && GWENB_int !== 1'b1 && (& WENB_int) !== 1'b1) begin
	      if (is_contention(AA_int, AB_int, ({128{GWENA_int}}|WENA_int), ({128{GWENB_int}}|WENB_int))) begin
          COL_CC = 1;
          WRITE_WRITE = 1;
	      end
        end else if (GWENA_int !== 1'b1 && (& WENA_int) !== 1'b1) begin
		if (is_contention(AA_int, AB_int, ({128{GWENA_int}}|WENA_int), ({128{GWENB_int}}|WENB_int))) begin
          COL_CC = 1;
          READ_WRITE = 1;
		end
        end else if (GWENB_int !== 1'b1 && (& WENB_int) !== 1'b1) begin
		if (is_contention(AA_int, AB_int, ({128{GWENA_int}}|WENA_int), ({128{GWENB_int}}|WENB_int))) begin
          COL_CC = 1;
          READ_WRITE = 1;
		end
        end else begin
          COL_CC = 1;
          READ_READ = 1;
        end
        if (!is_contention(AA_int, AB_int, ({128{GWENA_int}}|WENA_int), ({128{GWENB_int}}|WENB_int))) begin
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
        AB_int, ({128{GWENA_int}}|WENA_int), ({128{GWENB_int}}|WENB_int))) begin
          ROW_CC = 1;
          READ_READ_1 = 0;
          READ_WRITE_1 = 0;
          WRITE_WRITE_1 = 0;
        if (col_contention(AA_int, AB_int)) begin
          COL_CC = 1;
        end
        if (GWENA_int !== 1'b1 && (& WENA_int) !== 1'b1) begin
          WRITE_WRITE_1 = 1;
        end else if (is_contention(AA_int, AB_int, ({128{GWENA_int}}|WENA_int), ({128{GWENB_int}}|WENB_int))) begin
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
        end else if (is_contention(AA_int, AB_int, ({128{GWENA_int}}|WENA_int), ({128{GWENB_int}}|WENB_int))) begin
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
  end

  assign SIB_int = SEB_ ? SIB_ : {2{1'b0}};
  assign DB_int_bmux = TENB_ ? DB_ : TDB_;

  datapath_latch_sram_128_1024_dp uDQB0 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(SIB_int[0]), .D(DB_int_bmux[0]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[0]), .XQ(XQB), .Q(QB_int[0]));
  datapath_latch_sram_128_1024_dp uDQB1 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[0]), .D(DB_int_bmux[1]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[1]), .XQ(XQB), .Q(QB_int[1]));
  datapath_latch_sram_128_1024_dp uDQB2 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[1]), .D(DB_int_bmux[2]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[2]), .XQ(XQB), .Q(QB_int[2]));
  datapath_latch_sram_128_1024_dp uDQB3 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[2]), .D(DB_int_bmux[3]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[3]), .XQ(XQB), .Q(QB_int[3]));
  datapath_latch_sram_128_1024_dp uDQB4 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[3]), .D(DB_int_bmux[4]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[4]), .XQ(XQB), .Q(QB_int[4]));
  datapath_latch_sram_128_1024_dp uDQB5 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[4]), .D(DB_int_bmux[5]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[5]), .XQ(XQB), .Q(QB_int[5]));
  datapath_latch_sram_128_1024_dp uDQB6 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[5]), .D(DB_int_bmux[6]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[6]), .XQ(XQB), .Q(QB_int[6]));
  datapath_latch_sram_128_1024_dp uDQB7 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[6]), .D(DB_int_bmux[7]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[7]), .XQ(XQB), .Q(QB_int[7]));
  datapath_latch_sram_128_1024_dp uDQB8 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[7]), .D(DB_int_bmux[8]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[8]), .XQ(XQB), .Q(QB_int[8]));
  datapath_latch_sram_128_1024_dp uDQB9 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[8]), .D(DB_int_bmux[9]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[9]), .XQ(XQB), .Q(QB_int[9]));
  datapath_latch_sram_128_1024_dp uDQB10 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[9]), .D(DB_int_bmux[10]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[10]), .XQ(XQB), .Q(QB_int[10]));
  datapath_latch_sram_128_1024_dp uDQB11 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[10]), .D(DB_int_bmux[11]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[11]), .XQ(XQB), .Q(QB_int[11]));
  datapath_latch_sram_128_1024_dp uDQB12 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[11]), .D(DB_int_bmux[12]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[12]), .XQ(XQB), .Q(QB_int[12]));
  datapath_latch_sram_128_1024_dp uDQB13 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[12]), .D(DB_int_bmux[13]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[13]), .XQ(XQB), .Q(QB_int[13]));
  datapath_latch_sram_128_1024_dp uDQB14 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[13]), .D(DB_int_bmux[14]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[14]), .XQ(XQB), .Q(QB_int[14]));
  datapath_latch_sram_128_1024_dp uDQB15 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[14]), .D(DB_int_bmux[15]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[15]), .XQ(XQB), .Q(QB_int[15]));
  datapath_latch_sram_128_1024_dp uDQB16 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[15]), .D(DB_int_bmux[16]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[16]), .XQ(XQB), .Q(QB_int[16]));
  datapath_latch_sram_128_1024_dp uDQB17 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[16]), .D(DB_int_bmux[17]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[17]), .XQ(XQB), .Q(QB_int[17]));
  datapath_latch_sram_128_1024_dp uDQB18 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[17]), .D(DB_int_bmux[18]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[18]), .XQ(XQB), .Q(QB_int[18]));
  datapath_latch_sram_128_1024_dp uDQB19 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[18]), .D(DB_int_bmux[19]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[19]), .XQ(XQB), .Q(QB_int[19]));
  datapath_latch_sram_128_1024_dp uDQB20 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[19]), .D(DB_int_bmux[20]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[20]), .XQ(XQB), .Q(QB_int[20]));
  datapath_latch_sram_128_1024_dp uDQB21 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[20]), .D(DB_int_bmux[21]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[21]), .XQ(XQB), .Q(QB_int[21]));
  datapath_latch_sram_128_1024_dp uDQB22 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[21]), .D(DB_int_bmux[22]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[22]), .XQ(XQB), .Q(QB_int[22]));
  datapath_latch_sram_128_1024_dp uDQB23 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[22]), .D(DB_int_bmux[23]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[23]), .XQ(XQB), .Q(QB_int[23]));
  datapath_latch_sram_128_1024_dp uDQB24 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[23]), .D(DB_int_bmux[24]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[24]), .XQ(XQB), .Q(QB_int[24]));
  datapath_latch_sram_128_1024_dp uDQB25 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[24]), .D(DB_int_bmux[25]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[25]), .XQ(XQB), .Q(QB_int[25]));
  datapath_latch_sram_128_1024_dp uDQB26 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[25]), .D(DB_int_bmux[26]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[26]), .XQ(XQB), .Q(QB_int[26]));
  datapath_latch_sram_128_1024_dp uDQB27 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[26]), .D(DB_int_bmux[27]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[27]), .XQ(XQB), .Q(QB_int[27]));
  datapath_latch_sram_128_1024_dp uDQB28 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[27]), .D(DB_int_bmux[28]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[28]), .XQ(XQB), .Q(QB_int[28]));
  datapath_latch_sram_128_1024_dp uDQB29 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[28]), .D(DB_int_bmux[29]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[29]), .XQ(XQB), .Q(QB_int[29]));
  datapath_latch_sram_128_1024_dp uDQB30 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[29]), .D(DB_int_bmux[30]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[30]), .XQ(XQB), .Q(QB_int[30]));
  datapath_latch_sram_128_1024_dp uDQB31 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[30]), .D(DB_int_bmux[31]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[31]), .XQ(XQB), .Q(QB_int[31]));
  datapath_latch_sram_128_1024_dp uDQB32 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[31]), .D(DB_int_bmux[32]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[32]), .XQ(XQB), .Q(QB_int[32]));
  datapath_latch_sram_128_1024_dp uDQB33 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[32]), .D(DB_int_bmux[33]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[33]), .XQ(XQB), .Q(QB_int[33]));
  datapath_latch_sram_128_1024_dp uDQB34 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[33]), .D(DB_int_bmux[34]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[34]), .XQ(XQB), .Q(QB_int[34]));
  datapath_latch_sram_128_1024_dp uDQB35 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[34]), .D(DB_int_bmux[35]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[35]), .XQ(XQB), .Q(QB_int[35]));
  datapath_latch_sram_128_1024_dp uDQB36 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[35]), .D(DB_int_bmux[36]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[36]), .XQ(XQB), .Q(QB_int[36]));
  datapath_latch_sram_128_1024_dp uDQB37 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[36]), .D(DB_int_bmux[37]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[37]), .XQ(XQB), .Q(QB_int[37]));
  datapath_latch_sram_128_1024_dp uDQB38 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[37]), .D(DB_int_bmux[38]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[38]), .XQ(XQB), .Q(QB_int[38]));
  datapath_latch_sram_128_1024_dp uDQB39 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[38]), .D(DB_int_bmux[39]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[39]), .XQ(XQB), .Q(QB_int[39]));
  datapath_latch_sram_128_1024_dp uDQB40 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[39]), .D(DB_int_bmux[40]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[40]), .XQ(XQB), .Q(QB_int[40]));
  datapath_latch_sram_128_1024_dp uDQB41 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[40]), .D(DB_int_bmux[41]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[41]), .XQ(XQB), .Q(QB_int[41]));
  datapath_latch_sram_128_1024_dp uDQB42 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[41]), .D(DB_int_bmux[42]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[42]), .XQ(XQB), .Q(QB_int[42]));
  datapath_latch_sram_128_1024_dp uDQB43 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[42]), .D(DB_int_bmux[43]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[43]), .XQ(XQB), .Q(QB_int[43]));
  datapath_latch_sram_128_1024_dp uDQB44 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[43]), .D(DB_int_bmux[44]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[44]), .XQ(XQB), .Q(QB_int[44]));
  datapath_latch_sram_128_1024_dp uDQB45 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[44]), .D(DB_int_bmux[45]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[45]), .XQ(XQB), .Q(QB_int[45]));
  datapath_latch_sram_128_1024_dp uDQB46 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[45]), .D(DB_int_bmux[46]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[46]), .XQ(XQB), .Q(QB_int[46]));
  datapath_latch_sram_128_1024_dp uDQB47 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[46]), .D(DB_int_bmux[47]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[47]), .XQ(XQB), .Q(QB_int[47]));
  datapath_latch_sram_128_1024_dp uDQB48 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[47]), .D(DB_int_bmux[48]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[48]), .XQ(XQB), .Q(QB_int[48]));
  datapath_latch_sram_128_1024_dp uDQB49 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[48]), .D(DB_int_bmux[49]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[49]), .XQ(XQB), .Q(QB_int[49]));
  datapath_latch_sram_128_1024_dp uDQB50 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[49]), .D(DB_int_bmux[50]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[50]), .XQ(XQB), .Q(QB_int[50]));
  datapath_latch_sram_128_1024_dp uDQB51 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[50]), .D(DB_int_bmux[51]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[51]), .XQ(XQB), .Q(QB_int[51]));
  datapath_latch_sram_128_1024_dp uDQB52 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[51]), .D(DB_int_bmux[52]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[52]), .XQ(XQB), .Q(QB_int[52]));
  datapath_latch_sram_128_1024_dp uDQB53 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[52]), .D(DB_int_bmux[53]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[53]), .XQ(XQB), .Q(QB_int[53]));
  datapath_latch_sram_128_1024_dp uDQB54 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[53]), .D(DB_int_bmux[54]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[54]), .XQ(XQB), .Q(QB_int[54]));
  datapath_latch_sram_128_1024_dp uDQB55 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[54]), .D(DB_int_bmux[55]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[55]), .XQ(XQB), .Q(QB_int[55]));
  datapath_latch_sram_128_1024_dp uDQB56 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[55]), .D(DB_int_bmux[56]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[56]), .XQ(XQB), .Q(QB_int[56]));
  datapath_latch_sram_128_1024_dp uDQB57 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[56]), .D(DB_int_bmux[57]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[57]), .XQ(XQB), .Q(QB_int[57]));
  datapath_latch_sram_128_1024_dp uDQB58 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[57]), .D(DB_int_bmux[58]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[58]), .XQ(XQB), .Q(QB_int[58]));
  datapath_latch_sram_128_1024_dp uDQB59 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[58]), .D(DB_int_bmux[59]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[59]), .XQ(XQB), .Q(QB_int[59]));
  datapath_latch_sram_128_1024_dp uDQB60 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[59]), .D(DB_int_bmux[60]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[60]), .XQ(XQB), .Q(QB_int[60]));
  datapath_latch_sram_128_1024_dp uDQB61 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[60]), .D(DB_int_bmux[61]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[61]), .XQ(XQB), .Q(QB_int[61]));
  datapath_latch_sram_128_1024_dp uDQB62 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[61]), .D(DB_int_bmux[62]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[62]), .XQ(XQB), .Q(QB_int[62]));
  datapath_latch_sram_128_1024_dp uDQB63 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[62]), .D(DB_int_bmux[63]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[63]), .XQ(XQB), .Q(QB_int[63]));
  datapath_latch_sram_128_1024_dp uDQB64 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[65]), .D(DB_int_bmux[64]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[64]), .XQ(XQB), .Q(QB_int[64]));
  datapath_latch_sram_128_1024_dp uDQB65 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[66]), .D(DB_int_bmux[65]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[65]), .XQ(XQB), .Q(QB_int[65]));
  datapath_latch_sram_128_1024_dp uDQB66 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[67]), .D(DB_int_bmux[66]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[66]), .XQ(XQB), .Q(QB_int[66]));
  datapath_latch_sram_128_1024_dp uDQB67 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[68]), .D(DB_int_bmux[67]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[67]), .XQ(XQB), .Q(QB_int[67]));
  datapath_latch_sram_128_1024_dp uDQB68 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[69]), .D(DB_int_bmux[68]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[68]), .XQ(XQB), .Q(QB_int[68]));
  datapath_latch_sram_128_1024_dp uDQB69 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[70]), .D(DB_int_bmux[69]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[69]), .XQ(XQB), .Q(QB_int[69]));
  datapath_latch_sram_128_1024_dp uDQB70 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[71]), .D(DB_int_bmux[70]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[70]), .XQ(XQB), .Q(QB_int[70]));
  datapath_latch_sram_128_1024_dp uDQB71 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[72]), .D(DB_int_bmux[71]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[71]), .XQ(XQB), .Q(QB_int[71]));
  datapath_latch_sram_128_1024_dp uDQB72 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[73]), .D(DB_int_bmux[72]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[72]), .XQ(XQB), .Q(QB_int[72]));
  datapath_latch_sram_128_1024_dp uDQB73 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[74]), .D(DB_int_bmux[73]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[73]), .XQ(XQB), .Q(QB_int[73]));
  datapath_latch_sram_128_1024_dp uDQB74 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[75]), .D(DB_int_bmux[74]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[74]), .XQ(XQB), .Q(QB_int[74]));
  datapath_latch_sram_128_1024_dp uDQB75 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[76]), .D(DB_int_bmux[75]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[75]), .XQ(XQB), .Q(QB_int[75]));
  datapath_latch_sram_128_1024_dp uDQB76 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[77]), .D(DB_int_bmux[76]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[76]), .XQ(XQB), .Q(QB_int[76]));
  datapath_latch_sram_128_1024_dp uDQB77 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[78]), .D(DB_int_bmux[77]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[77]), .XQ(XQB), .Q(QB_int[77]));
  datapath_latch_sram_128_1024_dp uDQB78 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[79]), .D(DB_int_bmux[78]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[78]), .XQ(XQB), .Q(QB_int[78]));
  datapath_latch_sram_128_1024_dp uDQB79 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[80]), .D(DB_int_bmux[79]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[79]), .XQ(XQB), .Q(QB_int[79]));
  datapath_latch_sram_128_1024_dp uDQB80 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[81]), .D(DB_int_bmux[80]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[80]), .XQ(XQB), .Q(QB_int[80]));
  datapath_latch_sram_128_1024_dp uDQB81 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[82]), .D(DB_int_bmux[81]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[81]), .XQ(XQB), .Q(QB_int[81]));
  datapath_latch_sram_128_1024_dp uDQB82 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[83]), .D(DB_int_bmux[82]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[82]), .XQ(XQB), .Q(QB_int[82]));
  datapath_latch_sram_128_1024_dp uDQB83 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[84]), .D(DB_int_bmux[83]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[83]), .XQ(XQB), .Q(QB_int[83]));
  datapath_latch_sram_128_1024_dp uDQB84 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[85]), .D(DB_int_bmux[84]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[84]), .XQ(XQB), .Q(QB_int[84]));
  datapath_latch_sram_128_1024_dp uDQB85 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[86]), .D(DB_int_bmux[85]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[85]), .XQ(XQB), .Q(QB_int[85]));
  datapath_latch_sram_128_1024_dp uDQB86 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[87]), .D(DB_int_bmux[86]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[86]), .XQ(XQB), .Q(QB_int[86]));
  datapath_latch_sram_128_1024_dp uDQB87 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[88]), .D(DB_int_bmux[87]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[87]), .XQ(XQB), .Q(QB_int[87]));
  datapath_latch_sram_128_1024_dp uDQB88 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[89]), .D(DB_int_bmux[88]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[88]), .XQ(XQB), .Q(QB_int[88]));
  datapath_latch_sram_128_1024_dp uDQB89 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[90]), .D(DB_int_bmux[89]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[89]), .XQ(XQB), .Q(QB_int[89]));
  datapath_latch_sram_128_1024_dp uDQB90 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[91]), .D(DB_int_bmux[90]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[90]), .XQ(XQB), .Q(QB_int[90]));
  datapath_latch_sram_128_1024_dp uDQB91 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[92]), .D(DB_int_bmux[91]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[91]), .XQ(XQB), .Q(QB_int[91]));
  datapath_latch_sram_128_1024_dp uDQB92 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[93]), .D(DB_int_bmux[92]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[92]), .XQ(XQB), .Q(QB_int[92]));
  datapath_latch_sram_128_1024_dp uDQB93 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[94]), .D(DB_int_bmux[93]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[93]), .XQ(XQB), .Q(QB_int[93]));
  datapath_latch_sram_128_1024_dp uDQB94 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[95]), .D(DB_int_bmux[94]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[94]), .XQ(XQB), .Q(QB_int[94]));
  datapath_latch_sram_128_1024_dp uDQB95 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[96]), .D(DB_int_bmux[95]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[95]), .XQ(XQB), .Q(QB_int[95]));
  datapath_latch_sram_128_1024_dp uDQB96 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[97]), .D(DB_int_bmux[96]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[96]), .XQ(XQB), .Q(QB_int[96]));
  datapath_latch_sram_128_1024_dp uDQB97 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[98]), .D(DB_int_bmux[97]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[97]), .XQ(XQB), .Q(QB_int[97]));
  datapath_latch_sram_128_1024_dp uDQB98 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[99]), .D(DB_int_bmux[98]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[98]), .XQ(XQB), .Q(QB_int[98]));
  datapath_latch_sram_128_1024_dp uDQB99 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[100]), .D(DB_int_bmux[99]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[99]), .XQ(XQB), .Q(QB_int[99]));
  datapath_latch_sram_128_1024_dp uDQB100 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[101]), .D(DB_int_bmux[100]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[100]), .XQ(XQB), .Q(QB_int[100]));
  datapath_latch_sram_128_1024_dp uDQB101 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[102]), .D(DB_int_bmux[101]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[101]), .XQ(XQB), .Q(QB_int[101]));
  datapath_latch_sram_128_1024_dp uDQB102 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[103]), .D(DB_int_bmux[102]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[102]), .XQ(XQB), .Q(QB_int[102]));
  datapath_latch_sram_128_1024_dp uDQB103 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[104]), .D(DB_int_bmux[103]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[103]), .XQ(XQB), .Q(QB_int[103]));
  datapath_latch_sram_128_1024_dp uDQB104 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[105]), .D(DB_int_bmux[104]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[104]), .XQ(XQB), .Q(QB_int[104]));
  datapath_latch_sram_128_1024_dp uDQB105 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[106]), .D(DB_int_bmux[105]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[105]), .XQ(XQB), .Q(QB_int[105]));
  datapath_latch_sram_128_1024_dp uDQB106 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[107]), .D(DB_int_bmux[106]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[106]), .XQ(XQB), .Q(QB_int[106]));
  datapath_latch_sram_128_1024_dp uDQB107 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[108]), .D(DB_int_bmux[107]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[107]), .XQ(XQB), .Q(QB_int[107]));
  datapath_latch_sram_128_1024_dp uDQB108 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[109]), .D(DB_int_bmux[108]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[108]), .XQ(XQB), .Q(QB_int[108]));
  datapath_latch_sram_128_1024_dp uDQB109 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[110]), .D(DB_int_bmux[109]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[109]), .XQ(XQB), .Q(QB_int[109]));
  datapath_latch_sram_128_1024_dp uDQB110 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[111]), .D(DB_int_bmux[110]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[110]), .XQ(XQB), .Q(QB_int[110]));
  datapath_latch_sram_128_1024_dp uDQB111 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[112]), .D(DB_int_bmux[111]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[111]), .XQ(XQB), .Q(QB_int[111]));
  datapath_latch_sram_128_1024_dp uDQB112 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[113]), .D(DB_int_bmux[112]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[112]), .XQ(XQB), .Q(QB_int[112]));
  datapath_latch_sram_128_1024_dp uDQB113 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[114]), .D(DB_int_bmux[113]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[113]), .XQ(XQB), .Q(QB_int[113]));
  datapath_latch_sram_128_1024_dp uDQB114 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[115]), .D(DB_int_bmux[114]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[114]), .XQ(XQB), .Q(QB_int[114]));
  datapath_latch_sram_128_1024_dp uDQB115 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[116]), .D(DB_int_bmux[115]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[115]), .XQ(XQB), .Q(QB_int[115]));
  datapath_latch_sram_128_1024_dp uDQB116 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[117]), .D(DB_int_bmux[116]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[116]), .XQ(XQB), .Q(QB_int[116]));
  datapath_latch_sram_128_1024_dp uDQB117 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[118]), .D(DB_int_bmux[117]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[117]), .XQ(XQB), .Q(QB_int[117]));
  datapath_latch_sram_128_1024_dp uDQB118 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[119]), .D(DB_int_bmux[118]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[118]), .XQ(XQB), .Q(QB_int[118]));
  datapath_latch_sram_128_1024_dp uDQB119 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[120]), .D(DB_int_bmux[119]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[119]), .XQ(XQB), .Q(QB_int[119]));
  datapath_latch_sram_128_1024_dp uDQB120 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[121]), .D(DB_int_bmux[120]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[120]), .XQ(XQB), .Q(QB_int[120]));
  datapath_latch_sram_128_1024_dp uDQB121 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[122]), .D(DB_int_bmux[121]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[121]), .XQ(XQB), .Q(QB_int[121]));
  datapath_latch_sram_128_1024_dp uDQB122 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[123]), .D(DB_int_bmux[122]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[122]), .XQ(XQB), .Q(QB_int[122]));
  datapath_latch_sram_128_1024_dp uDQB123 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[124]), .D(DB_int_bmux[123]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[123]), .XQ(XQB), .Q(QB_int[123]));
  datapath_latch_sram_128_1024_dp uDQB124 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[125]), .D(DB_int_bmux[124]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[124]), .XQ(XQB), .Q(QB_int[124]));
  datapath_latch_sram_128_1024_dp uDQB125 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[126]), .D(DB_int_bmux[125]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[125]), .XQ(XQB), .Q(QB_int[125]));
  datapath_latch_sram_128_1024_dp uDQB126 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(QB_int[127]), .D(DB_int_bmux[126]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[126]), .XQ(XQB), .Q(QB_int[126]));
  datapath_latch_sram_128_1024_dp uDQB127 (.CLK(CLKB), .Q_update(QB_update), .D_update(DB_sh_update), .SE(SEB_), .SI(SIB_int[1]), .D(DB_int_bmux[127]), .DFTRAMBYP(DFTRAMBYP_), .mem_path(mem_path_B[127]), .XQ(XQB), .Q(QB_int[127]));



  function row_contention;
    input [9:0] aa;
    input [9:0] ab;
    input [127:0] wena;
    input [127:0] wenb;
    reg result;
    reg sameRow;
    reg sameMux;
    reg anyWrite;
  begin
    anyWrite = ((& wena) === 1'b1 && (& wenb) === 1'b1) ? 1'b0 : 1'b1;
    sameMux = (aa[1:0] == ab[1:0]) ? 1'b1 : 1'b0;
    if (aa[9:2] == ab[9:2]) begin
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
    if (aa[1:0] == ab[1:0])
      col_contention = 1'b1;
    else
      col_contention = 1'b0;
  end
  endfunction

  function is_contention;
    input [9:0] aa;
    input [9:0] ab;
    input [127:0] wena;
    input [127:0] wenb;
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
