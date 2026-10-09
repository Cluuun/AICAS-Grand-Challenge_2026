module add_sub #(
  parameter int unsigned K = 32,
  parameter int unsigned P = 24,
  parameter int unsigned E = 8
) (
  input  logic         clk_i,
  input  logic         rst_ni,
  input  logic [K-1:0] FP_A,
  input  logic [K-1:0] FP_B,
  input  logic         add_sub,
  output logic [K-1:0] FP_Z
);

  localparam int unsigned PLOG = $clog2(P+3) - 1;
  localparam logic [K-1:0] ZEROS = '0;
  localparam logic [K-1:0] ONES  = '1;

  logic [K-1:0] A_int;
  logic [K-1:0] B_int;
  logic expA_FF, expB_FF, expA_Z, expB_Z;
  logic fracA_Z, fracB_Z;

  logic isNaN_A, isNaN_B, isInf_A, isInf_B, isZero_A, isZero_B, isNaN, isInf;
  logic underflow_sub;

  logic sign_A, sign_B;
  logic [E-1:0] exp_A, exp_B;
  logic [E:0] efectExp;

  logic [P+3:0] efectFracA, efectFracB, efectFracB_align;
  logic [E:0] diffExpAB, diffExpBA, diffExp;
  logic [P+3:0] addAB, addSubAB, subAB;
  logic [P+3:0] frac_add_Norm1;
  logic isSUB;

  logic [P+3:0] subBAExpEq;
  logic [P+3:0] frac_sub_Norm1;
  logic sign;

  logic isZero_AorB;

  logic [P+3:0] frac, frac_Norm1;
  logic [P-2:0] frac_Norm2;
  logic [P+3:0] frac_stg2;
  logic [E-1:0] exp_Norm1, efectExp_stg2;
  logic sign_stg2, isSUB_stg2, isNaN_stg2, isInf_stg2, overflow, underflow;
  logic isZero_AorB_stg2;
  logic isTwo;
  logic [E-1:0] exp_add_Norm1, exp_sub_Norm1;
  logic isRoundUp, didNorm1;
  logic [K-1:0] FP_Z_int;
  logic [K-1:0] FP_Z_q;
  logic [E-1:0] exp_add_Norm1_aux;
  logic [P-2:0] frac_Norm2_aux;

  assign A_int = FP_A;
  assign B_int = FP_B;

  assign expA_FF = (A_int[K-2:K-E-1] == ONES[K-2:K-E-1]);
  assign expB_FF = (B_int[K-2:K-E-1] == ONES[K-2:K-E-1]);
  assign expA_Z  = (A_int[K-2:K-E-1] == ZEROS[K-2:K-E-1]);
  assign expB_Z  = (B_int[K-2:K-E-1] == ZEROS[K-2:K-E-1]);
  assign fracA_Z = (A_int[P-2:0] == ZEROS[P-2:0]);
  assign fracB_Z = (B_int[P-2:0] == ZEROS[P-2:0]);

  assign isNaN_A = expA_FF & (~fracA_Z);
  assign isNaN_B = expB_FF & (~fracB_Z);
  assign isInf_A = expA_FF;
  assign isInf_B = expB_FF;
  assign isZero_A = expA_Z & fracA_Z;
  assign isZero_B = expB_Z & fracB_Z;
  assign isZero_AorB = isZero_A | isZero_B;

  assign isNaN = (isNaN_A | isNaN_B) | (isInf_A & isInf_B & isSUB);
  assign isInf = (isInf_A ^ isInf_B) | (isInf_A & isInf_B & (~isSUB));

  assign sign_A = A_int[K-1];
  assign exp_A  = A_int[K-2:K-E-1];
  assign sign_B = add_sub ? B_int[K-1] : ~B_int[K-1];
  assign exp_B  = B_int[K-2:K-E-1];

  assign isSUB = sign_A ^ sign_B;

  assign diffExpAB = {1'b0, exp_A} - {1'b0, exp_B};
  assign diffExpBA = {1'b0, exp_B} - {1'b0, exp_A};
  assign diffExp   = diffExpAB[E] ? diffExpBA : diffExpAB;

  assign efectFracA = diffExpAB[E] ? {2'b01, B_int[P-2:0], 3'b000} : {2'b01, A_int[P-2:0], 3'b000};
  assign efectFracB = diffExpAB[E] ? {2'b01, A_int[P-2:0], 3'b000} : {2'b01, B_int[P-2:0], 3'b000};
  assign efectExp   = diffExpAB[E] ? {1'b0, exp_B} : {1'b0, exp_A};

  right_shifter #(
    .P    (P+3),
    .E    (E),
    .PLOG (PLOG)
  ) unioa (
    .frac      (efectFracB),
    .diff_exp  (diffExp),
    .frac_align(efectFracB_align)
  );

  assign addAB    = efectFracA + efectFracB_align;
  assign subAB    = efectFracA - efectFracB_align;
  assign addSubAB = (isSUB == 1'b0) ? addAB : subAB;

  assign subBAExpEq = {({2'b01, B_int[P-2:0]} - {2'b01, A_int[P-2:0]}), 3'b000};

  assign frac =
      isZero_A ? {2'b01, B_int[P-2:0], 3'b000} :
      isZero_B ? {2'b01, A_int[P-2:0], 3'b000} :
      ((!subBAExpEq[P+3]) && (exp_A == exp_B) && (isSUB == 1'b1)) ? subBAExpEq :
      addSubAB;

  assign sign =
      (sign_A == sign_B) ? sign_A :
      diffExpAB[E] ? sign_B :
      (!addSubAB[P+3]) ? sign_A :
      sign_B;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      frac_stg2        <= '0;
      efectExp_stg2    <= '0;
      isZero_AorB_stg2 <= 1'b0;
      isNaN_stg2       <= 1'b0;
      isInf_stg2       <= 1'b0;
      sign_stg2        <= 1'b0;
      isSUB_stg2       <= 1'b0;
    end else begin
      frac_stg2        <= frac;
      efectExp_stg2    <= efectExp[E-1:0];
      isZero_AorB_stg2 <= isZero_AorB;
      isNaN_stg2       <= isNaN;
      isInf_stg2       <= isInf;
      sign_stg2        <= sign;
      isSUB_stg2       <= isSUB;
    end
  end

  always_comb begin
    if (frac_stg2[P+3]) begin
      frac_add_Norm1 = {1'b0, frac_stg2[P+3:2], (frac_stg2[1] | frac_stg2[0])};
      didNorm1       = 1'b1;
    end else begin
      frac_add_Norm1 = frac_stg2;
      didNorm1       = 1'b0;
    end
  end

  assign isTwo            = (frac_stg2[P+2:2] == {P+1{1'b1}});
  assign exp_add_Norm1_aux = {{(E-1){1'b0}}, (didNorm1 | isTwo)};
  assign exp_add_Norm1    = efectExp_stg2 + exp_add_Norm1_aux;

  fp_leading_zeros_and_shift #(
    .P    (P+3),
    .E    (E),
    .PLOG (PLOG)
  ) subtraction_norm (
    .frac      (frac_stg2),
    .exp       (efectExp_stg2),
    .frac_Norm (frac_sub_Norm1),
    .exp_Norm  (exp_sub_Norm1),
    .underFlow (underflow_sub)
  );

  assign frac_Norm1 = (isSUB_stg2 == 1'b0) ? frac_add_Norm1 : frac_sub_Norm1;
  assign exp_Norm1  = (isSUB_stg2 == 1'b0) ? exp_add_Norm1[E-1:0] : exp_sub_Norm1[E-1:0];

  assign isRoundUp    = ((frac_Norm1[2] && (frac_Norm1[1] || frac_Norm1[0])) || (frac_Norm1[3:0] == 4'b1100));
  assign frac_Norm2_aux = {{(P-2){1'b0}}, isRoundUp};
  assign frac_Norm2   = frac_Norm1[P+1:3] + frac_Norm2_aux;

  assign overflow  = (exp_Norm1 == ONES[E-1:0]) && (isSUB_stg2 == 1'b0);
  assign underflow = (isSUB_stg2 == 1'b1) ? underflow_sub : 1'b0;

  assign FP_Z_int =
      isNaN_stg2 ? {sign_stg2, {E{1'b1}}, {(P-2){1'b0}}, 1'b1} :
      (isInf_stg2 || overflow) ? {sign_stg2, {E{1'b1}}, {(P-1){1'b0}}} :
      isZero_AorB_stg2 ? {sign_stg2, efectExp_stg2[E-1:0], frac_stg2[P+1:3]} :
      underflow ? {sign_stg2, {(K-1){1'b0}}} :
      {sign_stg2, exp_Norm1, frac_Norm2};

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni)
      FP_Z_q <= '0;
    else
      FP_Z_q <= FP_Z_int;
  end

  assign FP_Z = FP_Z_q;

endmodule
