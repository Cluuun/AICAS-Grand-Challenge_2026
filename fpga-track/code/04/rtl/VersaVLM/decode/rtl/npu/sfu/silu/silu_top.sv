`ifndef SILU_TOP_SV
`define SILU_TOP_SV

module silu_top #(
    parameter int USER_WIDTH = 16
) (
    input  logic                  clk,
    input  logic                  rst_n,
    input  logic                  enable_i,
    input  logic                  valid_i,
    input  logic [15:0]           fp16_i,
    input  logic [USER_WIDTH-1:0] user_i,
    output logic                  valid_o,
    output logic [15:0]           fp16_o,
    output logic [USER_WIDTH-1:0] user_o
);

    logic signed [15:0] x_q8_8_w;
    logic [11:0] offset_w;
    logic [7:0] lut_idx_w;
    logic [3:0] frac_w;
    logic signed [31:0] y0_q16_w;
    logic signed [31:0] y1_q16_w;
    logic signed [31:0] delta_q16_w;
    logic signed [31:0] interp_delta_q16_w;
    logic signed [31:0] interp_q16_w;
    logic signed [36:0] interp_prod_w;
    logic [15:0] fp16_conv_w;
    logic tail_neg_w;
    logic tail_pos_w;

    logic                  valid_in_q;
    logic [15:0]           fp16_in_q;
    logic [USER_WIDTH-1:0] user_in_q;
    logic                  valid_s0_q;
    logic                  valid_s1_q;
    logic                  valid_s2_q;
    logic                  valid_s3_q;
    logic                  valid_s4_q;
    logic                  lut_valid_q;
    logic [15:0]           lut_fp16_q;
    logic [USER_WIDTH-1:0] lut_user_q;
    logic [USER_WIDTH-1:0] user_s0_q;
    logic [USER_WIDTH-1:0] user_s1_q;
    logic [USER_WIDTH-1:0] user_s2_q;
    logic [USER_WIDTH-1:0] user_s3_q;
    logic [USER_WIDTH-1:0] user_s4_q;
    logic [15:0]           pass_fp16_s0_q;
    logic [15:0]           pass_fp16_s1_q;
    logic [15:0]           pass_fp16_s2_q;
    logic [15:0]           pass_fp16_s3_q;
    logic [15:0]           pass_fp16_s4_q;
    logic [3:0]            frac_s0_q;
    logic [3:0]            frac_s1_q;
    logic signed [31:0]    y0_q16_s0_q;
    logic signed [31:0]    y1_q16_s0_q;
    logic signed [31:0]    y0_q16_s1_q;
    logic signed [31:0]    y0_q16_s2_q;
    logic signed [31:0]    delta_q16_s1_q;
    logic signed [36:0]    interp_prod_s2_q;
    logic signed [31:0]    interp_q16_s3_q;
    logic                  tail_neg_s0_q;
    logic                  tail_pos_s0_q;
    logic                  tail_neg_s1_q;
    logic                  tail_pos_s1_q;
    logic                  tail_neg_s2_q;
    logic                  tail_pos_s2_q;
    logic                  tail_neg_s3_q;
    logic                  tail_pos_s3_q;
    logic                  tail_neg_s4_q;
    logic                  tail_pos_s4_q;
    logic                  conv_sign_w;
    logic [18:0]           conv_abs_w;
    logic [4:0]            conv_leading_pos_w;
    logic                  conv_zero_w;
    logic                  conv_sign_s4_q;
    logic [18:0]           conv_abs_s4_q;
    logic [4:0]            conv_leading_pos_s4_q;
    logic                  conv_zero_s4_q;

    function automatic logic signed [15:0] fp16_to_q8_8(input logic [15:0] fp16);
        logic sign;
        logic [4:0] exp;
        logic [9:0] frac;
        logic [10:0] mant;
        int shift;
        int signed value;
        begin
            sign = fp16[15];
            exp  = fp16[14:10];
            frac = fp16[9:0];
            mant = {1'b1, frac};
            value = 0;

            if (exp == 5'h1f) begin
                value = sign ? -32768 : 32767;
            end else if (exp != 5'd0) begin
                shift = int'(exp) - 17;
                if (shift >= 0)
                    value = int'(mant) <<< shift;
                else
                    value = int'(mant) >>> (-shift);

                if (sign)
                    value = -value;
            end

            if (value > 2048)
                fp16_to_q8_8 = 16'sd2048;
            else if (value < -2048)
                fp16_to_q8_8 = -16'sd2048;
            else
                fp16_to_q8_8 = 16'(value);
        end
    endfunction

    function automatic logic signed [31:0] silu_lut_q16(input logic [7:0] idx);
        begin
            unique case (idx)
                8'd0:   silu_lut_q16 = -32'sd176;
                8'd1:   silu_lut_q16 = -32'sd186;
                8'd2:   silu_lut_q16 = -32'sd196;
                8'd3:   silu_lut_q16 = -32'sd207;
                8'd4:   silu_lut_q16 = -32'sd219;
                8'd5:   silu_lut_q16 = -32'sd231;
                8'd6:   silu_lut_q16 = -32'sd244;
                8'd7:   silu_lut_q16 = -32'sd257;
                8'd8:   silu_lut_q16 = -32'sd272;
                8'd9:   silu_lut_q16 = -32'sd287;
                8'd10:  silu_lut_q16 = -32'sd303;
                8'd11:  silu_lut_q16 = -32'sd320;
                8'd12:  silu_lut_q16 = -32'sd337;
                8'd13:  silu_lut_q16 = -32'sd356;
                8'd14:  silu_lut_q16 = -32'sd375;
                8'd15:  silu_lut_q16 = -32'sd396;
                8'd16:  silu_lut_q16 = -32'sd418;
                8'd17:  silu_lut_q16 = -32'sd441;
                8'd18:  silu_lut_q16 = -32'sd465;
                8'd19:  silu_lut_q16 = -32'sd491;
                8'd20:  silu_lut_q16 = -32'sd517;
                8'd21:  silu_lut_q16 = -32'sd546;
                8'd22:  silu_lut_q16 = -32'sd575;
                8'd23:  silu_lut_q16 = -32'sd607;
                8'd24:  silu_lut_q16 = -32'sd639;
                8'd25:  silu_lut_q16 = -32'sd674;
                8'd26:  silu_lut_q16 = -32'sd711;
                8'd27:  silu_lut_q16 = -32'sd749;
                8'd28:  silu_lut_q16 = -32'sd789;
                8'd29:  silu_lut_q16 = -32'sd832;
                8'd30:  silu_lut_q16 = -32'sd876;
                8'd31:  silu_lut_q16 = -32'sd923;
                8'd32:  silu_lut_q16 = -32'sd972;
                8'd33:  silu_lut_q16 = -32'sd1024;
                8'd34:  silu_lut_q16 = -32'sd1078;
                8'd35:  silu_lut_q16 = -32'sd1136;
                8'd36:  silu_lut_q16 = -32'sd1196;
                8'd37:  silu_lut_q16 = -32'sd1259;
                8'd38:  silu_lut_q16 = -32'sd1325;
                8'd39:  silu_lut_q16 = -32'sd1394;
                8'd40:  silu_lut_q16 = -32'sd1467;
                8'd41:  silu_lut_q16 = -32'sd1544;
                8'd42:  silu_lut_q16 = -32'sd1624;
                8'd43:  silu_lut_q16 = -32'sd1708;
                8'd44:  silu_lut_q16 = -32'sd1796;
                8'd45:  silu_lut_q16 = -32'sd1888;
                8'd46:  silu_lut_q16 = -32'sd1985;
                8'd47:  silu_lut_q16 = -32'sd2087;
                8'd48:  silu_lut_q16 = -32'sd2193;
                8'd49:  silu_lut_q16 = -32'sd2304;
                8'd50:  silu_lut_q16 = -32'sd2421;
                8'd51:  silu_lut_q16 = -32'sd2543;
                8'd52:  silu_lut_q16 = -32'sd2670;
                8'd53:  silu_lut_q16 = -32'sd2803;
                8'd54:  silu_lut_q16 = -32'sd2943;
                8'd55:  silu_lut_q16 = -32'sd3088;
                8'd56:  silu_lut_q16 = -32'sd3240;
                8'd57:  silu_lut_q16 = -32'sd3399;
                8'd58:  silu_lut_q16 = -32'sd3564;
                8'd59:  silu_lut_q16 = -32'sd3737;
                8'd60:  silu_lut_q16 = -32'sd3917;
                8'd61:  silu_lut_q16 = -32'sd4105;
                8'd62:  silu_lut_q16 = -32'sd4300;
                8'd63:  silu_lut_q16 = -32'sd4503;
                8'd64:  silu_lut_q16 = -32'sd4715;
                8'd65:  silu_lut_q16 = -32'sd4935;
                8'd66:  silu_lut_q16 = -32'sd5163;
                8'd67:  silu_lut_q16 = -32'sd5401;
                8'd68:  silu_lut_q16 = -32'sd5647;
                8'd69:  silu_lut_q16 = -32'sd5902;
                8'd70:  silu_lut_q16 = -32'sd6167;
                8'd71:  silu_lut_q16 = -32'sd6440;
                8'd72:  silu_lut_q16 = -32'sd6724;
                8'd73:  silu_lut_q16 = -32'sd7016;
                8'd74:  silu_lut_q16 = -32'sd7318;
                8'd75:  silu_lut_q16 = -32'sd7630;
                8'd76:  silu_lut_q16 = -32'sd7950;
                8'd77:  silu_lut_q16 = -32'sd8280;
                8'd78:  silu_lut_q16 = -32'sd8620;
                8'd79:  silu_lut_q16 = -32'sd8968;
                8'd80:  silu_lut_q16 = -32'sd9324;
                8'd81:  silu_lut_q16 = -32'sd9689;
                8'd82:  silu_lut_q16 = -32'sd10062;
                8'd83:  silu_lut_q16 = -32'sd10442;
                8'd84:  silu_lut_q16 = -32'sd10829;
                8'd85:  silu_lut_q16 = -32'sd11222;
                8'd86:  silu_lut_q16 = -32'sd11620;
                8'd87:  silu_lut_q16 = -32'sd12023;
                8'd88:  silu_lut_q16 = -32'sd12429;
                8'd89:  silu_lut_q16 = -32'sd12837;
                8'd90:  silu_lut_q16 = -32'sd13245;
                8'd91:  silu_lut_q16 = -32'sd13654;
                8'd92:  silu_lut_q16 = -32'sd14060;
                8'd93:  silu_lut_q16 = -32'sd14462;
                8'd94:  silu_lut_q16 = -32'sd14858;
                8'd95:  silu_lut_q16 = -32'sd15246;
                8'd96:  silu_lut_q16 = -32'sd15624;
                8'd97:  silu_lut_q16 = -32'sd15989;
                8'd98:  silu_lut_q16 = -32'sd16339;
                8'd99:  silu_lut_q16 = -32'sd16670;
                8'd100: silu_lut_q16 = -32'sd16979;
                8'd101: silu_lut_q16 = -32'sd17264;
                8'd102: silu_lut_q16 = -32'sd17520;
                8'd103: silu_lut_q16 = -32'sd17745;
                8'd104: silu_lut_q16 = -32'sd17933;
                8'd105: silu_lut_q16 = -32'sd18082;
                8'd106: silu_lut_q16 = -32'sd18186;
                8'd107: silu_lut_q16 = -32'sd18241;
                8'd108: silu_lut_q16 = -32'sd18244;
                8'd109: silu_lut_q16 = -32'sd18188;
                8'd110: silu_lut_q16 = -32'sd18070;
                8'd111: silu_lut_q16 = -32'sd17884;
                8'd112: silu_lut_q16 = -32'sd17625;
                8'd113: silu_lut_q16 = -32'sd17290;
                8'd114: silu_lut_q16 = -32'sd16871;
                8'd115: silu_lut_q16 = -32'sd16366;
                8'd116: silu_lut_q16 = -32'sd15769;
                8'd117: silu_lut_q16 = -32'sd15075;
                8'd118: silu_lut_q16 = -32'sd14281;
                8'd119: silu_lut_q16 = -32'sd13380;
                8'd120: silu_lut_q16 = -32'sd12371;
                8'd121: silu_lut_q16 = -32'sd11249;
                8'd122: silu_lut_q16 = -32'sd10011;
                8'd123: silu_lut_q16 = -32'sd8653;
                8'd124: silu_lut_q16 = -32'sd7173;
                8'd125: silu_lut_q16 = -32'sd5570;
                8'd126: silu_lut_q16 = -32'sd3840;
                8'd127: silu_lut_q16 = -32'sd1984;
                8'd128: silu_lut_q16 = 32'sd0;
                8'd129: silu_lut_q16 = 32'sd2112;
                8'd130: silu_lut_q16 = 32'sd4352;
                8'd131: silu_lut_q16 = 32'sd6718;
                8'd132: silu_lut_q16 = 32'sd9211;
                8'd133: silu_lut_q16 = 32'sd11827;
                8'd134: silu_lut_q16 = 32'sd14565;
                8'd135: silu_lut_q16 = 32'sd17423;
                8'd136: silu_lut_q16 = 32'sd20397;
                8'd137: silu_lut_q16 = 32'sd23484;
                8'd138: silu_lut_q16 = 32'sd26679;
                8'd139: silu_lut_q16 = 32'sd29981;
                8'd140: silu_lut_q16 = 32'sd33383;
                8'd141: silu_lut_q16 = 32'sd36882;
                8'd142: silu_lut_q16 = 32'sd40473;
                8'd143: silu_lut_q16 = 32'sd44150;
                8'd144: silu_lut_q16 = 32'sd47911;
                8'd145: silu_lut_q16 = 32'sd51748;
                8'd146: silu_lut_q16 = 32'sd55658;
                8'd147: silu_lut_q16 = 32'sd59636;
                8'd148: silu_lut_q16 = 32'sd63676;
                8'd149: silu_lut_q16 = 32'sd67775;
                8'd150: silu_lut_q16 = 32'sd71926;
                8'd151: silu_lut_q16 = 32'sd76126;
                8'd152: silu_lut_q16 = 32'sd80371;
                8'd153: silu_lut_q16 = 32'sd84655;
                8'd154: silu_lut_q16 = 32'sd88976;
                8'd155: silu_lut_q16 = 32'sd93328;
                8'd156: silu_lut_q16 = 32'sd97709;
                8'd157: silu_lut_q16 = 32'sd102114;
                8'd158: silu_lut_q16 = 32'sd106541;
                8'd159: silu_lut_q16 = 32'sd110987;
                8'd160: silu_lut_q16 = 32'sd115448;
                8'd161: silu_lut_q16 = 32'sd119922;
                8'd162: silu_lut_q16 = 32'sd124406;
                8'd163: silu_lut_q16 = 32'sd128898;
                8'd164: silu_lut_q16 = 32'sd133396;
                8'd165: silu_lut_q16 = 32'sd137898;
                8'd166: silu_lut_q16 = 32'sd142403;
                8'd167: silu_lut_q16 = 32'sd146907;
                8'd168: silu_lut_q16 = 32'sd151411;
                8'd169: silu_lut_q16 = 32'sd155913;
                8'd170: silu_lut_q16 = 32'sd160412;
                8'd171: silu_lut_q16 = 32'sd164906;
                8'd172: silu_lut_q16 = 32'sd169395;
                8'd173: silu_lut_q16 = 32'sd173878;
                8'd174: silu_lut_q16 = 32'sd178354;
                8'd175: silu_lut_q16 = 32'sd182823;
                8'd176: silu_lut_q16 = 32'sd187284;
                8'd177: silu_lut_q16 = 32'sd191736;
                8'd178: silu_lut_q16 = 32'sd196180;
                8'd179: silu_lut_q16 = 32'sd200616;
                8'd180: silu_lut_q16 = 32'sd205042;
                8'd181: silu_lut_q16 = 32'sd209458;
                8'd182: silu_lut_q16 = 32'sd213866;
                8'd183: silu_lut_q16 = 32'sd218264;
                8'd184: silu_lut_q16 = 32'sd222652;
                8'd185: silu_lut_q16 = 32'sd227032;
                8'd186: silu_lut_q16 = 32'sd231401;
                8'd187: silu_lut_q16 = 32'sd235762;
                8'd188: silu_lut_q16 = 32'sd240113;
                8'd189: silu_lut_q16 = 32'sd244455;
                8'd190: silu_lut_q16 = 32'sd248789;
                8'd191: silu_lut_q16 = 32'sd253113;
                8'd192: silu_lut_q16 = 32'sd257429;
                8'd193: silu_lut_q16 = 32'sd261737;
                8'd194: silu_lut_q16 = 32'sd266036;
                8'd195: silu_lut_q16 = 32'sd270327;
                8'd196: silu_lut_q16 = 32'sd274611;
                8'd197: silu_lut_q16 = 32'sd278887;
                8'd198: silu_lut_q16 = 32'sd283156;
                8'd199: silu_lut_q16 = 32'sd287417;
                8'd200: silu_lut_q16 = 32'sd291672;
                8'd201: silu_lut_q16 = 32'sd295920;
                8'd202: silu_lut_q16 = 32'sd300161;
                8'd203: silu_lut_q16 = 32'sd304397;
                8'd204: silu_lut_q16 = 32'sd308626;
                8'd205: silu_lut_q16 = 32'sd312849;
                8'd206: silu_lut_q16 = 32'sd317067;
                8'd207: silu_lut_q16 = 32'sd321280;
                8'd208: silu_lut_q16 = 32'sd325487;
                8'd209: silu_lut_q16 = 32'sd329689;
                8'd210: silu_lut_q16 = 32'sd333887;
                8'd211: silu_lut_q16 = 32'sd338080;
                8'd212: silu_lut_q16 = 32'sd342268;
                8'd213: silu_lut_q16 = 32'sd346452;
                8'd214: silu_lut_q16 = 32'sd350632;
                8'd215: silu_lut_q16 = 32'sd354808;
                8'd216: silu_lut_q16 = 32'sd358981;
                8'd217: silu_lut_q16 = 32'sd363150;
                8'd218: silu_lut_q16 = 32'sd367315;
                8'd219: silu_lut_q16 = 32'sd371477;
                8'd220: silu_lut_q16 = 32'sd375636;
                8'd221: silu_lut_q16 = 32'sd379792;
                8'd222: silu_lut_q16 = 32'sd383946;
                8'd223: silu_lut_q16 = 32'sd388096;
                default: silu_lut_q16 = 32'sd392244;
            endcase
        end
    endfunction

    function automatic logic [15:0] fixed_q16_to_fp16(input logic signed [31:0] value_q16);
        logic sign;
        logic [31:0] abs_value;
        int leading_pos;
        int exp16;
        int shift;
        logic [31:0] mant_base;
        logic round_bit;
        logic sticky;
        logic [11:0] mant_round;
        begin
            sign = (value_q16 < 0);
            abs_value = sign ? $unsigned(-value_q16) : $unsigned(value_q16);
            leading_pos = -1;
            if (abs_value[31]) begin
                leading_pos = 31;
            end
            else if (abs_value[30]) begin
                leading_pos = 30;
            end
            else if (abs_value[29]) begin
                leading_pos = 29;
            end
            else if (abs_value[28]) begin
                leading_pos = 28;
            end
            else if (abs_value[27]) begin
                leading_pos = 27;
            end
            else if (abs_value[26]) begin
                leading_pos = 26;
            end
            else if (abs_value[25]) begin
                leading_pos = 25;
            end
            else if (abs_value[24]) begin
                leading_pos = 24;
            end
            else if (abs_value[23]) begin
                leading_pos = 23;
            end
            else if (abs_value[22]) begin
                leading_pos = 22;
            end
            else if (abs_value[21]) begin
                leading_pos = 21;
            end
            else if (abs_value[20]) begin
                leading_pos = 20;
            end
            else if (abs_value[19]) begin
                leading_pos = 19;
            end
            else if (abs_value[18]) begin
                leading_pos = 18;
            end
            else if (abs_value[17]) begin
                leading_pos = 17;
            end
            else if (abs_value[16]) begin
                leading_pos = 16;
            end
            else if (abs_value[15]) begin
                leading_pos = 15;
            end
            else if (abs_value[14]) begin
                leading_pos = 14;
            end
            else if (abs_value[13]) begin
                leading_pos = 13;
            end
            else if (abs_value[12]) begin
                leading_pos = 12;
            end
            else if (abs_value[11]) begin
                leading_pos = 11;
            end
            else if (abs_value[10]) begin
                leading_pos = 10;
            end
            else if (abs_value[9]) begin
                leading_pos = 9;
            end
            else if (abs_value[8]) begin
                leading_pos = 8;
            end
            else if (abs_value[7]) begin
                leading_pos = 7;
            end
            else if (abs_value[6]) begin
                leading_pos = 6;
            end
            else if (abs_value[5]) begin
                leading_pos = 5;
            end
            else if (abs_value[4]) begin
                leading_pos = 4;
            end
            else if (abs_value[3]) begin
                leading_pos = 3;
            end
            else if (abs_value[2]) begin
                leading_pos = 2;
            end
            else if (abs_value[1]) begin
                leading_pos = 1;
            end
            else if (abs_value[0]) begin
                leading_pos = 0;
            end

            fixed_q16_to_fp16 = {sign, 15'd0};
            mant_base = '0;
            round_bit = 1'b0;
            sticky = 1'b0;
            mant_round = '0;

            if (leading_pos >= 0) begin
                exp16 = leading_pos - 1;
                if (exp16 >= 31) begin
                    fixed_q16_to_fp16 = {sign, 5'h1f, 10'd0};
                end else if (exp16 > 0) begin
                    shift = leading_pos - 10;
                    if (shift > 0) begin
                        mant_base = abs_value >> shift;
                        round_bit = |(abs_value & (32'd1 << (shift - 1)));
                        sticky = |(abs_value & ((32'd1 << (shift - 1)) - 32'd1));
                    end else begin
                        mant_base = abs_value << (-shift);
                    end

                    mant_round = {1'b0, mant_base[10:0]} +
                                 ((round_bit && (sticky || mant_base[0])) ? 12'd1 : 12'd0);
                    if (mant_round[11]) begin
                        if (exp16 == 30)
                            fixed_q16_to_fp16 = {sign, 5'h1f, 10'd0};
                        else
                            fixed_q16_to_fp16 = {sign, exp16[4:0] + 5'd1, 10'd0};
                    end else begin
                        fixed_q16_to_fp16 = {sign, exp16[4:0], mant_round[9:0]};
                    end
                end
            end
        end
    endfunction

    function automatic logic [4:0] leading_pos_q16_19(input logic [18:0] abs_value);
        begin
            if (abs_value[18])
                leading_pos_q16_19 = 5'd18;
            else if (abs_value[17])
                leading_pos_q16_19 = 5'd17;
            else if (abs_value[16])
                leading_pos_q16_19 = 5'd16;
            else if (abs_value[15])
                leading_pos_q16_19 = 5'd15;
            else if (abs_value[14])
                leading_pos_q16_19 = 5'd14;
            else if (abs_value[13])
                leading_pos_q16_19 = 5'd13;
            else if (abs_value[12])
                leading_pos_q16_19 = 5'd12;
            else if (abs_value[11])
                leading_pos_q16_19 = 5'd11;
            else if (abs_value[10])
                leading_pos_q16_19 = 5'd10;
            else if (abs_value[9])
                leading_pos_q16_19 = 5'd9;
            else if (abs_value[8])
                leading_pos_q16_19 = 5'd8;
            else if (abs_value[7])
                leading_pos_q16_19 = 5'd7;
            else if (abs_value[6])
                leading_pos_q16_19 = 5'd6;
            else if (abs_value[5])
                leading_pos_q16_19 = 5'd5;
            else if (abs_value[4])
                leading_pos_q16_19 = 5'd4;
            else if (abs_value[3])
                leading_pos_q16_19 = 5'd3;
            else if (abs_value[2])
                leading_pos_q16_19 = 5'd2;
            else if (abs_value[1])
                leading_pos_q16_19 = 5'd1;
            else
                leading_pos_q16_19 = 5'd0;
        end
    endfunction

    function automatic logic [15:0] pack_q16_to_fp16(
        input logic        sign,
        input logic [18:0] abs_value,
        input logic [4:0]  leading_pos,
        input logic        is_zero
    );
        int exp16;
        int shift;
        logic [18:0] mant_base;
        logic round_bit;
        logic sticky;
        logic [11:0] mant_round;
        begin
            pack_q16_to_fp16 = {sign, 15'd0};
            mant_base = '0;
            round_bit = 1'b0;
            sticky = 1'b0;
            mant_round = '0;

            if (!is_zero) begin
                exp16 = int'(leading_pos) - 1;
                if (exp16 > 0) begin
                    shift = int'(leading_pos) - 10;
                    if (shift > 0) begin
                        mant_base = abs_value >> shift;
                        round_bit = |(abs_value & (19'd1 << (shift - 1)));
                        sticky = |(abs_value & ((19'd1 << (shift - 1)) - 19'd1));
                    end else begin
                        mant_base = abs_value << (-shift);
                    end

                    mant_round = {1'b0, mant_base[10:0]} +
                                 ((round_bit && (sticky || mant_base[0])) ? 12'd1 : 12'd0);
                    if (mant_round[11])
                        pack_q16_to_fp16 = {sign, exp16[4:0] + 5'd1, 10'd0};
                    else
                        pack_q16_to_fp16 = {sign, exp16[4:0], mant_round[9:0]};
                end
            end
        end
    endfunction

    always_comb begin
        x_q8_8_w = fp16_to_q8_8(fp16_in_q);
        offset_w = 12'(int'(x_q8_8_w) + 2048);
        lut_idx_w = offset_w[11:4];
        frac_w = offset_w[3:0];
        y0_q16_w = silu_lut_q16(lut_idx_w);
        y1_q16_w = silu_lut_q16(lut_idx_w + 8'd1);
        tail_neg_w = (x_q8_8_w <= -16'sd2048);
        tail_pos_w = (x_q8_8_w >= 16'sd1536);
    end

    always_comb begin
        delta_q16_w = y1_q16_s0_q - y0_q16_s0_q;
        interp_prod_w = delta_q16_s1_q * $signed({1'b0, frac_s1_q});
        if (interp_prod_s2_q >= 0)
            interp_delta_q16_w = 32'((interp_prod_s2_q + 37'sd8) >>> 4);
        else
            interp_delta_q16_w = 32'((interp_prod_s2_q - 37'sd8) >>> 4);
        interp_q16_w = y0_q16_s2_q + interp_delta_q16_w;
    end

    always_comb begin
        conv_sign_w = (interp_q16_s3_q < 0);
        conv_abs_w = conv_sign_w ? 19'($unsigned(-interp_q16_s3_q)) : 19'($unsigned(interp_q16_s3_q));
        conv_zero_w = (conv_abs_w == 19'd0);
        conv_leading_pos_w = leading_pos_q16_19(conv_abs_w);
        fp16_conv_w = pack_q16_to_fp16(conv_sign_s4_q, conv_abs_s4_q,
                                       conv_leading_pos_s4_q, conv_zero_s4_q);
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_in_q <= 1'b0;
            fp16_in_q <= '0;
            user_in_q <= '0;
            valid_s0_q <= 1'b0;
            valid_s1_q <= 1'b0;
            valid_s2_q <= 1'b0;
            valid_s3_q <= 1'b0;
            valid_s4_q <= 1'b0;
            lut_valid_q <= 1'b0;
            user_s0_q <= '0;
            user_s1_q <= '0;
            user_s2_q <= '0;
            user_s3_q <= '0;
            user_s4_q <= '0;
            lut_user_q <= '0;
            pass_fp16_s0_q <= '0;
            pass_fp16_s1_q <= '0;
            pass_fp16_s2_q <= '0;
            pass_fp16_s3_q <= '0;
            pass_fp16_s4_q <= '0;
            frac_s0_q <= '0;
            frac_s1_q <= '0;
            y0_q16_s0_q <= '0;
            y1_q16_s0_q <= '0;
            y0_q16_s1_q <= '0;
            y0_q16_s2_q <= '0;
            delta_q16_s1_q <= '0;
            interp_prod_s2_q <= '0;
            interp_q16_s3_q <= '0;
            tail_neg_s0_q <= 1'b0;
            tail_pos_s0_q <= 1'b0;
            tail_neg_s1_q <= 1'b0;
            tail_pos_s1_q <= 1'b0;
            tail_neg_s2_q <= 1'b0;
            tail_pos_s2_q <= 1'b0;
            tail_neg_s3_q <= 1'b0;
            tail_pos_s3_q <= 1'b0;
            tail_neg_s4_q <= 1'b0;
            tail_pos_s4_q <= 1'b0;
            conv_sign_s4_q <= 1'b0;
            conv_abs_s4_q <= '0;
            conv_leading_pos_s4_q <= '0;
            conv_zero_s4_q <= 1'b1;
            lut_fp16_q <= '0;
        end else begin
            valid_in_q <= valid_i;
            fp16_in_q <= fp16_i;
            user_in_q <= user_i;
            valid_s0_q <= valid_in_q;
            valid_s1_q <= valid_s0_q;
            valid_s2_q <= valid_s1_q;
            valid_s3_q <= valid_s2_q;
            valid_s4_q <= valid_s3_q;
            lut_valid_q <= valid_s4_q;

            user_s0_q <= user_in_q;
            user_s1_q <= user_s0_q;
            user_s2_q <= user_s1_q;
            user_s3_q <= user_s2_q;
            user_s4_q <= user_s3_q;
            lut_user_q <= user_s4_q;

            pass_fp16_s0_q <= fp16_in_q;
            pass_fp16_s1_q <= pass_fp16_s0_q;
            pass_fp16_s2_q <= pass_fp16_s1_q;
            pass_fp16_s3_q <= pass_fp16_s2_q;
            pass_fp16_s4_q <= pass_fp16_s3_q;
            frac_s0_q <= frac_w;
            frac_s1_q <= frac_s0_q;
            y0_q16_s0_q <= y0_q16_w;
            y1_q16_s0_q <= y1_q16_w;
            y0_q16_s1_q <= y0_q16_s0_q;
            y0_q16_s2_q <= y0_q16_s1_q;
            delta_q16_s1_q <= delta_q16_w;
            interp_prod_s2_q <= interp_prod_w;
            interp_q16_s3_q <= interp_q16_w;
            tail_neg_s0_q <= tail_neg_w;
            tail_pos_s0_q <= tail_pos_w;
            tail_neg_s1_q <= tail_neg_s0_q;
            tail_pos_s1_q <= tail_pos_s0_q;
            tail_neg_s2_q <= tail_neg_s1_q;
            tail_pos_s2_q <= tail_pos_s1_q;
            tail_neg_s3_q <= tail_neg_s2_q;
            tail_pos_s3_q <= tail_pos_s2_q;
            tail_neg_s4_q <= tail_neg_s3_q;
            tail_pos_s4_q <= tail_pos_s3_q;
            conv_sign_s4_q <= conv_sign_w;
            conv_abs_s4_q <= conv_abs_w;
            conv_leading_pos_s4_q <= conv_leading_pos_w;
            conv_zero_s4_q <= conv_zero_w;

            if (tail_neg_s4_q)
                lut_fp16_q <= 16'h0000;
            else if (tail_pos_s4_q)
                lut_fp16_q <= pass_fp16_s4_q;
            else
                lut_fp16_q <= fp16_conv_w;
        end
    end

    always_comb begin
        if (enable_i) begin
            valid_o = lut_valid_q;
            fp16_o  = lut_fp16_q;
            user_o  = lut_user_q;
        end else begin
            valid_o = valid_i;
            fp16_o  = fp16_i;
            user_o  = user_i;
        end
    end

endmodule

`endif
