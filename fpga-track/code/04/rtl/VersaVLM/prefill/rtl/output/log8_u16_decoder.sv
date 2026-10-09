`default_nettype none

// log8PV probability decoder.
//
// Exact mode implements:
//   code == 0 ? 0 : round(32767 * 2^(-16 * (255 - code) / 254))
//
// Mantissa mode implements the smaller approximation:
//   q = 255 - code
//   p = round(32767 * 2^(-(q[3:0]) / 16)) >> q[7:4]
module log8_u15_exact_decoder (
    input  wire logic [7:0]  code_i,
    output logic [14:0]      p_o
);
    always_comb begin
        unique case (code_i)
            8'd0: p_o = 15'd0;
            8'd1: p_o = 15'd0;
            8'd2: p_o = 15'd1;
            8'd3: p_o = 15'd1;
            8'd4: p_o = 15'd1;
            8'd5: p_o = 15'd1;
            8'd6: p_o = 15'd1;
            8'd7: p_o = 15'd1;
            8'd8: p_o = 15'd1;
            8'd9: p_o = 15'd1;
            8'd10: p_o = 15'd1;
            8'd11: p_o = 15'd1;
            8'd12: p_o = 15'd1;
            8'd13: p_o = 15'd1;
            8'd14: p_o = 15'd1;
            8'd15: p_o = 15'd1;
            8'd16: p_o = 15'd1;
            8'd17: p_o = 15'd1;
            8'd18: p_o = 15'd1;
            8'd19: p_o = 15'd1;
            8'd20: p_o = 15'd1;
            8'd21: p_o = 15'd1;
            8'd22: p_o = 15'd1;
            8'd23: p_o = 15'd1;
            8'd24: p_o = 15'd1;
            8'd25: p_o = 15'd1;
            8'd26: p_o = 15'd1;
            8'd27: p_o = 15'd2;
            8'd28: p_o = 15'd2;
            8'd29: p_o = 15'd2;
            8'd30: p_o = 15'd2;
            8'd31: p_o = 15'd2;
            8'd32: p_o = 15'd2;
            8'd33: p_o = 15'd2;
            8'd34: p_o = 15'd2;
            8'd35: p_o = 15'd2;
            8'd36: p_o = 15'd2;
            8'd37: p_o = 15'd2;
            8'd38: p_o = 15'd3;
            8'd39: p_o = 15'd3;
            8'd40: p_o = 15'd3;
            8'd41: p_o = 15'd3;
            8'd42: p_o = 15'd3;
            8'd43: p_o = 15'd3;
            8'd44: p_o = 15'd3;
            8'd45: p_o = 15'd3;
            8'd46: p_o = 15'd4;
            8'd47: p_o = 15'd4;
            8'd48: p_o = 15'd4;
            8'd49: p_o = 15'd4;
            8'd50: p_o = 15'd4;
            8'd51: p_o = 15'd4;
            8'd52: p_o = 15'd5;
            8'd53: p_o = 15'd5;
            8'd54: p_o = 15'd5;
            8'd55: p_o = 15'd5;
            8'd56: p_o = 15'd6;
            8'd57: p_o = 15'd6;
            8'd58: p_o = 15'd6;
            8'd59: p_o = 15'd6;
            8'd60: p_o = 15'd7;
            8'd61: p_o = 15'd7;
            8'd62: p_o = 15'd7;
            8'd63: p_o = 15'd7;
            8'd64: p_o = 15'd8;
            8'd65: p_o = 15'd8;
            8'd66: p_o = 15'd9;
            8'd67: p_o = 15'd9;
            8'd68: p_o = 15'd9;
            8'd69: p_o = 15'd10;
            8'd70: p_o = 15'd10;
            8'd71: p_o = 15'd11;
            8'd72: p_o = 15'd11;
            8'd73: p_o = 15'd12;
            8'd74: p_o = 15'd12;
            8'd75: p_o = 15'd13;
            8'd76: p_o = 15'd13;
            8'd77: p_o = 15'd14;
            8'd78: p_o = 15'd14;
            8'd79: p_o = 15'd15;
            8'd80: p_o = 15'd16;
            8'd81: p_o = 15'd16;
            8'd82: p_o = 15'd17;
            8'd83: p_o = 15'd18;
            8'd84: p_o = 15'd19;
            8'd85: p_o = 15'd20;
            8'd86: p_o = 15'd20;
            8'd87: p_o = 15'd21;
            8'd88: p_o = 15'd22;
            8'd89: p_o = 15'd23;
            8'd90: p_o = 15'd24;
            8'd91: p_o = 15'd25;
            8'd92: p_o = 15'd27;
            8'd93: p_o = 15'd28;
            8'd94: p_o = 15'd29;
            8'd95: p_o = 15'd30;
            8'd96: p_o = 15'd32;
            8'd97: p_o = 15'd33;
            8'd98: p_o = 15'd35;
            8'd99: p_o = 15'd36;
            8'd100: p_o = 15'd38;
            8'd101: p_o = 15'd39;
            8'd102: p_o = 15'd41;
            8'd103: p_o = 15'd43;
            8'd104: p_o = 15'd45;
            8'd105: p_o = 15'd47;
            8'd106: p_o = 15'd49;
            8'd107: p_o = 15'd51;
            8'd108: p_o = 15'd53;
            8'd109: p_o = 15'd56;
            8'd110: p_o = 15'd58;
            8'd111: p_o = 15'd61;
            8'd112: p_o = 15'd64;
            8'd113: p_o = 15'd66;
            8'd114: p_o = 15'd69;
            8'd115: p_o = 15'd73;
            8'd116: p_o = 15'd76;
            8'd117: p_o = 15'd79;
            8'd118: p_o = 15'd83;
            8'd119: p_o = 15'd86;
            8'd120: p_o = 15'd90;
            8'd121: p_o = 15'd94;
            8'd122: p_o = 15'd98;
            8'd123: p_o = 15'd103;
            8'd124: p_o = 15'd107;
            8'd125: p_o = 15'd112;
            8'd126: p_o = 15'd117;
            8'd127: p_o = 15'd123;
            8'd128: p_o = 15'd128;
            8'd129: p_o = 15'd134;
            8'd130: p_o = 15'd140;
            8'd131: p_o = 15'd146;
            8'd132: p_o = 15'd152;
            8'd133: p_o = 15'd159;
            8'd134: p_o = 15'd166;
            8'd135: p_o = 15'd174;
            8'd136: p_o = 15'd182;
            8'd137: p_o = 15'd190;
            8'd138: p_o = 15'd198;
            8'd139: p_o = 15'd207;
            8'd140: p_o = 15'd216;
            8'd141: p_o = 15'd226;
            8'd142: p_o = 15'd236;
            8'd143: p_o = 15'd246;
            8'd144: p_o = 15'd257;
            8'd145: p_o = 15'd269;
            8'd146: p_o = 15'd281;
            8'd147: p_o = 15'd293;
            8'd148: p_o = 15'd307;
            8'd149: p_o = 15'd320;
            8'd150: p_o = 15'd334;
            8'd151: p_o = 15'd349;
            8'd152: p_o = 15'd365;
            8'd153: p_o = 15'd381;
            8'd154: p_o = 15'd398;
            8'd155: p_o = 15'd416;
            8'd156: p_o = 15'd435;
            8'd157: p_o = 15'd454;
            8'd158: p_o = 15'd474;
            8'd159: p_o = 15'd495;
            8'd160: p_o = 15'd518;
            8'd161: p_o = 15'd541;
            8'd162: p_o = 15'd565;
            8'd163: p_o = 15'd590;
            8'd164: p_o = 15'd616;
            8'd165: p_o = 15'd644;
            8'd166: p_o = 15'd673;
            8'd167: p_o = 15'd703;
            8'd168: p_o = 15'd734;
            8'd169: p_o = 15'd767;
            8'd170: p_o = 15'd801;
            8'd171: p_o = 15'd837;
            8'd172: p_o = 15'd874;
            8'd173: p_o = 15'd913;
            8'd174: p_o = 15'd954;
            8'd175: p_o = 15'd996;
            8'd176: p_o = 15'd1041;
            8'd177: p_o = 15'd1087;
            8'd178: p_o = 15'd1136;
            8'd179: p_o = 15'd1187;
            8'd180: p_o = 15'd1240;
            8'd181: p_o = 15'd1295;
            8'd182: p_o = 15'd1353;
            8'd183: p_o = 15'd1413;
            8'd184: p_o = 15'd1476;
            8'd185: p_o = 15'd1542;
            8'd186: p_o = 15'd1611;
            8'd187: p_o = 15'd1683;
            8'd188: p_o = 15'd1758;
            8'd189: p_o = 15'd1836;
            8'd190: p_o = 15'd1918;
            8'd191: p_o = 15'd2004;
            8'd192: p_o = 15'd2093;
            8'd193: p_o = 15'd2187;
            8'd194: p_o = 15'd2284;
            8'd195: p_o = 15'd2386;
            8'd196: p_o = 15'd2493;
            8'd197: p_o = 15'd2604;
            8'd198: p_o = 15'd2720;
            8'd199: p_o = 15'd2841;
            8'd200: p_o = 15'd2968;
            8'd201: p_o = 15'd3101;
            8'd202: p_o = 15'd3239;
            8'd203: p_o = 15'd3384;
            8'd204: p_o = 15'd3535;
            8'd205: p_o = 15'd3692;
            8'd206: p_o = 15'd3857;
            8'd207: p_o = 15'd4029;
            8'd208: p_o = 15'd4209;
            8'd209: p_o = 15'd4397;
            8'd210: p_o = 15'd4593;
            8'd211: p_o = 15'd4798;
            8'd212: p_o = 15'd5012;
            8'd213: p_o = 15'd5236;
            8'd214: p_o = 15'd5470;
            8'd215: p_o = 15'd5714;
            8'd216: p_o = 15'd5969;
            8'd217: p_o = 15'd6235;
            8'd218: p_o = 15'd6514;
            8'd219: p_o = 15'd6804;
            8'd220: p_o = 15'd7108;
            8'd221: p_o = 15'd7425;
            8'd222: p_o = 15'd7757;
            8'd223: p_o = 15'd8103;
            8'd224: p_o = 15'd8464;
            8'd225: p_o = 15'd8842;
            8'd226: p_o = 15'd9237;
            8'd227: p_o = 15'd9649;
            8'd228: p_o = 15'd10080;
            8'd229: p_o = 15'd10530;
            8'd230: p_o = 15'd11000;
            8'd231: p_o = 15'd11490;
            8'd232: p_o = 15'd12003;
            8'd233: p_o = 15'd12539;
            8'd234: p_o = 15'd13099;
            8'd235: p_o = 15'd13683;
            8'd236: p_o = 15'd14294;
            8'd237: p_o = 15'd14932;
            8'd238: p_o = 15'd15598;
            8'd239: p_o = 15'd16294;
            8'd240: p_o = 15'd17022;
            8'd241: p_o = 15'd17781;
            8'd242: p_o = 15'd18575;
            8'd243: p_o = 15'd19404;
            8'd244: p_o = 15'd20270;
            8'd245: p_o = 15'd21174;
            8'd246: p_o = 15'd22119;
            8'd247: p_o = 15'd23107;
            8'd248: p_o = 15'd24138;
            8'd249: p_o = 15'd25215;
            8'd250: p_o = 15'd26341;
            8'd251: p_o = 15'd27516;
            8'd252: p_o = 15'd28744;
            8'd253: p_o = 15'd30027;
            8'd254: p_o = 15'd31367;
            8'd255: p_o = 15'd32767;
            default: p_o = 15'd0;
        endcase
    end
endmodule

module log8_u15_mant15_decoder (
    input  wire logic [7:0]  code_i,
    output logic [14:0]      p_o
);
    logic [7:0] q;
    logic [14:0] mant;

    always_comb begin
        q = 8'd255 - code_i;

        unique case (q[3:0])
            4'd0: mant = 15'd32767;
            4'd1: mant = 15'd31378;
            4'd2: mant = 15'd30047;
            4'd3: mant = 15'd28774;
            4'd4: mant = 15'd27554;
            4'd5: mant = 15'd26385;
            4'd6: mant = 15'd25267;
            4'd7: mant = 15'd24196;
            4'd8: mant = 15'd23170;
            4'd9: mant = 15'd22187;
            4'd10: mant = 15'd21247;
            4'd11: mant = 15'd20346;
            4'd12: mant = 15'd19483;
            4'd13: mant = 15'd18657;
            4'd14: mant = 15'd17866;
            default: mant = 15'd17109;
        endcase

        p_o = (code_i == 8'd0) ? 15'd0 : (mant >> q[7:4]);
    end
endmodule

module log8_u15_vector_decoder #(
    parameter int LANES = 32,
    parameter bit USE_MANT15 = 1'b0,
    parameter bit REGISTER_INPUT = 1'b0,
    parameter bit REGISTER_OUTPUT = 1'b0
) (
    input  wire logic                 clk_i,
    input  wire logic [LANES*8-1:0]   code_i,
    output logic      [LANES*15-1:0]  p_o
);
    logic [LANES*8-1:0]  code_q;
    logic [LANES*8-1:0]  code_dec;
    logic [LANES*15-1:0] p_comb;

    generate
        if (REGISTER_INPUT) begin : g_input_reg
            always_ff @(posedge clk_i) begin
                code_q <= code_i;
            end
            assign code_dec = code_q;
        end else begin : g_no_input_reg
            assign code_dec = code_i;
        end

        genvar lane;
        for (lane = 0; lane < LANES; lane = lane + 1) begin : g_lane
            if (USE_MANT15) begin : g_mant15
                log8_u15_mant15_decoder u_decoder (
                    .code_i (code_dec[lane*8 +: 8]),
                    .p_o    (p_comb[lane*15 +: 15])
                );
            end else begin : g_exact
                log8_u15_exact_decoder u_decoder (
                    .code_i (code_dec[lane*8 +: 8]),
                    .p_o    (p_comb[lane*15 +: 15])
                );
            end
        end

        if (REGISTER_OUTPUT) begin : g_output_reg
            always_ff @(posedge clk_i) begin
                p_o <= p_comb;
            end
        end else begin : g_no_output_reg
            assign p_o = p_comb;
        end
    endgenerate
endmodule

`default_nettype wire
