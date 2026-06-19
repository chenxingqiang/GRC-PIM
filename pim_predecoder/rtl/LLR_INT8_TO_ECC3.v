// Saturate GRC int8 LLR to ECC 3-bit signed prior (matches hw_llr.int8_to_llr3).

function automatic signed [2:0] sat_llr3;
    input signed [7:0] v;
    begin
        if (v > 3)
            sat_llr3 = 3'sd3;
        else if (v < -4)
            sat_llr3 = -3'sd4;
        else
            sat_llr3 = v[2:0];
    end
endfunction

module LLR_I8_VEC_TO_ECC9 (
    input  wire signed [23:0] LLR_I8,
    output wire signed [8:0]  LLR_ECC9
);
    wire signed [7:0] c0 = LLR_I8[7:0];
    wire signed [7:0] c1 = LLR_I8[15:8];
    wire signed [7:0] c2 = LLR_I8[23:16];
    assign LLR_ECC9 = {sat_llr3(c2), sat_llr3(c1), sat_llr3(c0)};
endmodule
