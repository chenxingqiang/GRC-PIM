`timescale 1ns / 1ps

module GRC_PIM_TOP_TB;
    reg clk;
    reg rst;
    reg cim_e;
    reg ce;
    reg [2:0] adc_in;
    reg cycle_last;
    reg symbol_last;
    reg signed [13 * 8 - 1:0] feat_in;
    reg feat_valid;
    wire use_gemv;
    wire [15:0] drift_q16;
    wire [15:0] gate_mean_q16;
    wire signed [8:0] out_llr;
    wire out_valid;

    GRC_PIM_TOP #(.N_CYCLES(2), .WEIGHT_DIR("pim_predecoder/rtl/Weights"))
        DUT (
            .CLK(clk), .RST(rst), .CIM_E(cim_e), .CE(ce),
            .ADC_IN(adc_in), .CYCLE_LAST(cycle_last), .SYMBOL_LAST(symbol_last),
            .FEAT_IN(feat_in), .FEAT_VALID(feat_valid),
            .USE_GEMV(use_gemv), .DRIFT_SCORE_Q16(drift_q16),
            .GATE_MEAN_Q16(gate_mean_q16), .OUT_LLR(out_llr), .OUT_VALID(out_valid)
        );

    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    initial begin
        rst = 1; ce = 0; cim_e = 1; adc_in = 3'b010;
        cycle_last = 0; symbol_last = 0; feat_valid = 0; feat_in = 0;
        #20 rst = 0;
        // Two-cycle B' accumulation with toggling bits (drift).
        ce = 1;
        #10 cycle_last = 1; #10 cycle_last = 0;
        adc_in = 3'b011;
        #10 cycle_last = 1; #10 cycle_last = 0;
        feat_in = {13{8'sd4}};
        feat_valid = 1;
        symbol_last = 1;
        #10 symbol_last = 0;
        #100;
        $display("drift_q16=%0d gate_mean=%0d use_gemv=%0d out_valid=%0d out_llr=%0d",
                 drift_q16, gate_mean_q16, use_gemv, out_valid, out_llr);
        $finish;
    end
endmodule
