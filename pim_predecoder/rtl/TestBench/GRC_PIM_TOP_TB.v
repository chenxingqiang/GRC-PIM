`timescale 1ns / 1ps

module GRC_PIM_TOP_TB;
    reg clk;
    reg rst;
    reg cim_e;
    reg ce;
    reg [2:0] adc_in;
    reg [3:0] cycle;
    reg last_row, batch_last;
    reg signed [13 * 8 - 1:0] feat_in;
    reg feat_valid;
    wire use_gemv;
    wire [15:0] drift_q16;
    wire [15:0] gate_mean_q16;
    wire signed [23:0] out_llr;
    wire out_valid;

    GRC_PIM_TOP #(.N_SYMBOLS(1), .N_BITS(2), .N_CYCLES(2), .WEIGHT_DIR("pim_predecoder/rtl/Weights"))
        DUT (
            .CLK(clk), .RST(rst), .CIM_E(cim_e), .CE(ce),
            .ADC_IN(adc_in), .SYM_IDX(4'd0), .CYC_IDX(cycle),
            .CYCLE_ROW_LAST(last_row), .BATCH_LAST(batch_last),
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
        cycle = 0; last_row = 0; batch_last = 0; feat_valid = 0; feat_in = 0;
        repeat (2) @(negedge clk);
        rst = 0; ce = 1; last_row = 1; feat_valid = 1; feat_in = {13{8'sd4}};
        @(negedge clk);
        cycle = 1; adc_in = 3'b011; batch_last = 1;
        @(negedge clk);
        ce = 0; last_row = 0; batch_last = 0; feat_valid = 0;
        wait (out_valid);
        // Identical ADC[2:1] across reads: gate closed, output = summed LUTs.
        if ({drift_q16, use_gemv, out_llr} !== {16'd0, 1'b0, 8'd3, 8'd2, 8'd3})
            $fatal(1, "unexpected gated output: drift=%0d use=%b llr=%h", drift_q16, use_gemv, out_llr);
        $display("TOP smoke: correct B-prime fallback");
        $finish;
    end

    initial #10000 $fatal(1, "Timed out waiting for OUT_VALID");
endmodule
