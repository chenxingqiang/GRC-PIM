// Global temporal drift score — mean consecutive-cycle bit disagreement in Q16.
//
// The drift gate is a BAND, not a floor. TAU_D_Q16 asks whether multi-cycle
// evidence exists at all; TAU_D_MAX_Q16 asks whether the impairment is still
// within the strength the calibrator was calibrated for. The upper bound is
// what rejects conductance-domain channels whose drift score runs an order of
// magnitude past anything in the training distribution -- the model-confidence
// gate does not catch those, its score stays near 0.9 there.
//
// Both bounds are calibrated per code scale, at the per-codeword granularity
// the detector actually publishes at (see sn-article.tex Sec. 4.4). These
// defaults are the 1024-symbol rate-16/17 values.
module DRIFT_DETECTOR #(
    parameter BIT_WIDTH = 16,
    parameter TAU_D_Q16 = 16'd721,      // 0.011 * 65535, validation-calibrated
    parameter TAU_D_MAX_Q16 = 16'd2490  // 0.038 * 65535, validation-calibrated
) (
    input  wire                     CLK,
    input  wire                     RST,
    input  wire                     SAMPLE_EN,
    input  wire                     CYCLE_END,
    input  wire                     BIT_A,
    input  wire                     BIT_B,
    output reg  [BIT_WIDTH - 1:0]   DRIFT_SCORE_Q16,
    output wire                     DRIFT_GATE_OPEN
);
    reg [31:0] disagree_cnt;
    reg [31:0] total_cnt;
    reg        prev_bit;
    reg        have_prev;

    always @(posedge CLK or posedge RST) begin
        if (RST) begin
            disagree_cnt <= 0;
            total_cnt <= 0;
            have_prev <= 0;
            prev_bit <= 0;
            DRIFT_SCORE_Q16 <= 0;
        end else if (SAMPLE_EN) begin
            if (have_prev && (BIT_A != prev_bit))
                disagree_cnt <= disagree_cnt + 1;
            total_cnt <= total_cnt + 1;
            prev_bit <= BIT_A;
            have_prev <= 1;
        end else if (CYCLE_END && total_cnt > 0) begin
            DRIFT_SCORE_Q16 <= (disagree_cnt * 65535) / total_cnt;
            disagree_cnt <= 0;
            total_cnt <= 0;
            have_prev <= 0;
        end
    end

    assign DRIFT_GATE_OPEN = (DRIFT_SCORE_Q16 >= TAU_D_Q16)
                          && (DRIFT_SCORE_Q16 <= TAU_D_MAX_Q16);
endmodule

// Dual gate: drift score band + mean model-confidence gate.
module GRC_DUAL_GATE #(
    parameter Q_WIDTH = 16,
    parameter TAU_D_Q16 = 16'd721,       // 0.011
    parameter TAU_D_MAX_Q16 = 16'd2490,  // 0.038
    parameter TAU_G_Q16 = 16'd39321      // 0.6
) (
    input  wire [Q_WIDTH - 1:0] DRIFT_SCORE_Q16,
    input  wire [Q_WIDTH - 1:0] GATE_MEAN_Q16,
    output wire                 USE_GEMV
);
    wire drift_ok = (DRIFT_SCORE_Q16 >= TAU_D_Q16)
                 && (DRIFT_SCORE_Q16 <= TAU_D_MAX_Q16);
    wire gate_ok  = (GATE_MEAN_Q16 >= TAU_G_Q16);
    assign USE_GEMV = drift_ok & gate_ok;
endmodule
