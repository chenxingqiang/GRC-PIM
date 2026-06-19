// Global temporal drift score — mean consecutive-cycle bit disagreement in Q16.
module DRIFT_DETECTOR #(
    parameter BIT_WIDTH = 16,
    parameter TAU_D_Q16 = 16'd1638   // 0.025 * 65535
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

    assign DRIFT_GATE_OPEN = (DRIFT_SCORE_Q16 >= TAU_D_Q16);
endmodule

// Dual gate: drift score + mean model-confidence gate.
module GRC_DUAL_GATE #(
    parameter Q_WIDTH = 16,
    parameter TAU_D_Q16 = 16'd1638,   // 0.025
    parameter TAU_G_Q16 = 16'd39321     // 0.6
) (
    input  wire [Q_WIDTH - 1:0] DRIFT_SCORE_Q16,
    input  wire [Q_WIDTH - 1:0] GATE_MEAN_Q16,
    output wire                 USE_GEMV
);
    wire drift_ok = (DRIFT_SCORE_Q16 >= TAU_D_Q16);
    wire gate_ok  = (GATE_MEAN_Q16 >= TAU_G_Q16);
    assign USE_GEMV = drift_ok & gate_ok;
endmodule
