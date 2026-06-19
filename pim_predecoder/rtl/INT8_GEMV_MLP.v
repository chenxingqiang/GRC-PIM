// Signed INT8 multiply-accumulate cell for GEMV MLP.
module INT8_MAC #(
    parameter ACCUM_BIT = 20
) (
    input  wire signed [7:0]  A,
    input  wire signed [7:0]  B,
    input  wire               ACCUM_RST,
    input  wire               ACCUM_EN,
    output reg  signed [ACCUM_BIT - 1:0] ACCUM
);
    always @(posedge ACCUM_EN or posedge ACCUM_RST) begin
        if (ACCUM_RST)
            ACCUM <= 0;
        else
            ACCUM <= ACCUM + $signed(A) * $signed(B);
    end
endmodule

// Piecewise sigmoid LUT: maps 8-bit signed pre-activation -> Q16 gate score.
module SIGMOID_LUT_Q16 (
    input  wire signed [7:0] X,
    output reg  [15:0]       Y
);
    always @(*) begin
        if (X <= -8)
            Y = 16'd2621;
        else if (X <= 0)
            Y = 16'd19660;
        else if (X <= 8)
            Y = 16'd45875;
        else
            Y = 16'd62914;
    end
endmodule

function automatic signed [7:0] relu8;
    input signed [19:0] x;
    reg signed [7:0] y;
    begin
        y = x[11:4];
        relu8 = (y < 0) ? 8'sd0 : y;
    end
endfunction

// Sequential INT8 GEMV MLP: 13 -> 24 (ReLU) -> 3 delta + gate.
module GEMV_MLP_CORE #(
    parameter FEAT_NUM = 13,
    parameter HIDDEN = 24,
    parameter OUT_NUM = 3,
    parameter WEIGHT_DIR = "pim_predecoder/rtl/Weights"
) (
    input  wire               CLK,
    input  wire               RST,
    input  wire               START,
    input  wire signed [FEAT_NUM * 8 - 1:0] FEAT_IN,
    input  wire signed [OUT_NUM * 8 - 1:0]  LLR_ANCHOR,
    output reg                DONE,
    output reg  signed [OUT_NUM * 8 - 1:0]  LLR_OUT,
    output reg  [15:0]        GATE_Q16
);
    localparam S_IDLE = 0, S_LOAD = 1, S_FC1 = 2, S_FC2 = 3, S_GATE = 4, S_GATE_APPLY = 5, S_OUT = 6;
    reg [2:0] state;
    reg [4:0] mac_idx;
    reg [4:0] row_idx;
    reg signed [7:0] fc1_w [0:HIDDEN * FEAT_NUM - 1];
    reg signed [7:0] fc1_b [0:HIDDEN - 1];
    reg signed [7:0] fc2_w [0:OUT_NUM * HIDDEN - 1];
    reg signed [7:0] fc2_b [0:OUT_NUM - 1];
    reg signed [7:0] gate_w [0:FEAT_NUM - 1];
    reg signed [7:0] gate_b [0:0];
    reg signed [19:0] acc;
    reg signed [7:0] feat [0:FEAT_NUM - 1];
    reg signed [7:0] hidden [0:HIDDEN - 1];
    reg signed [7:0] delta [0:OUT_NUM - 1];
    reg signed [7:0] gate_pre;
    reg signed [19:0] gate_acc_hold;
    integer i;

    initial begin
        $readmemh({WEIGHT_DIR, "/fc1_w.hex"}, fc1_w);
        $readmemh({WEIGHT_DIR, "/fc1_b.hex"}, fc1_b);
        $readmemh({WEIGHT_DIR, "/fc2_w.hex"}, fc2_w);
        $readmemh({WEIGHT_DIR, "/fc2_b.hex"}, fc2_b);
        $readmemh({WEIGHT_DIR, "/gate_w.hex"}, gate_w);
        $readmemh({WEIGHT_DIR, "/gate_b.hex"}, gate_b);
    end

    wire signed [19:0] fc1_next_acc = acc + $signed(feat[mac_idx]) * $signed(fc1_w[row_idx * FEAT_NUM + mac_idx]);
    wire signed [19:0] fc2_next_acc = acc + $signed(hidden[mac_idx]) * $signed(fc2_w[row_idx * HIDDEN + mac_idx]);
    wire signed [19:0] gate_next_acc = acc + $signed(feat[mac_idx]) * $signed(gate_w[mac_idx]);
    wire signed [7:0] gate_pre_combo = gate_acc_hold[11:4];
    wire [15:0] gate_lut_combo;
    SIGMOID_LUT_Q16 SIG2 (.X(gate_pre_combo), .Y(gate_lut_combo));

    always @(posedge CLK or posedge RST) begin
        if (RST) begin
            state <= S_IDLE;
            DONE <= 0;
            mac_idx <= 0;
            row_idx <= 0;
            acc <= 0;
            gate_acc_hold <= 0;
        end else begin
            DONE <= 0;
            case (state)
                S_IDLE: begin
                    if (START)
                        state <= S_LOAD;
                end
                S_LOAD: begin
                    for (i = 0; i < FEAT_NUM; i = i + 1)
                        feat[i] <= FEAT_IN[i * 8 +: 8];
                    row_idx <= 0;
                    mac_idx <= 0;
                    acc <= fc1_b[0];
                    state <= S_FC1;
                end
                S_FC1: begin
                    acc <= fc1_next_acc;
                    if (mac_idx == FEAT_NUM - 1) begin
                        hidden[row_idx] <= relu8(fc1_next_acc);
                        mac_idx <= 0;
                        if (row_idx == HIDDEN - 1) begin
                            row_idx <= 0;
                            acc <= fc2_b[0];
                            state <= S_FC2;
                        end else begin
                            row_idx <= row_idx + 1;
                            acc <= fc1_b[row_idx + 1];
                        end
                    end else
                        mac_idx <= mac_idx + 1;
                end
                S_FC2: begin
                    acc <= fc2_next_acc;
                    if (mac_idx == HIDDEN - 1) begin
                        delta[row_idx] <= fc2_next_acc[11:4];
                        mac_idx <= 0;
                        if (row_idx == OUT_NUM - 1) begin
                            row_idx <= 0;
                            acc <= gate_b[0];
                            state <= S_GATE;
                        end else begin
                            row_idx <= row_idx + 1;
                            acc <= fc2_b[row_idx + 1];
                        end
                    end else
                        mac_idx <= mac_idx + 1;
                end
                S_GATE: begin
                    acc <= gate_next_acc;
                    if (mac_idx == FEAT_NUM - 1) begin
                        gate_acc_hold <= gate_next_acc;
                        state <= S_GATE_APPLY;
                    end else
                        mac_idx <= mac_idx + 1;
                end
                S_GATE_APPLY: begin
                    gate_pre <= gate_acc_hold[11:4];
                    GATE_Q16 <= gate_lut_combo;
                    state <= S_OUT;
                end
                S_OUT: begin
                    LLR_OUT[0 * 8 +: 8] <= LLR_ANCHOR[0 * 8 +: 8] + delta[0];
                    LLR_OUT[1 * 8 +: 8] <= LLR_ANCHOR[1 * 8 +: 8] + delta[1];
                    LLR_OUT[2 * 8 +: 8] <= LLR_ANCHOR[2 * 8 +: 8] + delta[2];
                    DONE <= 1;
                    state <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
