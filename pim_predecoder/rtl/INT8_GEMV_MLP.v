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
// LANES : MAC products evaluated per clock cycle inside one row. A layer of
//         M rows x K features therefore costs M * ceil(K / LANES) cycles.
// PIPE  : 1 folds the per-symbol epilogue (gate LUT, output add, DONE) into the
//         last gate step and captures the next symbol's FEAT_IN in that same
//         cycle, so the four cycles of per-symbol overhead are overlapped with
//         the next symbol. The producer must present the next symbol's FEAT_IN
//         at the DONE edge; LLR_ANCHOR still belongs to the current symbol.
module GEMV_MLP_CORE #(
    parameter FEAT_NUM = 13,
    parameter HIDDEN = 24,
    parameter OUT_NUM = 3,
    parameter WEIGHT_DIR = "pim_predecoder/rtl/Weights",
    parameter LANES = 1,
    parameter PIPE = 0
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
    localparam FC1_STEPS = (FEAT_NUM + LANES - 1) / LANES;
    localparam FC2_STEPS = (HIDDEN + LANES - 1) / LANES;
    localparam GATE_STEPS = (FEAT_NUM + LANES - 1) / LANES;
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

    reg signed [19:0] fc1_lane_sum;
    reg signed [19:0] fc2_lane_sum;
    reg signed [19:0] gate_lane_sum;
    integer l1, l2, l3;

    // Sum up to LANES products of the current row step. Terms past the row width
    // are skipped, so LANES greater than the feature count is harmless.
    always @(*) begin
        fc1_lane_sum = 20'sd0;
        for (l1 = 0; l1 < LANES; l1 = l1 + 1)
            if (mac_idx * LANES + l1 < FEAT_NUM)
                fc1_lane_sum = fc1_lane_sum
                    + $signed(feat[mac_idx * LANES + l1])
                    * $signed(fc1_w[row_idx * FEAT_NUM + mac_idx * LANES + l1]);
    end

    always @(*) begin
        fc2_lane_sum = 20'sd0;
        for (l2 = 0; l2 < LANES; l2 = l2 + 1)
            if (mac_idx * LANES + l2 < HIDDEN)
                fc2_lane_sum = fc2_lane_sum
                    + $signed(hidden[mac_idx * LANES + l2])
                    * $signed(fc2_w[row_idx * HIDDEN + mac_idx * LANES + l2]);
    end

    always @(*) begin
        gate_lane_sum = 20'sd0;
        for (l3 = 0; l3 < LANES; l3 = l3 + 1)
            if (mac_idx * LANES + l3 < FEAT_NUM)
                gate_lane_sum = gate_lane_sum
                    + $signed(feat[mac_idx * LANES + l3])
                    * $signed(gate_w[mac_idx * LANES + l3]);
    end

    wire signed [19:0] fc1_step_acc = acc + fc1_lane_sum;
    wire signed [19:0] fc2_step_acc = acc + fc2_lane_sum;
    wire signed [19:0] gate_step_acc = acc + gate_lane_sum;
    wire signed [7:0] gate_pre_combo = PIPE ? gate_step_acc[11:4] : gate_acc_hold[11:4];
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
                    if (START) begin
                        if (PIPE) begin
                            for (i = 0; i < FEAT_NUM; i = i + 1)
                                feat[i] <= FEAT_IN[i * 8 +: 8];
                            row_idx <= 0;
                            mac_idx <= 0;
                            acc <= fc1_b[0];
                            state <= S_FC1;
                        end else
                            state <= S_LOAD;
                    end
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
                    acc <= fc1_step_acc;
                    if (mac_idx == FC1_STEPS - 1) begin
                        hidden[row_idx] <= relu8(fc1_step_acc);
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
                    acc <= fc2_step_acc;
                    if (mac_idx == FC2_STEPS - 1) begin
                        delta[row_idx] <= fc2_step_acc[11:4];
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
                    acc <= gate_step_acc;
                    if (mac_idx == GATE_STEPS - 1) begin
                        if (PIPE) begin
                            GATE_Q16 <= gate_lut_combo;
                            LLR_OUT[0 * 8 +: 8] <= LLR_ANCHOR[0 * 8 +: 8] + delta[0];
                            LLR_OUT[1 * 8 +: 8] <= LLR_ANCHOR[1 * 8 +: 8] + delta[1];
                            LLR_OUT[2 * 8 +: 8] <= LLR_ANCHOR[2 * 8 +: 8] + delta[2];
                            DONE <= 1;
                            for (i = 0; i < FEAT_NUM; i = i + 1)
                                feat[i] <= FEAT_IN[i * 8 +: 8];
                            row_idx <= 0;
                            mac_idx <= 0;
                            acc <= fc1_b[0];
                            state <= S_FC1;
                        end else begin
                            gate_acc_hold <= gate_step_acc;
                            state <= S_GATE_APPLY;
                        end
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

// Symbol-level three-stage pipeline: FC1 -> FC2 -> gate.
//
// Same 13 -> 24 (ReLU) -> 3 delta + gate network and the same integer
// arithmetic as GEMV_MLP_CORE (LANES=1, PIPE=0), so every output is
// bit-identical to the sequential reference. Only the schedule changes:
// consecutive symbols overlap across the stages, so the steady-state DONE
// period becomes the depth of the deepest stage instead of the sum of all
// stage latencies.
//
//   stage 1 FC1  : ROWS_PER_CYCLE hidden rows per cycle, each row a fully
//                  parallel FEAT_NUM-input dot product plus bias, then relu8
//   stage 2 FC2  : OUT_NUM output rows per cycle, HIDDEN-wide dot products
//                  plus bias, truncated to [11:4]
//   stage 3 GATE : FEAT_NUM-input dot product, sigmoid LUT, then
//                  LLR_OUT = LLR_ANCHOR + delta
//
// Stage 2 of symbol i runs in the first cycle of symbol i+1's FC1 slot and
// stage 3 one cycle later, so both are hidden behind the FC1 stream.
//
// Handshake: FEAT_IN and LLR_ANCHOR belong to the symbol whose FC1 slot is
// currently running and must stay stable for ceil(HIDDEN / ROWS_PER_CYCLE)
// cycles. DONE is a one-cycle pulse per completed symbol, in input order.
module GEMV_MLP_PIPE #(
    parameter ROWS_PER_CYCLE = 15,
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
    localparam FC1_CYCLES = (HIDDEN + ROWS_PER_CYCLE - 1) / ROWS_PER_CYCLE;

    reg signed [7:0] fc1_w [0:HIDDEN * FEAT_NUM - 1];
    reg signed [7:0] fc1_b [0:HIDDEN - 1];
    reg signed [7:0] fc2_w [0:OUT_NUM * HIDDEN - 1];
    reg signed [7:0] fc2_b [0:OUT_NUM - 1];
    reg signed [7:0] gate_w [0:FEAT_NUM - 1];
    reg signed [7:0] gate_b [0:0];

    initial begin
        $readmemh({WEIGHT_DIR, "/fc1_w.hex"}, fc1_w);
        $readmemh({WEIGHT_DIR, "/fc1_b.hex"}, fc1_b);
        $readmemh({WEIGHT_DIR, "/fc2_w.hex"}, fc2_w);
        $readmemh({WEIGHT_DIR, "/fc2_b.hex"}, fc2_b);
        $readmemh({WEIGHT_DIR, "/gate_w.hex"}, gate_w);
        $readmemh({WEIGHT_DIR, "/gate_b.hex"}, gate_b);
    end

    // ------------------------------------------------------------------
    // Control: one symbol slot per FC1_CYCLES cycles.
    // ------------------------------------------------------------------
    reg       running;
    reg [7:0] pos;                  // FC1 row block inside the current slot
    reg       v_fc2;                // a symbol is waiting for stage 2
    reg       v_gate;               // a symbol is waiting for stage 3

    wire s1_last  = running && (pos == FC1_CYCLES - 1);
    wire fc2_fire = running && (pos == 0) && v_fc2;
    wire gate_fire = running && v_gate;

    // ------------------------------------------------------------------
    // Stage 1: ROWS_PER_CYCLE fully parallel FEAT_NUM-wide dot products.
    // Rows past HIDDEN are clamped to row 0 and never written back.
    // ------------------------------------------------------------------
    reg signed [19:0] fc1_sum [0:ROWS_PER_CYCLE - 1];
    reg [7:0]         row_of  [0:ROWS_PER_CYCLE - 1];
    integer r, k, o, h;

    always @(*) begin
        for (r = 0; r < ROWS_PER_CYCLE; r = r + 1) begin
            if (pos * ROWS_PER_CYCLE + r < HIDDEN)
                row_of[r] = pos * ROWS_PER_CYCLE + r;
            else
                row_of[r] = 0;
            fc1_sum[r] = fc1_b[row_of[r]];
            for (k = 0; k < FEAT_NUM; k = k + 1)
                fc1_sum[r] = fc1_sum[r]
                    + $signed(FEAT_IN[k * 8 +: 8])
                    * $signed(fc1_w[row_of[r] * FEAT_NUM + k]);
        end
    end

    // ------------------------------------------------------------------
    // Stage 2: all OUT_NUM rows in one cycle.
    // ------------------------------------------------------------------
    reg signed [7:0] hidden [0:HIDDEN - 1];
    reg signed [7:0] delta  [0:OUT_NUM - 1];
    reg signed [19:0] fc2_sum [0:OUT_NUM - 1];

    always @(*) begin
        for (o = 0; o < OUT_NUM; o = o + 1) begin
            fc2_sum[o] = fc2_b[o];
            for (h = 0; h < HIDDEN; h = h + 1)
                fc2_sum[o] = fc2_sum[o]
                    + $signed(hidden[h]) * $signed(fc2_w[o * HIDDEN + h]);
        end
    end

    // ------------------------------------------------------------------
    // Stage 3: gate dot product and sigmoid LUT.
    // ------------------------------------------------------------------
    reg signed [7:0] feat_s2 [0:FEAT_NUM - 1];
    reg signed [7:0] feat_s3 [0:FEAT_NUM - 1];
    reg signed [OUT_NUM * 8 - 1:0] anchor_s2;
    reg signed [OUT_NUM * 8 - 1:0] anchor_s3;

    reg signed [19:0] gate_sum;
    wire [15:0] gate_lut;

    always @(*) begin
        gate_sum = gate_b[0];
        for (k = 0; k < FEAT_NUM; k = k + 1)
            gate_sum = gate_sum
                + $signed(feat_s3[k]) * $signed(gate_w[k]);
    end

    SIGMOID_LUT_Q16 GATE_LUT (.X(gate_sum[11:4]), .Y(gate_lut));

    // ------------------------------------------------------------------
    // Pipeline registers.
    // ------------------------------------------------------------------
    always @(posedge CLK or posedge RST) begin
        if (RST) begin
            running <= 0;
            pos <= 0;
            v_fc2 <= 0;
            v_gate <= 0;
            DONE <= 0;
            GATE_Q16 <= 0;
            LLR_OUT <= 0;
        end else begin
            DONE <= 0;

            // Stage 3: apply the gate and the LLR update for this symbol.
            if (gate_fire) begin
                GATE_Q16 <= gate_lut;
                LLR_OUT[0 * 8 +: 8] <= anchor_s3[0 * 8 +: 8] + delta[0];
                LLR_OUT[1 * 8 +: 8] <= anchor_s3[1 * 8 +: 8] + delta[1];
                LLR_OUT[2 * 8 +: 8] <= anchor_s3[2 * 8 +: 8] + delta[2];
                DONE <= 1;
            end

            // Stage 2: 3 x 24 dot products, then hand the symbol to stage 3.
            if (fc2_fire) begin
                for (o = 0; o < OUT_NUM; o = o + 1)
                    delta[o] <= fc2_sum[o][11:4];
                for (k = 0; k < FEAT_NUM; k = k + 1)
                    feat_s3[k] <= feat_s2[k];
                anchor_s3 <= anchor_s2;
            end

            // Stage 1: retire this cycle's row block, then, on the last block
            // of the symbol, latch the inputs stages 2 and 3 still need after
            // FEAT_IN moves on to the next symbol.
            if (running)
                for (r = 0; r < ROWS_PER_CYCLE; r = r + 1)
                    if (pos * ROWS_PER_CYCLE + r < HIDDEN)
                        hidden[pos * ROWS_PER_CYCLE + r] <= relu8(fc1_sum[r]);
            if (s1_last) begin
                for (k = 0; k < FEAT_NUM; k = k + 1)
                    feat_s2[k] <= FEAT_IN[k * 8 +: 8];
                anchor_s2 <= LLR_ANCHOR;
            end

            // Valid bookkeeping. Clears precede sets so that a C=1 schedule,
            // where a stage-2 pulse and a slot end share a cycle, stays full.
            if (fc2_fire)
                v_fc2 <= 0;
            if (s1_last)
                v_fc2 <= 1;
            if (gate_fire)
                v_gate <= 0;
            if (fc2_fire)
                v_gate <= 1;

            if (!running) begin
                if (START) begin
                    running <= 1;
                    pos <= 0;
                end
            end else
                pos <= (pos == FC1_CYCLES - 1) ? 0 : pos + 1;
        end
    end
endmodule
