module GRC_PIM_TOP #(
    parameter N_SYMBOLS = 8,
    parameter N_CYCLES = 4,
    parameter N_BITS = 16,
    parameter SYM_IDX_W = 4,
    parameter CYC_IDX_W = 4,
    parameter ADC_BIT = 3,
    parameter LLR_BIT = 3,
    parameter FIELD = 3,
    parameter FEAT_NUM = 13,
    parameter HIDDEN = 24,
    parameter TAU_D_Q16 = 16'd721,
    parameter TAU_D_MAX_Q16 = 16'd2490,
    parameter TAU_G_Q16 = 16'd39321,
    parameter WEIGHT_DIR = "pim_predecoder/rtl/Weights"
) (
    input  wire                           CLK,
    input  wire                           RST,
    input  wire                           CIM_E,
    input  wire                           CE,
    input  wire [ADC_BIT - 1:0]           ADC_IN,
    input  wire [SYM_IDX_W - 1:0]         SYM_IDX,
    input  wire [CYC_IDX_W - 1:0]         CYC_IDX,
    input  wire                           CYCLE_ROW_LAST,
    input  wire                           BATCH_LAST,
    input  wire signed [FEAT_NUM * 8 - 1:0] FEAT_IN,
    input  wire                           FEAT_VALID,
    output wire [N_BITS - 1:0]            CUR_BITS_BUS,
    output wire signed [FIELD * 8 - 1:0]  ACCUM_DBG,
    output wire signed [FIELD * LLR_BIT - 1:0] CYCLE_LLR_DBG,
    output wire                           USE_GEMV,
    output wire [15:0]                    DRIFT_SCORE_Q16,
    output wire [15:0]                    GATE_MEAN_Q16,
    output reg  signed [FIELD * 8 - 1:0]  OUT_LLR,
    output reg                            OUT_VALID,
    output reg  [SYM_IDX_W - 1:0]         OUT_SYM_IDX,
    output reg  [15:0]                    OUT_GATE_Q16,
    output reg                            BATCH_DONE
);
    localparam S_STREAM = 0, S_MLP = 1, S_EMIT = 2, S_DONE = 3;
    reg [1:0] state;
    reg [SYM_IDX_W - 1:0] mlp_sym;
    reg [SYM_IDX_W - 1:0] emit_sym;
    reg [15:0] gate_sum;
    reg [7:0] gate_cnt;
    reg mlp_start;
    wire mlp_done;
    reg [15:0] gate_mean_r;
    reg use_gemv_r;
    reg batch_end_pulse;

    wire [FIELD * LLR_BIT - 1:0] cycle_llr;
    reg  signed [7:0] accum_mem [0:N_SYMBOLS - 1][0:FIELD - 1];
    reg  signed [7:0] anchor_mem [0:N_SYMBOLS - 1][0:FIELD - 1];
    reg  signed [FEAT_NUM * 8 - 1:0] feat_mem [0:N_SYMBOLS - 1];
    reg  signed [7:0] gemv_mem [0:N_SYMBOLS - 1][0:FIELD - 1];
    reg  [15:0] sym_gate_mem [0:N_SYMBOLS - 1];

    wire signed [FIELD * 8 - 1:0] llr_gemv;
    wire [15:0] sym_gate_q16;

    reg [N_BITS - 1:0] cur_bits;
    wire [N_BITS - 1:0] drift_bits_v2;
    genvar gi;
    generate
        for (gi = 0; gi < N_BITS / 2; gi = gi + 1) begin
            assign drift_bits_v2[gi * 2 +: 2] =
                (CE && (SYM_IDX == gi)) ? ADC_IN[2:1] : cur_bits[gi * 2 +: 2];
        end
    endgenerate

    reg signed [FIELD * 8 - 1:0] accum_dbg_r;

    GRC_MESSAGE_TRANS #(.INPUT_BIT(ADC_BIT), .LLR_BIT(LLR_BIT), .FIELD(FIELD))
        LLR_LUT (.INPUT(ADC_IN), .CIM_E(CIM_E), .OUTPUT_LLR(cycle_llr));

    assign CYCLE_LLR_DBG = cycle_llr;
    assign ACCUM_DBG = accum_dbg_r;
    assign CUR_BITS_BUS = cur_bits;

    reg signed [FEAT_NUM * 8 - 1:0] mlp_feat_hold;
    reg signed [FIELD * 8 - 1:0] mlp_anchor_hold;
    reg mlp_inputs_ready;

    integer gi_mean;
    reg [31:0] gate_sum_compute;

    task prefetch_mlp_sym;
        input [SYM_IDX_W - 1:0] sym;
        begin
            mlp_feat_hold <= feat_mem[sym];
            mlp_anchor_hold <= {anchor_mem[sym][2], anchor_mem[sym][1], anchor_mem[sym][0]};
        end
    endtask

    GEMV_MLP_CORE #(.FEAT_NUM(FEAT_NUM), .HIDDEN(HIDDEN), .WEIGHT_DIR(WEIGHT_DIR))
        MLP (.CLK(CLK), .RST(RST), .START(mlp_start),
             .FEAT_IN(mlp_feat_hold), .LLR_ANCHOR(mlp_anchor_hold),
             .DONE(mlp_done), .LLR_OUT(llr_gemv), .GATE_Q16(sym_gate_q16));

    DRIFT_DETECTOR_CODEWORD #(.N_BITS(N_BITS))
        DRIFT (.CLK(CLK), .RST(RST), .CYCLE_ROW_LAST(CYCLE_ROW_LAST),
               .CUR_BITS(drift_bits_v2), .BATCH_END(batch_end_pulse),
               .DRIFT_SCORE_Q16(DRIFT_SCORE_Q16));

    assign GATE_MEAN_Q16 = gate_mean_r;
    assign USE_GEMV = use_gemv_r;

    function automatic signed [7:0] sext3;
        input [2:0] v;
        begin
            sext3 = {{5{v[2]}}, v};
        end
    endfunction

    wire signed [7:0] c0 = sext3(cycle_llr[2:0]);
    wire signed [7:0] c1 = sext3(cycle_llr[5:3]);
    wire signed [7:0] c2 = sext3(cycle_llr[8:6]);
    wire signed [7:0] next_accum0 = (CYC_IDX == 0) ? c0 : accum_mem[SYM_IDX][0] + c0;
    wire signed [7:0] next_accum1 = (CYC_IDX == 0) ? c1 : accum_mem[SYM_IDX][1] + c1;
    wire signed [7:0] next_accum2 = (CYC_IDX == 0) ? c2 : accum_mem[SYM_IDX][2] + c2;

    always @(posedge CLK or posedge RST) begin
        if (RST) begin
            state <= S_STREAM;
            mlp_sym <= 0;
            emit_sym <= 0;
            gate_sum <= 0;
            gate_cnt <= 0;
            gate_mean_r <= 0;
            mlp_start <= 0;
            use_gemv_r <= 0;
            OUT_VALID <= 0;
            OUT_LLR <= 0;
            OUT_SYM_IDX <= 0;
            batch_end_pulse <= 0;
            cur_bits <= 0;
            OUT_GATE_Q16 <= 0;
            BATCH_DONE <= 0;
            mlp_feat_hold <= 0;
            mlp_anchor_hold <= 0;
            mlp_inputs_ready <= 0;
        end else begin
            OUT_VALID <= 0;
            mlp_start <= 0;
            batch_end_pulse <= 0;

            if (CE && state == S_STREAM) begin
                cur_bits[SYM_IDX * 2 +: 2] <= ADC_IN[2:1];
                accum_mem[SYM_IDX][0] <= next_accum0;
                accum_mem[SYM_IDX][1] <= next_accum1;
                accum_mem[SYM_IDX][2] <= next_accum2;
                accum_dbg_r <= {next_accum2, next_accum1, next_accum0};
                if (FEAT_VALID)
                    feat_mem[SYM_IDX] <= FEAT_IN;

                if (CYC_IDX == N_CYCLES - 1) begin
                    anchor_mem[SYM_IDX][0] <= next_accum0;
                    anchor_mem[SYM_IDX][1] <= next_accum1;
                    anchor_mem[SYM_IDX][2] <= next_accum2;
                end

                if (BATCH_LAST) begin
                    batch_end_pulse <= 1;
                    mlp_sym <= 0;
                    gate_sum <= 0;
                    gate_cnt <= 0;
                    state <= S_MLP;
                    prefetch_mlp_sym({SYM_IDX_W{1'b0}});
                    mlp_inputs_ready <= 1;
                end
            end

            case (state)
                S_MLP: begin
                    if (mlp_inputs_ready && !mlp_start && !mlp_done)
                        mlp_start <= 1;
                    else if (mlp_done) begin
                        mlp_inputs_ready <= 0;
                        sym_gate_mem[mlp_sym] <= sym_gate_q16;
                        gemv_mem[mlp_sym][0] <= llr_gemv[0 * 8 +: 8];
                        gemv_mem[mlp_sym][1] <= llr_gemv[1 * 8 +: 8];
                        gemv_mem[mlp_sym][2] <= llr_gemv[2 * 8 +: 8];
                        gate_sum <= gate_sum + sym_gate_q16;
                        gate_cnt <= gate_cnt + 1;
                        if (mlp_sym == N_SYMBOLS - 1) begin
                            gate_sum_compute = sym_gate_q16;
                            for (gi_mean = 0; gi_mean < N_SYMBOLS - 1; gi_mean = gi_mean + 1)
                                gate_sum_compute = gate_sum_compute + sym_gate_mem[gi_mean];
                            gate_mean_r <= gate_sum_compute / N_SYMBOLS;
                            use_gemv_r = (DRIFT_SCORE_Q16 >= TAU_D_Q16) &&
                                (DRIFT_SCORE_Q16 <= TAU_D_MAX_Q16) &&
                                (gate_sum_compute / N_SYMBOLS >= TAU_G_Q16);
                            emit_sym <= 0;
                            state <= S_EMIT;
                        end else begin
                            mlp_sym <= mlp_sym + 1;
                            prefetch_mlp_sym(mlp_sym + 1);
                            mlp_inputs_ready <= 1;
                        end
                    end
                end
                S_EMIT: begin
                    OUT_SYM_IDX <= emit_sym;
                    OUT_GATE_Q16 <= sym_gate_mem[emit_sym];
                    if (use_gemv_r) begin
                        OUT_LLR[0 * 8 +: 8] <= gemv_mem[emit_sym][0];
                        OUT_LLR[1 * 8 +: 8] <= gemv_mem[emit_sym][1];
                        OUT_LLR[2 * 8 +: 8] <= gemv_mem[emit_sym][2];
                    end else begin
                        OUT_LLR[0 * 8 +: 8] <= anchor_mem[emit_sym][0];
                        OUT_LLR[1 * 8 +: 8] <= anchor_mem[emit_sym][1];
                        OUT_LLR[2 * 8 +: 8] <= anchor_mem[emit_sym][2];
                    end
                    OUT_VALID <= 1;
                    if (emit_sym == N_SYMBOLS - 1) begin
                        state <= S_DONE;
                        BATCH_DONE <= 1;
                    end else
                        emit_sym <= emit_sym + 1;
                end
                S_DONE: begin
                    BATCH_DONE <= 1;
                end
                default: ;
            endcase
        end
    end
endmodule
