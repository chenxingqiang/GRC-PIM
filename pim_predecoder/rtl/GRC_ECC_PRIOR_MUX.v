// Merge GRC corrected priors (info symbols) with ECC_BUFFER parity LLRs.
// Drives ECC_TOP.INPUT_LLR for tape-out integration.

module GRC_ECC_PRIOR_MUX #(
    parameter SYMBOL_NUM = 288,
    parameter INFO_NUM   = 256,
    parameter FIELD      = 3,
    parameter LLR_BIT    = 3,
    parameter SYM_IDX_W  = 10
) (
    input  wire        CLK,
    input  wire        RST,
    input  wire [SYMBOL_NUM * FIELD * LLR_BIT - 1:0] BUFFER_LLR,
    input  wire        GRC_VALID,
    input  wire [SYM_IDX_W - 1:0] GRC_SYM_IDX,
    input  wire signed [FIELD * 8 - 1:0] GRC_LLR_I8,
    input  wire        GRC_BATCH_DONE,
    input  wire        USE_GRC_LLR,
    output reg  [SYMBOL_NUM * FIELD * LLR_BIT - 1:0] ECC_INPUT_LLR,
    output reg         PRIORS_READY
);
    localparam LLR9_W = FIELD * LLR_BIT;

    reg signed [LLR9_W - 1:0] prior_mem [0:SYMBOL_NUM - 1];
    reg [INFO_NUM - 1:0] grc_written;
    reg [1:0] pack_cnt;
    reg batch_done_q;
    wire batch_done_rise = GRC_BATCH_DONE && !batch_done_q;
    wire signed [LLR9_W - 1:0] grc_llr9;

    LLR_I8_VEC_TO_ECC9 QUANT (.LLR_I8(GRC_LLR_I8), .LLR_ECC9(grc_llr9));

    integer s;

    always @(posedge CLK or posedge RST) begin
        if (RST) begin
            PRIORS_READY <= 0;
            ECC_INPUT_LLR <= 0;
            grc_written <= 0;
            pack_cnt <= 0;
            batch_done_q <= 0;
        end else begin
            PRIORS_READY <= 0;
            batch_done_q <= GRC_BATCH_DONE;

            if (GRC_VALID && USE_GRC_LLR && (GRC_SYM_IDX < INFO_NUM)) begin
                prior_mem[GRC_SYM_IDX] <= grc_llr9;
                grc_written[GRC_SYM_IDX] <= 1'b1;
            end

            if (batch_done_rise)
                pack_cnt <= 2'd2;
            else if (pack_cnt != 0) begin
                if (pack_cnt == 2'd1) begin
                    for (s = 0; s < SYMBOL_NUM; s = s + 1) begin
                        if (s < INFO_NUM && USE_GRC_LLR && grc_written[s])
                            ECC_INPUT_LLR[s * LLR9_W +: LLR9_W] <= prior_mem[s];
                        else
                            ECC_INPUT_LLR[s * LLR9_W +: LLR9_W] <=
                                BUFFER_LLR[s * LLR9_W +: LLR9_W];
                    end
                    PRIORS_READY <= 1;
                end
                pack_cnt <= pack_cnt - 1;
            end
        end
    end
endmodule
