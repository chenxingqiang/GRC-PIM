// Hardware-aligned LLR LUT — matches deps/NBLDPC_PIM/NBLDPC_Verilog/LLR_TRANS.v
module GRC_MESSAGE_TRANS #(
    parameter INPUT_BIT = 3,
    parameter LLR_BIT = 3,
    parameter FIELD = 3
) (
    input  wire [INPUT_BIT - 1:0] INPUT,
    input  wire                   CIM_E,
    output wire [FIELD * LLR_BIT - 1:0] OUTPUT_LLR
);
    reg signed [FIELD - 1:0][LLR_BIT - 1:0] OUTPUT_LLR_LOC;
    generate
        genvar i, j;
        for (i = 0; i < FIELD; i = i + 1) begin
            for (j = 0; j < LLR_BIT; j = j + 1) begin
                assign OUTPUT_LLR[i * LLR_BIT + j] = OUTPUT_LLR_LOC[i][j];
            end
        end
    endgenerate
    always @(*) begin
        case (INPUT)
            3'b000: begin
                OUTPUT_LLR_LOC[0] <= 3'sb010;
                OUTPUT_LLR_LOC[1] <= 3'sb001;
                OUTPUT_LLR_LOC[2] <= 3'sb000;
            end
            3'b001: begin
                OUTPUT_LLR_LOC[0] <= 3'sb001;
                OUTPUT_LLR_LOC[1] <= 3'sb010;
                if (CIM_E == 0)
                    OUTPUT_LLR_LOC[2] <= 3'sb000;
                else
                    OUTPUT_LLR_LOC[2] <= 3'sb001;
            end
            3'b010: begin
                OUTPUT_LLR_LOC[0] <= 3'sb001;
                OUTPUT_LLR_LOC[1] <= 3'sb001;
                OUTPUT_LLR_LOC[2] <= 3'sb010;
            end
            3'b011: begin
                OUTPUT_LLR_LOC[0] <= 3'sb010;
                OUTPUT_LLR_LOC[1] <= 3'sb001;
                OUTPUT_LLR_LOC[2] <= 3'sb001;
            end
            3'b100: begin
                OUTPUT_LLR_LOC[0] <= 3'sb001;
                OUTPUT_LLR_LOC[1] <= 3'sb010;
                OUTPUT_LLR_LOC[2] <= 3'sb000;
            end
            default: begin
                OUTPUT_LLR_LOC[0] <= 3'sb000;
                OUTPUT_LLR_LOC[1] <= 3'sb000;
                OUTPUT_LLR_LOC[2] <= 3'sb000;
            end
        endcase
    end
endmodule

// Accumulate T-cycle integer LLRs per symbol (B' product in hardware domain).
module B_PRIME_ACCUM #(
    parameter FIELD = 3,
    parameter LLR_BIT = 3,
    parameter ACCUM_BIT = 8
) (
    input  wire signed [FIELD * LLR_BIT - 1:0] CYCLE_LLR,
    input  wire                                ACCUM_RST,
    input  wire                                ACCUM_EN,
    output reg  signed [FIELD * ACCUM_BIT - 1:0] ACCUM_LLR
);
    integer k;
    wire signed [LLR_BIT - 1:0] c0, c1, c2;
    wire signed [ACCUM_BIT - 1:0] a0, a1, a2;
    assign c0 = CYCLE_LLR[0 * LLR_BIT +: LLR_BIT];
    assign c1 = CYCLE_LLR[1 * LLR_BIT +: LLR_BIT];
    assign c2 = CYCLE_LLR[2 * LLR_BIT +: LLR_BIT];
    assign a0 = ACCUM_LLR[0 * ACCUM_BIT +: ACCUM_BIT];
    assign a1 = ACCUM_LLR[1 * ACCUM_BIT +: ACCUM_BIT];
    assign a2 = ACCUM_LLR[2 * ACCUM_BIT +: ACCUM_BIT];
    always @(posedge ACCUM_EN or posedge ACCUM_RST) begin
        if (ACCUM_RST) begin
            ACCUM_LLR <= 0;
        end else if (ACCUM_EN) begin
            ACCUM_LLR[0 * ACCUM_BIT +: ACCUM_BIT] <= a0 + {{(ACCUM_BIT-LLR_BIT){c0[LLR_BIT-1]}}, c0};
            ACCUM_LLR[1 * ACCUM_BIT +: ACCUM_BIT] <= a1 + {{(ACCUM_BIT-LLR_BIT){c1[LLR_BIT-1]}}, c1};
            ACCUM_LLR[2 * ACCUM_BIT +: ACCUM_BIT] <= a2 + {{(ACCUM_BIT-LLR_BIT){c2[LLR_BIT-1]}}, c2};
        end
    end
endmodule
