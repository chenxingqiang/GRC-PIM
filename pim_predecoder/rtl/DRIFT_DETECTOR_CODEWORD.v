// Codeword-level drift: compare all bit positions between consecutive cycles.
module DRIFT_DETECTOR_CODEWORD #(
    parameter N_BITS = 16,
    parameter Q_WIDTH = 16
) (
    input  wire                  CLK,
    input  wire                  RST,
    input  wire                  CYCLE_ROW_LAST,
    input  wire [N_BITS - 1:0]   CUR_BITS,
    input  wire                  BATCH_END,
    output reg  [Q_WIDTH - 1:0]  DRIFT_SCORE_Q16
);
    reg [N_BITS - 1:0] prev_bits;
    reg                have_prev;
    reg [31:0]         disagree;
    reg [31:0]         total;

    integer i;
    reg [31:0] diff_count;

    always @(*) begin
        diff_count = 0;
        if (have_prev) begin
            for (i = 0; i < N_BITS; i = i + 1) begin
                if (CUR_BITS[i] != prev_bits[i])
                    diff_count = diff_count + 1;
            end
        end
    end

    wire [31:0] final_disagree = disagree + ((CYCLE_ROW_LAST && have_prev) ? diff_count : 32'd0);
    wire [31:0] final_total = total + ((CYCLE_ROW_LAST && have_prev) ? N_BITS : 32'd0);

    always @(posedge CLK or posedge RST) begin
        if (RST) begin
            prev_bits <= 0;
            have_prev <= 0;
            disagree <= 0;
            total <= 0;
            DRIFT_SCORE_Q16 <= 0;
        end else begin
            if (CYCLE_ROW_LAST) begin
                if (have_prev) begin
                    disagree <= disagree + diff_count;
                    total <= total + N_BITS;
                end
                if (!BATCH_END) begin
                    prev_bits <= CUR_BITS;
                    have_prev <= 1;
                end
            end
            if (BATCH_END) begin
                if (final_total > 0)
                    DRIFT_SCORE_Q16 <= (final_disagree * 65535) / final_total;
                disagree <= 0;
                total <= 0;
                have_prev <= 0;
            end
        end
    end
endmodule
