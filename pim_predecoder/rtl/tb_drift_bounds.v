// Verification of the Q16 drift score and the band-pass gate bounds.
//
// Checks four things the band-pass gate depends on:
//   1. Q16 scaling is exact for per-codeword accumulation.
//   2. The 32-bit product's wrap point is where the per-codeword BATCH_END
//      requirement comes from, pinned so it cannot regress silently.
//   3. The gate opens only inside [TAU_D_Q16, TAU_D_MAX_Q16], per code scale.
//   4. At per-codeword granularity the in-distribution and out-of-family
//      score distributions overlap, so the band is a trade-off and not a
//      perfect separator. Both tails are pinned.
`timescale 1ns/1ps

module tb_drift_bounds;
    localparam integer N_BITS = 64;

    reg clk = 0, rst = 1, cycle_row_last = 0, batch_end = 0;
    reg [N_BITS-1:0] cur_bits;
    wire [15:0] score;
    integer errors = 0;

    always #5 clk = ~clk;

    DRIFT_DETECTOR_CODEWORD #(.N_BITS(N_BITS)) dut (
        .CLK(clk), .RST(rst), .CYCLE_ROW_LAST(cycle_row_last),
        .CUR_BITS(cur_bits), .BATCH_END(batch_end), .DRIFT_SCORE_Q16(score));

    // Drive `n_rows` cycle rows, flipping `n_flip` of the N_BITS positions on
    // every row after the first, then pulse BATCH_END and check the score.
    //
    // Stimulus is applied on the falling edge: driving CUR_BITS right after a
    // rising edge races the DUT's own `prev_bits <= CUR_BITS`, since both sit
    // in the same active region and their order is not defined.
    task run_case(input integer n_rows, input integer n_flip,
                  input integer exp_q16, input [255:0] name);
        integer r;
        reg [N_BITS-1:0] mask;
        begin
            rst = 1; cycle_row_last = 0; batch_end = 0; cur_bits = 0;
            @(negedge clk); @(negedge clk);
            rst = 0;
            mask = ({{(N_BITS-1){1'b0}}, 1'b1} << n_flip) - 1'b1;

            @(negedge clk);
            cur_bits = 0;
            cycle_row_last = 1;
            @(posedge clk);                       // row 0 establishes prev_bits
            for (r = 1; r < n_rows; r = r + 1) begin
                @(negedge clk); cur_bits = mask;  // n_flip positions differ
                @(posedge clk);
                @(negedge clk); cur_bits = 0;     // and differ again
                @(posedge clk);
            end
            @(negedge clk);
            cycle_row_last = 0;
            batch_end = 1;
            @(posedge clk);
            @(negedge clk);
            batch_end = 0;
            if (score !== exp_q16[15:0]) begin
                $display("FAIL %0s: got %0d expected %0d", name, score, exp_q16);
                errors = errors + 1;
            end else begin
                $display("ok   %0s: Q16=%0d", name, score);
            end
        end
    endtask

    // Band-pass bound check on the combinational gate.
    reg [15:0] d_in, g_in;
    wire use_gemv;
    GRC_DUAL_GATE #(.TAU_D_Q16(16'd721), .TAU_D_MAX_Q16(16'd2490),
                    .TAU_G_Q16(16'd39321))
        gate (.DRIFT_SCORE_Q16(d_in), .GATE_MEAN_Q16(g_in), .USE_GEMV(use_gemv));

    // Same gate carrying the 256-symbol rate-8/9 band.
    wire use_gemv_256;
    GRC_DUAL_GATE #(.TAU_D_Q16(16'd328), .TAU_D_MAX_Q16(16'd2949),
                    .TAU_G_Q16(16'd39321))
        gate_256 (.DRIFT_SCORE_Q16(d_in), .GATE_MEAN_Q16(g_in),
                  .USE_GEMV(use_gemv_256));

    task check_gate(input [15:0] d, input [15:0] g, input exp,
                    input [255:0] name);
        begin
            d_in = d; g_in = g; #1;
            if (use_gemv !== exp) begin
                $display("FAIL %0s: drift=%0d gate=%0d -> %b expected %b",
                         name, d, g, use_gemv, exp);
                errors = errors + 1;
            end else begin
                $display("ok   %0s: drift=%0d -> use=%b", name, d, use_gemv);
            end
        end
    endtask

    task check_gate_256(input [15:0] d, input [15:0] g, input exp,
                        input [255:0] name);
        begin
            d_in = d; g_in = g; #1;
            if (use_gemv_256 !== exp) begin
                $display("FAIL %0s: drift=%0d gate=%0d -> %b expected %b",
                         name, d, g, use_gemv_256, exp);
                errors = errors + 1;
            end else begin
                $display("ok   %0s: drift=%0d -> use=%b", name, d, use_gemv_256);
            end
        end
    endtask

    initial begin
        @(posedge clk);

        // 2 rows flipping 32 of 64 bits -> 32/64 = 0.5 -> 32767
        run_case(2, 32, 32767, "half of one comparison row");
        // 2 rows flipping 2 of 64 bits -> 2/64 = 0.03125 -> 2047
        run_case(2, 2, 2047, "0.03125 in band");

        // Block accumulation stays exact well past one codeword's worth of
        // comparisons: 398 rows keeps final_disagree at 796, far under the
        // 65537 wrap point.
        run_case(200, 2, 2047, "0.03125 accumulated over 200 rows");
        // Documented limit, not a supported mode: 3998 comparisons with 32
        // flips each puts final_disagree at 127936, so the 32-bit product
        // wraps and the score reads 15981 instead of 32767. BATCH_END must
        // fire per codeword; this case pins where that requirement comes from.
        run_case(2000, 32, 15981, "32-bit wrap point (documented limit)");

        // Band-pass bounds, with the confidence gate held open.
        check_gate(16'd720,   16'd65535, 1'b0, "just below tau_d");
        check_gate(16'd721,   16'd65535, 1'b1, "exactly tau_d");
        check_gate(16'd2003,  16'd65535, 1'b1, "block-stat max 0.030565");
        check_gate(16'd2490,  16'd65535, 1'b1, "exactly tau_d_max");
        check_gate(16'd2491,  16'd65535, 1'b0, "just above tau_d_max");
        check_gate(16'd3034,  16'd65535, 1'b0, "C3 RTN dominated 0.0463");
        check_gate(16'd13539, 16'd65535, 1'b0, "C4 strong drift 0.2066");
        check_gate(16'd0,     16'd65535, 1'b0, "C1 static spread 0.0");
        // Confidence gate still has veto power inside the band.
        check_gate(16'd2003,  16'd39320, 1'b0, "in band but tau_g closed");

        // Per-codeword granularity: the detector publishes once per codeword,
        // and at that granularity the in-distribution and conductance-domain
        // score distributions overlap. These two cases pin the overlap so the
        // band is not mistaken for a perfect separator: the worst measured
        // in-distribution codeword (0.04147) is rejected, costing 1.05% of
        // codewords their calibration, and the mildest measured C3 codeword
        // (0.02803) is admitted, leaking 7.65% of that channel through.
        check_gate(16'd2717,  16'd65535, 1'b0, "in-dist worst codeword 0.04147");
        check_gate(16'd1837,  16'd65535, 1'b1, "C3 mildest codeword 0.02803");

        // The 256-symbol rate-8/9 configuration carries its own band, because
        // its per-codeword drift score has 2.0x the spread of the 1024-symbol
        // score (s.d. 0.00662 vs 0.00331) -- it averages over fewer symbols.
        // Applying the same mean+2 s.d. rule gives 0.045 rather than 0.038,
        // which is why the bound is per-scale rather than global.
        check_gate_256(16'd327,   16'd65535, 1'b0, "256: just below tau_d");
        check_gate_256(16'd328,   16'd65535, 1'b1, "256: exactly tau_d");
        check_gate_256(16'd2027,  16'd65535, 1'b1, "256: per-cw mean 0.03094");
        check_gate_256(16'd2949,  16'd65535, 1'b1, "256: exactly tau_d_max");
        check_gate_256(16'd2950,  16'd65535, 1'b0, "256: just above tau_d_max");
        check_gate_256(16'd3925,  16'd65535, 1'b0, "256: worst codeword 0.05990");
        check_gate_256(16'd3034,  16'd65535, 1'b0, "256: C3 still rejected");

        if (errors == 0) $display("\nALL CHECKS PASSED");
        else             $display("\n%0d CHECK(S) FAILED", errors);
        $finish;
    end
endmodule
