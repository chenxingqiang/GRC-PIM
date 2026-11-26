// Throughput and bit-exactness testbench for GEMV_MLP_PIPE.
//
// Phase A runs NSYM symbols through the delivered sequential reference,
// GEMV_MLP_CORE with LANES=1 and PIPE=0, and records LLR_OUT / GATE_Q16.
// Phase B streams the same NSYM symbols back to back into GEMV_MLP_PIPE and
// compares every completed symbol against the recorded reference.
//
// The feature vectors are the eight distinct rows of symbol 0 in
// pim_predecoder/rtl/cosim/stimulus.txt (cycles 0..7); the anchors are a small
// per-symbol pattern so the LLR add is exercised with varying signs.
//
// Testbench consistency check: the eight symbols use distinct anchors and
// features, so a mis-ordered or dropped pipeline slot is detected.
//
// Configuration is selected on the command line, e.g.
//   iverilog -g2012 -DROWS_PER_CYCLE=15 -DNSYM=8 ...
//
// Output:
//   PIPE_RESULT rows_per_cycle=<n> cold=<n> steady=<n> checks=ok|fail
`timescale 1ns / 1ps

`ifndef ROWS_PER_CYCLE
`define ROWS_PER_CYCLE 15
`endif
`ifndef NSYM
`define NSYM 8
`endif

module GEMV_PIPE_TB;
    localparam RPC = `ROWS_PER_CYCLE;
    localparam NSYM = `NSYM;
    localparam HIDDEN = 24;
    localparam OUT_NUM = 3;
    localparam FEAT_NUM = 13;
    localparam FC1_CYCLES = (HIDDEN + RPC - 1) / RPC;
    localparam FEAT_W = FEAT_NUM * 8;
    localparam WEIGHT_DIR = "pim_predecoder/rtl/Weights";

    reg clk, rst;
    reg pipe_start, ref_start;
    reg signed [FEAT_W - 1:0] pipe_feat, ref_feat;
    reg signed [OUT_NUM * 8 - 1:0] pipe_anchor, ref_anchor;
    wire pipe_done, ref_done;
    wire signed [OUT_NUM * 8 - 1:0] pipe_llr, ref_llr;
    wire [15:0] pipe_gate, ref_gate;

    reg signed [7:0] feat_tab [0:NSYM - 1][0:FEAT_NUM - 1];
    reg signed [7:0] anchor_tab [0:NSYM - 1][0:OUT_NUM - 1];
    reg signed [OUT_NUM * 8 - 1:0] exp_llr [0:NSYM - 1];
    reg [15:0] exp_gate [0:NSYM - 1];
    integer done_cyc [0:NSYM - 1];

    integer cyc;
    integer start_cyc, cold, steady, mism, ndone;
    integer k, j, c;
    integer fd, rc, sym_rd, cyc_rd, dummy;

    GEMV_MLP_PIPE #(
        .ROWS_PER_CYCLE(RPC),
        .FEAT_NUM(FEAT_NUM),
        .HIDDEN(HIDDEN),
        .OUT_NUM(OUT_NUM),
        .WEIGHT_DIR(WEIGHT_DIR)
    ) PIPE (
        .CLK(clk), .RST(rst), .START(pipe_start),
        .FEAT_IN(pipe_feat), .LLR_ANCHOR(pipe_anchor),
        .DONE(pipe_done), .LLR_OUT(pipe_llr), .GATE_Q16(pipe_gate)
    );

    GEMV_MLP_CORE #(
        .FEAT_NUM(FEAT_NUM),
        .HIDDEN(HIDDEN),
        .OUT_NUM(OUT_NUM),
        .WEIGHT_DIR(WEIGHT_DIR),
        .LANES(1),
        .PIPE(0)
    ) REF (
        .CLK(clk), .RST(rst), .START(ref_start),
        .FEAT_IN(ref_feat), .LLR_ANCHOR(ref_anchor),
        .DONE(ref_done), .LLR_OUT(ref_llr), .GATE_Q16(ref_gate)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    always @(posedge clk) cyc <= cyc + 1;

    // Compare the symbol that just completed against the recorded reference.
    task check_done;
        begin
            done_cyc[ndone] = cyc;
            if (pipe_llr !== exp_llr[ndone] || pipe_gate !== exp_gate[ndone]) begin
                mism = mism + 1;
                $display("MISMATCH sym=%0d out0=%0d out1=%0d out2=%0d gate=%0d exp0=%0d exp1=%0d exp2=%0d exp_gate=%0d",
                         ndone,
                         $signed(pipe_llr[0 * 8 +: 8]),
                         $signed(pipe_llr[1 * 8 +: 8]),
                         $signed(pipe_llr[2 * 8 +: 8]),
                         pipe_gate,
                         $signed(exp_llr[ndone][0 * 8 +: 8]),
                         $signed(exp_llr[ndone][1 * 8 +: 8]),
                         $signed(exp_llr[ndone][2 * 8 +: 8]),
                         exp_gate[ndone]);
            end
            ndone = ndone + 1;
        end
    endtask

    initial begin
        cyc = 0;
        rst = 1;
        pipe_start = 0;
        ref_start = 0;
        pipe_feat = 0;
        ref_feat = 0;
        pipe_anchor = 0;
        ref_anchor = 0;
        mism = 0;
        ndone = 0;

        // Eight feature vectors: rows cyc=0..7 of symbol 0 in the cosim
        // stimulus file. Each row is: sym cyc x y f0..f12 t0 t1.
        fd = $fopen("pim_predecoder/rtl/cosim/stimulus.txt", "r");
        if (fd == 0)
            $fatal(1, "cannot open pim_predecoder/rtl/cosim/stimulus.txt");
        rc = $fscanf(fd, "%d", dummy);   // N_SYMBOLS
        rc = $fscanf(fd, "%d", dummy);   // N_CYCLES
        for (k = 0; k < NSYM; k = k + 1) begin
            rc = $fscanf(fd, "%d", sym_rd);
            rc = $fscanf(fd, "%d", cyc_rd);
            rc = $fscanf(fd, "%d", dummy);
            rc = $fscanf(fd, "%d", dummy);
            if (sym_rd != 0 || cyc_rd != k)
                $fatal(1, "stimulus order: expected symbol 0 cycle %0d, got %0d/%0d",
                       k, sym_rd, cyc_rd);
            for (j = 0; j < FEAT_NUM; j = j + 1) begin
                rc = $fscanf(fd, "%d", dummy);
                feat_tab[k][j] = dummy;
            end
            for (j = 0; j < 2; j = j + 1)
                rc = $fscanf(fd, "%d", dummy);
        end
        $fclose(fd);

        for (k = 0; k < NSYM; k = k + 1) begin
            anchor_tab[k][0] =  2 + k;
            anchor_tab[k][1] = -1 - k;
            anchor_tab[k][2] =  3 + 2 * k;
        end

        repeat (4) @(posedge clk);
        #1 rst = 0;
        @(posedge clk); #1;

        // ---- Phase A: sequential reference outputs -------------------
        for (k = 0; k < NSYM; k = k + 1) begin
            for (j = 0; j < FEAT_NUM; j = j + 1)
                ref_feat[j * 8 +: 8] = feat_tab[k][j];
            for (j = 0; j < OUT_NUM; j = j + 1)
                ref_anchor[j * 8 +: 8] = anchor_tab[k][j];
            ref_start = 1;
            @(posedge clk); #1;
            ref_start = 0;
            @(posedge clk); #1;
            while (!ref_done) begin
                @(posedge clk); #1;
            end
            exp_llr[k] = ref_llr;
            exp_gate[k] = ref_gate;
        end

        // ---- Phase B: back-to-back symbols through the pipeline ------
        rst = 1;
        repeat (4) @(posedge clk);
        #1 rst = 0;
        @(posedge clk); #1;

        ndone = 0;
        for (j = 0; j < FEAT_NUM; j = j + 1)
            pipe_feat[j * 8 +: 8] = feat_tab[0][j];
        for (j = 0; j < OUT_NUM; j = j + 1)
            pipe_anchor[j * 8 +: 8] = anchor_tab[0][j];

        pipe_start = 1;
        @(posedge clk); #1;
        pipe_start = 0;
        start_cyc = cyc;

        // A new symbol enters every FC1_CYCLES cycles, whether or not a DONE
        // has been seen, which is what "back to back" means here.
        for (k = 0; k < NSYM; k = k + 1) begin
            if (k > 0) begin
                for (j = 0; j < FEAT_NUM; j = j + 1)
                    pipe_feat[j * 8 +: 8] = feat_tab[k][j];
                for (j = 0; j < OUT_NUM; j = j + 1)
                    pipe_anchor[j * 8 +: 8] = anchor_tab[k][j];
            end
            for (c = 0; c < FC1_CYCLES; c = c + 1) begin
                @(posedge clk); #1;
                if (pipe_done)
                    check_done;
            end
        end
        while (ndone < NSYM) begin
            @(posedge clk); #1;
            if (pipe_done)
                check_done;
        end

        cold = done_cyc[0] - start_cyc;
        steady = done_cyc[NSYM - 1] - done_cyc[NSYM - 2];
        for (k = 2; k < NSYM; k = k + 1) begin
            if (done_cyc[k] - done_cyc[k - 1] != steady) begin
                mism = mism + 1;
                $display("MISMATCH steady interval k=%0d delta=%0d expected=%0d",
                         k, done_cyc[k] - done_cyc[k - 1], steady);
            end
        end

        if (mism == 0) begin
            $display("PIPE_RESULT rows_per_cycle=%0d cold=%0d steady=%0d checks=ok",
                     RPC, cold, steady);
        end else begin
            $display("PIPE_RESULT rows_per_cycle=%0d cold=%0d steady=%0d checks=fail",
                     RPC, cold, steady);
            $fatal(1);
        end
        $finish;
    end

    // Simulation watchdog.
    initial begin
        #5000000;
        $display("MISMATCH timeout");
        $fatal(1);
    end
endmodule
