// Cycle-count and bit-exactness testbench for GEMV_MLP_CORE width/pipe options.
//
// The same feature vector and anchor are presented for NSYM back-to-back
// symbols. The feature vector is row 0 of pim_predecoder/rtl/cosim/stimulus.txt
// (symbol 0, cycle 0). The anchor is a small hand-written LLR triple.
//
// Configuration is selected on the command line, e.g.
//   iverilog -g2012 -DLANES=8 -DPIPE=1 ...
// Expected outputs default to the measured LANES=1, PIPE=0 reference and can be
// overridden with -DEXP_OUT0=.. -DEXP_OUT1=.. -DEXP_OUT2=.. -DEXP_GATE=..
`timescale 1ns / 1ps

`ifndef LANES
`define LANES 1
`endif
`ifndef PIPE
`define PIPE 0
`endif
`ifndef NSYM
`define NSYM 5
`endif
`ifndef EXP_OUT0
`define EXP_OUT0 8'sd4
`endif
`ifndef EXP_OUT1
`define EXP_OUT1 8'sd27
`endif
`ifndef EXP_OUT2
`define EXP_OUT2 8'sd71
`endif
`ifndef EXP_GATE
`define EXP_GATE 16'd62914
`endif

module GEMV_LANES_TB;
    localparam LANES = `LANES;
    localparam PIPE = `PIPE;
    localparam NSYM = `NSYM;
    localparam FEAT_W = 13 * 8;

    reg clk, rst, start;
    reg signed [FEAT_W - 1:0] feat_in;
    reg signed [23:0] anchor;
    wire done;
    wire signed [23:0] llr_out;
    wire [15:0] gate_q16;

    reg signed [7:0] feat_tab [0:12];

    integer cyc;
    integer k, j;
    integer start_cyc;
    integer done_cyc [0:NSYM - 1];
    integer cold, steady, delta, mism;
    reg [7:0] exp0, exp1, exp2;
    reg [15:0] exp_gate;

    GEMV_MLP_CORE #(
        .LANES(LANES),
        .PIPE(PIPE),
        .WEIGHT_DIR("pim_predecoder/rtl/Weights")
    ) dut (
        .CLK(clk), .RST(rst), .START(start),
        .FEAT_IN(feat_in), .LLR_ANCHOR(anchor),
        .DONE(done), .LLR_OUT(llr_out), .GATE_Q16(gate_q16)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    always @(posedge clk) cyc <= cyc + 1;

    initial begin
        cyc = 0;
        rst = 1;
        start = 0;
        feat_in = 0;
        anchor = 0;
        mism = 0;

        // Row 0 of pim_predecoder/rtl/cosim/stimulus.txt.
        feat_tab[0]  = -8'sd28;
        feat_tab[1]  = -8'sd8;
        feat_tab[2]  =  8'sd1;
        feat_tab[3]  = -8'sd27;
        feat_tab[4]  = -8'sd29;
        feat_tab[5]  =  8'sd19;
        feat_tab[6]  = -8'sd5;
        feat_tab[7]  =  8'sd15;
        feat_tab[8]  =  8'sd22;
        feat_tab[9]  = -8'sd22;
        feat_tab[10] = -8'sd1;
        feat_tab[11] = -8'sd13;
        feat_tab[12] = -8'sd22;

        for (j = 0; j < 13; j = j + 1)
            feat_in[j * 8 +: 8] = feat_tab[j];
        anchor[0 * 8 +: 8] = 8'sd2;
        anchor[1 * 8 +: 8] = -8'sd1;
        anchor[2 * 8 +: 8] = 8'sd3;

        exp0 = `EXP_OUT0;
        exp1 = `EXP_OUT1;
        exp2 = `EXP_OUT2;
        exp_gate = `EXP_GATE;

        repeat (4) @(posedge clk);
        #1 rst = 0;
        @(posedge clk); #1;

        // First symbol: present START for one clock.
        start = 1;
        @(posedge clk); #1;
        start = 0;
        start_cyc = cyc;

        for (k = 0; k < NSYM; k = k + 1) begin
            @(posedge clk); #1;
            while (!done) begin
                @(posedge clk); #1;
            end
            done_cyc[k] = cyc;

            if (llr_out[0 * 8 +: 8] !== exp0 ||
                llr_out[1 * 8 +: 8] !== exp1 ||
                llr_out[2 * 8 +: 8] !== exp2 ||
                gate_q16 !== exp_gate) begin
                mism = mism + 1;
                $display("MISMATCH sym=%0d out0=%0d out1=%0d out2=%0d gate=%0d exp0=%0d exp1=%0d exp2=%0d exp_gate=%0d",
                         k, $signed(llr_out[0 * 8 +: 8]), $signed(llr_out[1 * 8 +: 8]),
                         $signed(llr_out[2 * 8 +: 8]), gate_q16,
                         $signed(exp0), $signed(exp1), $signed(exp2), exp_gate);
            end

            if (k == 0)
                $display("FIRST lanes=%0d pipe=%0d out0=%0d out1=%0d out2=%0d gate=%0d",
                         LANES, PIPE, $signed(llr_out[0 * 8 +: 8]),
                         $signed(llr_out[1 * 8 +: 8]), $signed(llr_out[2 * 8 +: 8]),
                         gate_q16);

            // PIPE=1 restarts from the DONE edge; PIPE=0 needs a new START.
            if (!PIPE && k < NSYM - 1) begin
                start = 1;
                @(posedge clk); #1;
                start = 0;
            end
        end

        cold = done_cyc[0] - start_cyc;
        steady = done_cyc[NSYM - 1] - done_cyc[NSYM - 2];
        for (k = 2; k < NSYM; k = k + 1) begin
            delta = done_cyc[k] - done_cyc[k - 1];
            if (delta != steady) begin
                mism = mism + 1;
                $display("MISMATCH steady interval k=%0d delta=%0d expected=%0d",
                         k, delta, steady);
            end
        end

        $display("RESULT lanes=%0d pipe=%0d cold=%0d steady=%0d out0=%0d out1=%0d out2=%0d gate=%0d",
                 LANES, PIPE, cold, steady,
                 $signed(llr_out[0 * 8 +: 8]), $signed(llr_out[1 * 8 +: 8]),
                 $signed(llr_out[2 * 8 +: 8]), gate_q16);

        if (mism != 0) begin
            $display("MISMATCH lanes=%0d pipe=%0d errors=%0d", LANES, PIPE, mism);
            $fatal(1);
        end
        $display("PASS lanes=%0d pipe=%0d", LANES, PIPE);
        $finish;
    end
endmodule
