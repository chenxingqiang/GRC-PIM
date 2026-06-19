`timescale 1ns / 1ps

`ifndef GRC_N_SYMBOLS
`define GRC_N_SYMBOLS 8
`endif
`ifndef GRC_N_CYCLES
`define GRC_N_CYCLES 4
`endif
`ifndef GRC_N_BITS
`define GRC_N_BITS (`GRC_N_SYMBOLS * 2)
`endif
`ifndef GRC_SYM_IDX_W
`define GRC_SYM_IDX_W 4
`endif
`ifndef GRC_CYC_IDX_W
`define GRC_CYC_IDX_W 4
`endif

module GRC_PIM_TOP_COSIM_TB #(
    parameter N_SYMBOLS = `GRC_N_SYMBOLS,
    parameter N_CYCLES = `GRC_N_CYCLES,
    parameter N_BITS = `GRC_N_BITS,
    parameter SYM_IDX_W = `GRC_SYM_IDX_W,
    parameter CYC_IDX_W = `GRC_CYC_IDX_W,
    parameter FEAT_W = 13 * 8
) ();
    reg clk, rst, ce, cim_e;
    reg [2:0] adc_in;
    reg [SYM_IDX_W - 1:0] sym_idx;
    reg [CYC_IDX_W - 1:0] cyc_idx;
    reg cycle_row_last, batch_last, feat_valid;
    reg signed [FEAT_W - 1:0] feat_in;

    wire [N_BITS - 1:0] cur_bits;
    wire signed [23:0] accum_dbg;
    wire signed [8:0] cycle_llr_dbg;
    wire use_gemv;
    wire [15:0] drift_q16, gate_mean_q16, out_gate_q16;
    wire signed [23:0] out_llr;
    wire out_valid;
    wire [SYM_IDX_W - 1:0] out_sym_idx;
    wire batch_done;

    integer fd_in, fd_out, n_read, cy_cfg, sym_cfg, total, i;
    integer tmp_adc, tmp_row, tmp_batch, tmp_valid;
    integer f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12;

    GRC_PIM_TOP #(
        .N_SYMBOLS(N_SYMBOLS), .N_CYCLES(N_CYCLES), .N_BITS(N_BITS),
        .SYM_IDX_W(SYM_IDX_W), .CYC_IDX_W(CYC_IDX_W),
        .WEIGHT_DIR("pim_predecoder/rtl/Weights")
    ) dut (
        .CLK(clk), .RST(rst), .CIM_E(cim_e), .CE(ce),
        .ADC_IN(adc_in), .SYM_IDX(sym_idx), .CYC_IDX(cyc_idx),
        .CYCLE_ROW_LAST(cycle_row_last), .BATCH_LAST(batch_last),
        .FEAT_IN(feat_in), .FEAT_VALID(feat_valid),
        .CUR_BITS_BUS(cur_bits), .ACCUM_DBG(accum_dbg),
        .CYCLE_LLR_DBG(cycle_llr_dbg), .USE_GEMV(use_gemv),
        .DRIFT_SCORE_Q16(drift_q16), .GATE_MEAN_Q16(gate_mean_q16),
        .OUT_LLR(out_llr), .OUT_VALID(out_valid), .OUT_SYM_IDX(out_sym_idx),
        .OUT_GATE_Q16(out_gate_q16), .BATCH_DONE(batch_done)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    task load_feat_dec;
        input integer f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12;
        begin
            feat_in[0  * 8 +: 8] = f0;
            feat_in[1  * 8 +: 8] = f1;
            feat_in[2  * 8 +: 8] = f2;
            feat_in[3  * 8 +: 8] = f3;
            feat_in[4  * 8 +: 8] = f4;
            feat_in[5  * 8 +: 8] = f5;
            feat_in[6  * 8 +: 8] = f6;
            feat_in[7  * 8 +: 8] = f7;
            feat_in[8  * 8 +: 8] = f8;
            feat_in[9  * 8 +: 8] = f9;
            feat_in[10 * 8 +: 8] = f10;
            feat_in[11 * 8 +: 8] = f11;
            feat_in[12 * 8 +: 8] = f12;
        end
    endtask

    initial begin
        rst = 1; ce = 0; cim_e = 1; feat_valid = 0; feat_in = 0;
        fd_in = $fopen("pim_predecoder/rtl/cosim/stimulus.txt", "r");
        fd_out = $fopen("pim_predecoder/rtl/cosim/rtl.log", "w");
        if (fd_in == 0) begin
            $display("ERROR: cannot open stimulus.txt");
            $finish;
        end
        n_read = $fscanf(fd_in, "%d %d", sym_cfg, cy_cfg);
        total = sym_cfg * cy_cfg;
        #20 rst = 0;
        for (i = 0; i < total; i = i + 1) begin
            n_read = $fscanf(fd_in, "%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d",
                             cyc_idx, sym_idx, tmp_adc, tmp_valid,
                             f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12,
                             tmp_row, tmp_batch);
            load_feat_dec(f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12);
            adc_in = tmp_adc[2:0];
            feat_valid = tmp_valid;
            cycle_row_last = tmp_row;
            batch_last = tmp_batch;
            ce = 1;
            @(posedge clk);
            #1;
            $fwrite(fd_out, "CYCLE %0d %0d %0d %0d %0d %0d %0d %0d %0d\n",
                    cyc_idx, sym_idx, adc_in,
                    $signed(cycle_llr_dbg[2:0]), $signed(cycle_llr_dbg[5:3]), $signed(cycle_llr_dbg[8:6]),
                    $signed(accum_dbg[7:0]), $signed(accum_dbg[15:8]), $signed(accum_dbg[23:16]));
            ce = 0;
            cycle_row_last = 0;
            batch_last = 0;
        end
        while (!batch_done) @(posedge clk);
        @(posedge clk);
        $fwrite(fd_out, "BATCH %0d %0d %0d\n", drift_q16, gate_mean_q16, use_gemv);
        $fclose(fd_in);
        $fclose(fd_out);
        $display("Cosim TB done drift=%0d gate=%0d use=%0d", drift_q16, gate_mean_q16, use_gemv);
        $finish;
    end

    always @(posedge clk) begin
        if (out_valid)
            $fwrite(fd_out, "OUT %0d %0d %0d %0d %0d\n",
                    out_sym_idx,
                    $signed(out_llr[7:0]), $signed(out_llr[15:8]), $signed(out_llr[23:16]),
                    out_gate_q16);
    end
endmodule
