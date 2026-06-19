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
`ifndef GRC_ECC_SYMBOL_NUM
`define GRC_ECC_SYMBOL_NUM 288
`endif
`ifndef GRC_INFO_NUM
`define GRC_INFO_NUM 256
`endif

module GRC_ECC_CASCADE_TB #(
    parameter N_SYMBOLS = `GRC_N_SYMBOLS,
    parameter N_CYCLES = `GRC_N_CYCLES,
    parameter N_BITS = `GRC_N_BITS,
    parameter SYM_IDX_W = `GRC_SYM_IDX_W,
    parameter CYC_IDX_W = `GRC_CYC_IDX_W,
    parameter ECC_SYMBOL_NUM = `GRC_ECC_SYMBOL_NUM,
    parameter INFO_NUM = `GRC_INFO_NUM,
    parameter ECC_BUS_W = ECC_SYMBOL_NUM * 9,
    parameter FEAT_W = 13 * 8
) ();
    reg clk, rst, ce, cim_e;
    reg [2:0] adc_in;
    reg [SYM_IDX_W - 1:0] sym_idx;
    reg [CYC_IDX_W - 1:0] cyc_idx;
    reg cycle_row_last, batch_last, feat_valid;
    reg signed [FEAT_W - 1:0] feat_in;
    reg [ECC_BUS_W - 1:0] buffer_llr;

    wire use_gemv, batch_done, out_valid, priors_ready;
    wire [SYM_IDX_W - 1:0] out_sym_idx;
    wire signed [23:0] out_llr;
    wire [ECC_BUS_W - 1:0] ecc_input_llr;

    reg grc_valid_d, grc_valid_d2;
    reg [SYM_IDX_W - 1:0] grc_sym_d, grc_sym_d2;
    reg signed [23:0] grc_llr_d, grc_llr_d2;
    reg batch_done_d, batch_done_d2;

    always @(posedge clk) begin
        grc_valid_d <= out_valid;
        grc_valid_d2 <= grc_valid_d;
        grc_sym_d <= out_sym_idx;
        grc_sym_d2 <= grc_sym_d;
        grc_llr_d <= out_llr;
        grc_llr_d2 <= grc_llr_d;
        batch_done_d <= batch_done;
        batch_done_d2 <= batch_done_d;
    end

    integer fd_in, fd_buf, fd_out, n_read, cy_cfg, sym_cfg, total, i, b;
    integer tmp_adc, tmp_row, tmp_batch, tmp_valid;
    integer f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12;

    GRC_PIM_TOP #(
        .N_SYMBOLS(N_SYMBOLS), .N_CYCLES(N_CYCLES), .N_BITS(N_BITS),
        .SYM_IDX_W(SYM_IDX_W), .CYC_IDX_W(CYC_IDX_W),
        .WEIGHT_DIR("pim_predecoder/rtl/Weights")
    ) grc (
        .CLK(clk), .RST(rst), .CIM_E(cim_e), .CE(ce),
        .ADC_IN(adc_in), .SYM_IDX(sym_idx), .CYC_IDX(cyc_idx),
        .CYCLE_ROW_LAST(cycle_row_last), .BATCH_LAST(batch_last),
        .FEAT_IN(feat_in), .FEAT_VALID(feat_valid),
        .CUR_BITS_BUS(), .ACCUM_DBG(), .CYCLE_LLR_DBG(),
        .USE_GEMV(use_gemv), .DRIFT_SCORE_Q16(), .GATE_MEAN_Q16(),
        .OUT_LLR(out_llr), .OUT_VALID(out_valid), .OUT_SYM_IDX(out_sym_idx),
        .OUT_GATE_Q16(), .BATCH_DONE(batch_done)
    );

    GRC_ECC_PRIOR_MUX #(
        .SYMBOL_NUM(ECC_SYMBOL_NUM), .INFO_NUM(INFO_NUM), .SYM_IDX_W(SYM_IDX_W)
    ) mux (
        .CLK(clk), .RST(rst),
        .BUFFER_LLR(buffer_llr),
        .GRC_VALID(grc_valid_d2), .GRC_SYM_IDX(grc_sym_d2), .GRC_LLR_I8(grc_llr_d2),
        .GRC_BATCH_DONE(batch_done_d2), .USE_GRC_LLR(use_gemv),
        .ECC_INPUT_LLR(ecc_input_llr), .PRIORS_READY(priors_ready)
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
        rst = 1; ce = 0; cim_e = 1; feat_valid = 0; feat_in = 0; buffer_llr = 0;
        fd_in = $fopen("pim_predecoder/rtl/cosim_cascade/stimulus.txt", "r");
        fd_buf = $fopen("pim_predecoder/rtl/cosim_cascade/buffer_llr.txt", "r");
        fd_out = $fopen("pim_predecoder/rtl/cosim_cascade/rtl.log", "w");
        for (b = 0; b < ECC_BUS_W; b = b + 1) begin
            n_read = $fscanf(fd_buf, "%d", buffer_llr[b]);
            if (n_read != 1) begin
                $display("ERROR: buffer_llr read failed at bit %0d", b);
                $finish;
            end
        end
        #20 rst = 0;
        n_read = $fscanf(fd_in, "%d %d", sym_cfg, cy_cfg);
        total = sym_cfg * cy_cfg;
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
            ce = 0;
            cycle_row_last = 0;
            batch_last = 0;
        end
        while (!priors_ready) @(posedge clk);
        @(posedge clk);
        $fwrite(fd_out, "PRIORS");
        for (b = 0; b < ECC_BUS_W; b = b + 1)
            $fwrite(fd_out, " %0d", ecc_input_llr[b]);
        $fwrite(fd_out, "\n");
        $fclose(fd_in);
        $fclose(fd_buf);
        $fclose(fd_out);
        $display("Cascade TB done use_gemv=%0d", use_gemv);
        $finish;
    end

    always @(posedge clk) begin
        if (out_valid)
            $fwrite(fd_out, "OUT %0d %0d %0d %0d\n",
                    out_sym_idx,
                    $signed(out_llr[7:0]), $signed(out_llr[15:8]), $signed(out_llr[23:16]));
    end
endmodule
