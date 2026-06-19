`timescale 1ns / 1ps

// Standalone LUT + B' accumulate test (matches hw_llr.py / simulate_grc_pim.py).
module GRC_LLR_LUT_TB;
    reg [2:0] adc;
    reg cim_e;
    wire signed [8:0] llr;

    GRC_MESSAGE_TRANS #(.INPUT_BIT(3), .LLR_BIT(3), .FIELD(3)) dut (
        .INPUT(adc), .CIM_E(cim_e), .OUTPUT_LLR(llr)
    );

    integer pass, fail;
    reg signed [2:0] e0, e1, e2;

    task check;
        input [2:0] word;
        input integer exp0_i, exp1_i, exp2_i;
        begin
            adc = word;
            #1;
            e0 = $signed(llr[2:0]);
            e1 = $signed(llr[5:3]);
            e2 = $signed(llr[8:6]);
            if (e0 == exp0_i && e1 == exp1_i && e2 == exp2_i) begin
                pass = pass + 1;
                $display("PASS adc=%b -> [%0d,%0d,%0d]", word, e0, e1, e2);
            end else begin
                fail = fail + 1;
                $display("FAIL adc=%b got [%0d,%0d,%0d] expect [%0d,%0d,%0d]",
                         word, e0, e1, e2, exp0_i, exp1_i, exp2_i);
            end
        end
    endtask

    initial begin
        pass = 0; fail = 0; cim_e = 1;
        check(3'b010, 1, 1, 2);
        check(3'b011, 2, 1, 1);
        check(3'b001, 1, 2, 1);
        $display("LUT summary: %0d pass, %0d fail", pass, fail);
        if (fail != 0) $fatal(1, "LUT mismatch");
        $finish;
    end
endmodule
