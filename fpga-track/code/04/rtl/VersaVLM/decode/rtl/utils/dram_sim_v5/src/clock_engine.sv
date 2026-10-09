// Copyright 2023 ETH Zurich and
// University of Bologna

// Solderpad Hardware License
// Version 0.51, see LICENSE for details.

// SPDX-License-Identifier: SHL-0.51

// Author: Chi Zhang <chizhang@iis.ee.ethz.ch>, ETH Zurich
// Date: 07.June.2023


`ifndef VIVADO_XSIM_SIMPLE_DRAM
import "DPI-C" function void run_ns(input int ns);
`endif

// clock engine for dram simulation
module clock_engine #(
	parameter time ClkPeriod	= 1ns
) (
	input logic clk_i,	// Clock
	input logic rst_ni	// Asynchronous reset active low
);

	always_ff @(posedge clk_i or negedge rst_ni) begin : proc_dram_engine
		if(rst_ni) begin
`ifndef VIVADO_XSIM_SIMPLE_DRAM
			// run DRAMsys every clk
			run_ns(int'(ClkPeriod / 1ns));
`endif
		end
	end

endmodule : clock_engine
