`ifdef VIVADO_XSIM_SIMPLE_DRAM

module sim_dram #(
  parameter int unsigned DataWidth      = 32'd512,
  parameter int unsigned AddrWidth      = 32'd64,
  parameter longint unsigned BASE       = 64'h80000000,
  parameter              DRAMType       = "DDR4",
  parameter              CustomerDRAM   = "none",
  parameter              InitPath       = "none",
  parameter type         addr_t         = logic [AddrWidth-1:0],
  parameter type         data_t         = logic [DataWidth-1:0],
  parameter type         strb_t         = logic [DataWidth/8-1:0]
)(
    input  logic                 clk_i,
    input  logic                 rst_ni,
    input  logic                 req_valid_i,
    output logic                 req_ready_o,
    input  logic                 we_i,
    input  addr_t                addr_i,
    input  data_t                wdata_i,
    input  strb_t                wstrb_i,
    output logic                 rsp_valid_o,
    input  logic                 rsp_ready_i,
    output data_t                rdata_o,
    output logic                 b_valid_o,
    input  logic                 b_ready_i
);

byte unsigned mem [bit [63:0]];
bit [63:0] req_addr_u;
bit delayed_mode;
int read_latency_cycles;
int write_latency_cycles;
bit pending_valid;
bit pending_we;
int pending_countdown;
data_t pending_rdata;

always_comb begin
    req_addr_u = addr_i;
end

assign req_ready_o = rst_ni && (!delayed_mode || (!pending_valid && !rsp_valid_o && !b_valid_o));

task automatic load_memfile(input bit [63:0] base_addr, input string init_file);
    integer fd;
    integer code;
    integer byte_value;
    bit [63:0] offset;
    begin
        fd = $fopen(init_file, "r");
        if (fd != 0) begin
            offset = 0;
            while (!$feof(fd)) begin
                code = $fscanf(fd, "%h", byte_value);
                if (code == 1) begin
                    mem[base_addr + offset] = byte_value[7:0];
                    offset++;
                end
            end
            $fclose(fd);
        end
    end
endtask

integer addr_list;
integer scan_code;
string init_addr_s;
addr_t init_addr;
string init_file;
string init_path = InitPath;

initial begin
    delayed_mode = $test$plusargs("SIMPLE_DRAM_STALL");
    read_latency_cycles = delayed_mode ? 96 : 0;
    write_latency_cycles = delayed_mode ? 32 : 0;
    scan_code = $value$plusargs("SIMPLE_DRAM_READ_LATENCY=%d", read_latency_cycles);
    scan_code = $value$plusargs("SIMPLE_DRAM_WRITE_LATENCY=%d", write_latency_cycles);

    if (init_path != "none") begin
        scan_code = $value$plusargs("MEM=%s", init_path);
        addr_list = $fopen({init_path, "/dram_load/addr_list.txt"}, "r");
        if (addr_list != 0) begin
            while (!$feof(addr_list)) begin
                scan_code = $fscanf(addr_list, "%s", init_addr_s);
                if (scan_code == 1 && init_addr_s != "") begin
                    scan_code = $sscanf(init_addr_s, "%h", init_addr);
                    init_file = {init_path, "/dram_load/0x", init_addr_s, ".txt"};
                    load_memfile(init_addr, init_file);
                end
            end
            $fclose(addr_list);
        end
    end
end

task load_a_byte_to_dram(input longint dram_addr_ofst, input int data_byte);
    bit [63:0] byte_addr;
    begin
        byte_addr = dram_addr_ofst;
        mem[byte_addr] = data_byte[7:0];
    end
endtask

task check_a_byte_in_dram(input longint dram_addr_ofst, output logic [7:0] data_byte);
    bit [63:0] byte_addr;
    begin
        byte_addr = dram_addr_ofst;
        data_byte = mem.exists(byte_addr) ? mem[byte_addr] : 8'h00;
    end
endtask

task preload_elf_binary(input string elf_binary);
    $display("[XSIM_SIMPLE_DRAM] ELF preload is not supported: %s", elf_binary);
endtask

always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
        rsp_valid_o <= 1'b0;
        b_valid_o <= 1'b0;
        rdata_o <= '0;
        pending_valid <= 1'b0;
        pending_we <= 1'b0;
        pending_countdown <= 0;
        pending_rdata <= '0;
    end else if (!delayed_mode) begin
        if (rsp_valid_o && rsp_ready_i)
            rsp_valid_o <= 1'b0;
        if (b_valid_o && b_ready_i)
            b_valid_o <= 1'b0;

        if (req_valid_i && req_ready_o) begin
            if (we_i) begin
                for (int i = 0; i < (DataWidth / 8); i++)
                    if (wstrb_i[i])
                        mem[req_addr_u + i] = wdata_i[8*i +: 8];
                b_valid_o <= 1'b1;
            end else begin
                for (int i = 0; i < (DataWidth / 8); i++)
                    rdata_o[8*i +: 8] <= mem.exists(req_addr_u + i) ? mem[req_addr_u + i] : 8'h00;
                rsp_valid_o <= 1'b1;
            end
        end
    end else begin
        if (rsp_valid_o && rsp_ready_i)
            rsp_valid_o <= 1'b0;
        if (b_valid_o && b_ready_i)
            b_valid_o <= 1'b0;

        if (pending_valid && !rsp_valid_o && !b_valid_o) begin
            if (pending_countdown > 0) begin
                pending_countdown <= pending_countdown - 1;
            end else begin
                pending_valid <= 1'b0;
                if (pending_we) begin
                    b_valid_o <= 1'b1;
                end else begin
                    rdata_o <= pending_rdata;
                    rsp_valid_o <= 1'b1;
                end
            end
        end

        if (req_valid_i && req_ready_o) begin
            pending_valid <= 1'b1;
            pending_we <= we_i;
            pending_countdown <= we_i ? write_latency_cycles : read_latency_cycles;
            if (we_i) begin
                for (int i = 0; i < (DataWidth / 8); i++)
                    if (wstrb_i[i])
                        mem[req_addr_u + i] = wdata_i[8*i +: 8];
                pending_rdata <= '0;
            end else begin
                for (int i = 0; i < (DataWidth / 8); i++)
                    pending_rdata[8*i +: 8] <= mem.exists(req_addr_u + i) ? mem[req_addr_u + i] : 8'h00;
            end
        end
    end
end

endmodule : sim_dram

`else

// Copyright 2023 ETH Zurich and
// University of Bologna

// Solderpad Hardware License
// Version 0.51, see LICENSE for details.

// SPDX-License-Identifier: SHL-0.51

// Author: Chi Zhang <chizhang@iis.ee.ethz.ch>, ETH Zurich
// Date: 07.June.2023

// dram model using dramsys library
import "DPI-C" function int add_dram(input string resources_path, input string simulationJson_path, input longint dram_base_addr);
import "DPI-C" function int dram_get_inflight_read(input int dram_id);
import "DPI-C" function int dram_can_accept_req(input int dram_id);
import "DPI-C" function int dram_has_write_rsp(input int dram_id);
import "DPI-C" function int dram_get_write_rsp(input int dram_id);
import "DPI-C" function int dram_has_read_rsp(input int dram_id);
import "DPI-C" function void dram_write_buffer(input int dram_id, input int byte_int, input int idx);
import "DPI-C" function void dram_write_strobe(input int dram_id, input int strob_int, input int idx);
import "DPI-C" function void dram_send_req(input int dram_id, input longint addr, input longint length , input longint is_write, input longint strob_enable);
import "DPI-C" function void dram_get_read_rsp(input int dram_id, input longint length, inout byte buffer[]);
import "DPI-C" function int dram_get_read_rsp_byte(input int dram_id);
import "DPI-C" function int dram_peek_read_rsp_byte(input int dram_id, input int idx);
import "DPI-C" function void dram_preload_byte(input int dram_id, input longint dram_addr_ofst, input int byte_int);
import "DPI-C" function int dram_check_byte(input int dram_id, input longint dram_addr_ofst);
import "DPI-C" function void dram_load_elf(input string app_path);
import "DPI-C" function void dram_load_memfile(input int dram_id, input longint addr_ofst, input string init_path);
import "DPI-C" function void close_dram(input int dram_id);


module sim_dram #(
  parameter int unsigned DataWidth      = 32'd512,  // Data signal width
  parameter int unsigned AddrWidth      = 32'd64,    // Addr signal width
  parameter longint unsigned BASE       = 64'h80000000, // DRAM Base addr
  parameter              DRAMType       = "DDR4",   //DRAM type
  parameter              CustomerDRAM   = "none",   //DRAM type
  parameter              InitPath       = "none",   //mem path
  // DEPENDENT PARAMETERS, DO NOT OVERWRITE!
  parameter type         addr_t         = logic [AddrWidth-1:0],
  parameter type         data_t         = logic [DataWidth-1:0],
  parameter type         strb_t         = logic [DataWidth/8-1:0]
)(
    input  logic                 clk_i,      // Clock
    input  logic                 rst_ni,     // Asynchronous reset active low
    // requests ports
    input  logic                 req_valid_i,// request valid
    output logic                 req_ready_o,// request ready
    input  logic                 we_i,       // write enable
    input  addr_t                addr_i,     // request address
    input  data_t                wdata_i,    // write data
    input  strb_t                wstrb_i,    // write strb
    // response ports
    output logic                 rsp_valid_o,
    input  logic                 rsp_ready_i,
    output data_t                rdata_o,     // read data
    // write response
    output logic                 b_valid_o,
    input  logic                 b_ready_i
);

// always_ff @( posedge clk_i or negedge rst_ni ) begin
//         if (~rst_ni) begin
//         end
//         else begin
//             if (req_valid_i & req_ready_o &~we_i) begin
//                 $display("read addr : %h", addr_i);
//             end
//         end
//     end

// always_ff @( posedge clk_i or negedge rst_ni ) begin
//         if (~rst_ni) begin
//         end
//         else begin
//             if (rsp_valid_o & rsp_ready_i) begin
//                 $display("read data : %h", rdata_o);
//             end
//         end
//     end


typedef logic [7:0] my_byte_t;

int dram_id;

string resources_path = `DRAMSYS_LIB;
string simulationJson_path;
string app_path;
string init_path = InitPath;

//create model
initial begin
    // void'($value$plusargs("DRAMSYS_RES=%s", resources_path));
    case (DRAMType)
        "DDR4":  simulationJson_path = {resources_path, "/ddr4-example.json"} ;
        "DDR3":  simulationJson_path = {resources_path, "/ddr3-example.json"};
        "HBM2":  simulationJson_path = {resources_path, "/hbm2-example.json"};
        "LPDDR4":  simulationJson_path = {resources_path, "/lpddr4-example.json"};
        default:  simulationJson_path = {resources_path, "/ddr4-example.json"};
    endcase

    if (CustomerDRAM != "none") begin
        simulationJson_path = {resources_path, "/", CustomerDRAM, ".json"};
        $display("[DRAMSys] Use Customer DRAM configuration: %s",simulationJson_path);
    end

    $display("[DRAMSys] resources_path=%s", resources_path);
    $display("[DRAMSys] simulationJson_path=%s", simulationJson_path);
    if (resources_path.len() == 0 || simulationJson_path.len() == 0) begin
        $fatal(1,"[DRAMSys] no DRAMsys configuration found!");
    end
    dram_id = add_dram(resources_path, simulationJson_path, BASE);
    void'($value$plusargs("ONE_DRAM_PRELOAD=%s", app_path));
    if (app_path.len() != 0) begin
        $display("[DRAMSys] Preloading elf: %s\n", app_path);
        dram_load_elf(app_path);
    end
end

integer addr_list;
string init_addr_s;
addr_t init_addr;
string init_file;
// initialize dram
initial begin
    
    if (init_path != "none") begin
        void'($value$plusargs("MEM=%s", init_path));

        addr_list = $fopen({init_path, "/dram_load/addr_list.txt"}, "r");
        
        while (1) begin
            void'($fscanf(addr_list,"%s",init_addr_s));
            if ($feof(addr_list)) break;

            void'($sscanf(init_addr_s, "%h", init_addr));
            
            if (init_addr_s != "") begin
                init_file = {init_path, "/dram_load/0x", init_addr_s, ".txt"};
                dram_load_memfile(dram_id, init_addr, init_file);
                // $display("[DRAMSys] Preloading mem: %s\n", init_file);
            end
        end
        
        $fclose(addr_list);
        
    end
end

//interface to manualy modify DRAM
task load_a_byte_to_dram(input longint dram_addr_ofst, input int data_byte );
    dram_preload_byte(dram_id, dram_addr_ofst, data_byte);
endtask

//interface to check a byte in DRAM
task check_a_byte_in_dram(input longint dram_addr_ofst, output logic[7:0] data_byte );
    automatic int byte_int;
    byte_int = dram_check_byte(dram_id, dram_addr_ofst);
    data_byte = byte_int;
endtask

//interface to manualy modify DRAM
task preload_elf_binary(input string elf_binary );
    dram_load_elf(elf_binary);
endtask

always_ff @(posedge clk_i or negedge rst_ni) begin : proc_dram
    if(~rst_ni) begin
        req_ready_o <= 1'b0;
        rsp_valid_o <= 1'b0;
        b_valid_o <= 1'b0;
        rdata_o <= '0;
    end else begin
        // Default assignments
        rsp_valid_o <= 1'b0;
        b_valid_o <= 1'b0;
        req_ready_o <= 1'b0;

        // Request
        if (req_valid_i & req_ready_o) begin
            for (int i = 0; i < (DataWidth/8); i++) begin
                dram_write_buffer(dram_id, wdata_i[8*i +: 8], i);
                dram_write_strobe(dram_id, wstrb_i[i], i);
            end
            dram_send_req(dram_id, longint'(addr_i), (DataWidth/8), longint'(we_i), !(&wstrb_i) && we_i);
        end

        if (dram_can_accept_req(dram_id)) begin
            req_ready_o <= 1'b1;
        end

        // Read response
        if (rsp_valid_o & rsp_ready_i) begin
            for (int i = 0; i < (DataWidth/8); i++) begin
                void'(dram_get_read_rsp_byte(dram_id));
            end
        end

        if (dram_has_read_rsp(dram_id)) begin
            rsp_valid_o <= 1'b1;
            for (int i = 0; i < (DataWidth/8); i++) begin
                rdata_o[8*i +: 8] <= dram_peek_read_rsp_byte(dram_id, i);
            end
        end

        // Write response
        if (b_valid_o & b_ready_i) begin
            void'(dram_get_write_rsp(dram_id));
        end

        if (dram_has_write_rsp(dram_id)) begin
            b_valid_o <= 1;
        end
    end
end

final begin
    close_dram(dram_id);
end


endmodule : sim_dram

`endif
