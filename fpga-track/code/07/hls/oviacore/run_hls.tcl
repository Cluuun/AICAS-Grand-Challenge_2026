# AICAS 2026 — OViA HLS Synthesis Script
# Target: xck26-sfvc784-2LV-c (Kria K26 SOM) @ 100MHz
# Tool: Vitis HLS 2024.1

open_project ovia_prj
set_top ovia_core

add_files ovia_core.cpp
add_files ovia_core.h
add_files -tb tb_oviacore.cpp

open_solution -reset solution1
set_part xck26-sfvc784-2LV-c

create_clock -period 10 -name default

config_interface -m_axi_addr64
config_interface -m_axi_max_read_burst_length 16
config_interface -m_axi_max_write_burst_length 16
config_compile -pipeline_loops 64
config_compile -name_max_length 256
config_schedule -enable_dsp_full_reg

csim_design
csynth_design

export_design -format ip_catalog -display_name "OViA Accelerator" \
    -description "OCM-anchored Vision Accelerator: 16x16 INT8 systolic array with OCM shared memory bridge" \
    -vendor "yingchu" -version "1.0"

exit
