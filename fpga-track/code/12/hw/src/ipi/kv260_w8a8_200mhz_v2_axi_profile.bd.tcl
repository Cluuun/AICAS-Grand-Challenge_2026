################################################################
# Profile-enabled variant of kv260_w8a8_200mhz_v2.bd.tcl
#
# Inserts the lightweight AXI profile wrapper around the linear
# accelerator HLS kernel. The original block design script is left
# untouched.
################################################################

namespace eval _tcl {
proc get_script_folder {} {
   set script_path [file normalize [info script]]
   return [file dirname $script_path]
}
}
variable script_folder
set script_folder [_tcl::get_script_folder]

set list_projs [get_projects -quiet]
if { $list_projs eq "" } {
   set existing_project [file normalize [file join [pwd] myproj project_1.xpr]]
   if {[file exists $existing_project]} {
      open_project $existing_project
   } else {
      create_project project_1 myproj -part xck26-sfvc784-2LV-c
      set_property BOARD_PART xilinx.com:kv260_som:part0:1.4 [current_project]
   }
}

set hls_ip_repo [file normalize [file join $script_folder ../hls/vlm_w8a8/smolvlm_prefill_unified/linear_accelerator_kv260/solution1/impl/ip]]
if {[file exists [file join $hls_ip_repo component.xml]]} {
   set ip_repo_paths [get_property ip_repo_paths [current_project]]
   if {[lsearch -exact $ip_repo_paths $hls_ip_repo] < 0} {
      set_property ip_repo_paths [concat $ip_repo_paths [list $hls_ip_repo]] [current_project]
      update_ip_catalog
   }
}

set wrapper_pkg_script [file normalize [file join $script_folder ../rtl/package_smolvlm_axi_profile_wrapper.tcl]]
if {![file exists $wrapper_pkg_script]} {
   error "Missing wrapper package script: $wrapper_pkg_script"
}
source $wrapper_pkg_script

if {[info exists SMOLVLM_AXI_PROFILE_WRAPPER_REPO] && [file exists [file join $SMOLVLM_AXI_PROFILE_WRAPPER_REPO component.xml]]} {
   set ip_repo_paths [get_property ip_repo_paths [current_project]]
   if {[lsearch -exact $ip_repo_paths $SMOLVLM_AXI_PROFILE_WRAPPER_REPO] < 0} {
      set_property ip_repo_paths [concat $ip_repo_paths [list $SMOLVLM_AXI_PROFILE_WRAPPER_REPO]] [current_project]
   }
   if {[catch {update_ip_catalog -rebuild}]} {
      update_ip_catalog
   }
} else {
   error "Failed to package smolvlm_axi_profile_wrapper"
}

set base_script [file join $script_folder kv260_w8a8_200mhz_v2.bd.tcl]
if {![file exists $base_script]} {
   error "Missing base BD script: $base_script"
}
source $base_script

if {[current_bd_design -quiet] eq ""} {
   set existing_bd [get_files -quiet smolvlm_kv260_bd.bd]
   if {$existing_bd ne ""} {
      open_bd_design $existing_bd
      current_bd_design smolvlm_kv260_bd
   }
}

namespace eval axi_profile_bd_patch {
proc disconnect_intf_pin_if_connected {pin_name} {
    set pin [get_bd_intf_pins -quiet $pin_name]
    if {$pin eq ""} {
        return
    }
    set net [get_bd_intf_nets -quiet -of_objects $pin]
    if {$net ne ""} {
        disconnect_bd_intf_net $net $pin
    }
}

proc clear_addr_space {space_name} {
    set space [get_bd_addr_spaces -quiet $space_name]
    if {$space eq ""} {
        return
    }
    set segs [get_bd_addr_segs -quiet -of_objects $space]
    if {$segs ne ""} {
        delete_bd_objs $segs
    }
}

proc delete_addr_seg_if_exists {seg_name} {
    set seg [get_bd_addr_segs -quiet $seg_name]
    if {$seg ne ""} {
        delete_bd_objs $seg
    }
}

proc design_clk_freq_hz {} {
    set freq ""
    set hls_clk [get_bd_pins -quiet smolvlm_prefill_unified_0/ap_clk]
    if {$hls_clk ne ""} {
        set freq [get_property FREQ_HZ $hls_clk]
    }
    if {$freq eq ""} {
        set ps_clk [get_bd_pins -quiet zynq_ultra_ps_e_0/pl_clk0]
        if {$ps_clk ne ""} {
            set freq [get_property FREQ_HZ $ps_clk]
        }
    }
    if {$freq eq ""} {
        set freq 199998001
    }
    return $freq
}

proc set_pin_freq_hz_if_exists {pin_name freq} {
    set pin [get_bd_pins -quiet $pin_name]
    if {$pin ne ""} {
        set_property FREQ_HZ $freq $pin
    }
}

proc set_intf_freq_hz_if_exists {intf_name freq} {
    set intf [get_bd_intf_pins -quiet $intf_name]
    if {$intf ne ""} {
        set_property FREQ_HZ $freq $intf
    }
}

proc add_smolvlm_axi_profile_wrapper {} {
    set wrapper_vlnv "skyward.com:user:smolvlm_axi_profile_wrapper:1.0"
    if {[llength [get_ipdefs -all $wrapper_vlnv]] == 0} {
        error "SmolVLM AXI profile wrapper IP not found in catalog: $wrapper_vlnv"
    }

    current_bd_instance /

    if {[get_bd_cells -quiet smolvlm_axi_profile_wrapper_0] eq ""} {
        create_bd_cell -type ip -vlnv $wrapper_vlnv smolvlm_axi_profile_wrapper_0
    }

    set wrapper [get_bd_cells smolvlm_axi_profile_wrapper_0]
    set_property -dict [list \
        CONFIG.C_S_AXI_CONTROL_ADDR_WIDTH {7} \
        CONFIG.C_M_AXI_CONTROL_ADDR_WIDTH {7} \
        CONFIG.C_AXI_GMEM_DATA_WIDTH {128} \
        CONFIG.C_AXI_GMEM_WSTRB_WIDTH {16} \
    ] $wrapper

    set_property -dict [list CONFIG.NUM_MI {2}] [get_bd_cells smartconnect_ctrl]

    disconnect_intf_pin_if_connected smolvlm_prefill_unified_0/s_axi_control
    connect_bd_intf_net -intf_net smartconnect_ctrl_M00_AXI \
        [get_bd_intf_pins smartconnect_ctrl/M00_AXI] \
        [get_bd_intf_pins smolvlm_axi_profile_wrapper_0/s_axi_control]
    connect_bd_intf_net -intf_net smolvlm_axi_profile_wrapper_0_m_axi_control \
        [get_bd_intf_pins smolvlm_axi_profile_wrapper_0/m_axi_control] \
        [get_bd_intf_pins smolvlm_prefill_unified_0/s_axi_control]
    connect_bd_intf_net -intf_net smartconnect_ctrl_M01_AXI \
        [get_bd_intf_pins smartconnect_ctrl/M01_AXI] \
        [get_bd_intf_pins smolvlm_axi_profile_wrapper_0/s_axi_profile]

    disconnect_intf_pin_if_connected smolvlm_prefill_unified_0/m_axi_gmem_d
    connect_bd_intf_net -intf_net smolvlm_prefill_unified_0_m_axi_gmem_d \
        [get_bd_intf_pins smolvlm_axi_profile_wrapper_0/m_axi_gmem_d] \
        [get_bd_intf_pins smartconnect_data/S00_AXI]
    connect_bd_intf_net -intf_net smolvlm_prefill_unified_0_to_profile_gmem_d \
        [get_bd_intf_pins smolvlm_prefill_unified_0/m_axi_gmem_d] \
        [get_bd_intf_pins smolvlm_axi_profile_wrapper_0/s_axi_gmem_d]

    disconnect_intf_pin_if_connected smolvlm_prefill_unified_0/m_axi_gmem_w
    connect_bd_intf_net -intf_net smolvlm_prefill_unified_0_m_axi_gmem_w \
        [get_bd_intf_pins smolvlm_axi_profile_wrapper_0/m_axi_gmem_w] \
        [get_bd_intf_pins smartconnect_weight/S00_AXI]
    connect_bd_intf_net -intf_net smolvlm_prefill_unified_0_to_profile_gmem_w \
        [get_bd_intf_pins smolvlm_prefill_unified_0/m_axi_gmem_w] \
        [get_bd_intf_pins smolvlm_axi_profile_wrapper_0/s_axi_gmem_w]

    connect_bd_net -net zynq_ultra_ps_e_0_pl_clk0 \
        [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] \
        [get_bd_pins smolvlm_axi_profile_wrapper_0/ap_clk]
    connect_bd_net -net proc_sys_reset_0_peripheral_aresetn \
        [get_bd_pins proc_sys_reset_0/peripheral_aresetn] \
        [get_bd_pins smolvlm_axi_profile_wrapper_0/ap_rst_n]

    set profile_clk_hz [design_clk_freq_hz]
    set_pin_freq_hz_if_exists smolvlm_axi_profile_wrapper_0/ap_clk $profile_clk_hz
    foreach intf_name {
        smolvlm_axi_profile_wrapper_0/s_axi_control
        smolvlm_axi_profile_wrapper_0/m_axi_control
        smolvlm_axi_profile_wrapper_0/s_axi_profile
        smolvlm_axi_profile_wrapper_0/s_axi_gmem_d
        smolvlm_axi_profile_wrapper_0/m_axi_gmem_d
        smolvlm_axi_profile_wrapper_0/s_axi_gmem_w
        smolvlm_axi_profile_wrapper_0/m_axi_gmem_w
    } {
        set_intf_freq_hz_if_exists $intf_name $profile_clk_hz
    }

    clear_addr_space smolvlm_prefill_unified_0/Data_m_axi_gmem_d
    clear_addr_space smolvlm_prefill_unified_0/Data_m_axi_gmem_w
    clear_addr_space smolvlm_axi_profile_wrapper_0/Data_m_axi_control
    clear_addr_space smolvlm_axi_profile_wrapper_0/Data_m_axi_gmem_d
    clear_addr_space smolvlm_axi_profile_wrapper_0/Data_m_axi_gmem_w
    delete_addr_seg_if_exists zynq_ultra_ps_e_0/Data/SEG_smolvlm_prefill_unified_0_Reg
    delete_addr_seg_if_exists zynq_ultra_ps_e_0/Data/SEG_smolvlm_axi_profile_wrapper_0_reg0
    delete_addr_seg_if_exists zynq_ultra_ps_e_0/Data/SEG_smolvlm_axi_profile_wrapper_0_reg0_1

    assign_bd_address -offset 0xA0000000 -range 0x00000080 \
        -target_address_space [get_bd_addr_spaces zynq_ultra_ps_e_0/Data] \
        [get_bd_addr_segs smolvlm_axi_profile_wrapper_0/s_axi_control/reg0] -force
    assign_bd_address -offset 0xA0030000 -range 0x00000100 \
        -target_address_space [get_bd_addr_spaces zynq_ultra_ps_e_0/Data] \
        [get_bd_addr_segs smolvlm_axi_profile_wrapper_0/s_axi_profile/reg0] -force
    assign_bd_address -offset 0x00000000 -range 0x00000080 \
        -target_address_space [get_bd_addr_spaces smolvlm_axi_profile_wrapper_0/Data_m_axi_control] \
        [get_bd_addr_segs smolvlm_prefill_unified_0/s_axi_control/Reg] -force

    assign_bd_address -offset 0x00000000 -range {0x10000000000000000} \
        -target_address_space [get_bd_addr_spaces smolvlm_prefill_unified_0/Data_m_axi_gmem_d] \
        [get_bd_addr_segs smolvlm_axi_profile_wrapper_0/s_axi_gmem_d/reg0] -force
    assign_bd_address -offset 0x00000000 -range {0x10000000000000000} \
        -target_address_space [get_bd_addr_spaces smolvlm_prefill_unified_0/Data_m_axi_gmem_w] \
        [get_bd_addr_segs smolvlm_axi_profile_wrapper_0/s_axi_gmem_w/reg0] -force

    assign_bd_address -offset 0x000800000000 -range 0x000800000000 \
        -target_address_space [get_bd_addr_spaces smolvlm_axi_profile_wrapper_0/Data_m_axi_gmem_d] \
        [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP0/HPC0_DDR_HIGH] -force
    assign_bd_address -offset 0x00000000 -range 0x80000000 \
        -target_address_space [get_bd_addr_spaces smolvlm_axi_profile_wrapper_0/Data_m_axi_gmem_d] \
        [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP0/HPC0_DDR_LOW] -force
    assign_bd_address -offset 0x000800000000 -range 0x000800000000 \
        -target_address_space [get_bd_addr_spaces smolvlm_axi_profile_wrapper_0/Data_m_axi_gmem_w] \
        [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP1/HPC1_DDR_HIGH] -force
    assign_bd_address -offset 0x00000000 -range 0x80000000 \
        -target_address_space [get_bd_addr_spaces smolvlm_axi_profile_wrapper_0/Data_m_axi_gmem_w] \
        [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP1/HPC1_DDR_LOW] -force

    validate_bd_design
    save_bd_design
}
}

axi_profile_bd_patch::add_smolvlm_axi_profile_wrapper
