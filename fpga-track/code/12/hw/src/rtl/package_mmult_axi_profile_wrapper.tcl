namespace eval mmult_profile_wrapper_ip {
proc get_script_folder {} {
    set script_path [file normalize [info script]]
    return [file dirname $script_path]
}

proc ensure_bus_param {busif name value} {
    set param [ipx::get_bus_parameters -quiet $name -of_objects $busif]
    if {$param eq ""} {
        set param [ipx::add_bus_parameter $name $busif]
    }
    set_property value $value $param
}

proc ensure_port_map {busif logical physical} {
    set port_map [ipx::get_port_maps -quiet $logical -of_objects $busif]
    if {$port_map eq ""} {
        set port_map [ipx::add_port_map $logical $busif]
    }
    set_property physical_name $physical $port_map
}

proc ensure_clock_busif {core} {
    set busif [ipx::get_bus_interfaces -quiet ap_clk -of_objects $core]
    if {$busif eq ""} {
        set busif [ipx::add_bus_interface ap_clk $core]
    }
    set_property bus_type_vlnv xilinx.com:signal:clock:1.0 $busif
    set_property abstraction_type_vlnv xilinx.com:signal:clock_rtl:1.0 $busif
    set_property interface_mode slave $busif
    ensure_port_map $busif CLK ap_clk
    ensure_bus_param $busif ASSOCIATED_BUSIF "s_axi_control:m_axi_control:s_axi_profile:s_axi_gmem_a:m_axi_gmem_a:s_axi_gmem_b:m_axi_gmem_b:s_axi_gmem_c:m_axi_gmem_c"
    ensure_bus_param $busif ASSOCIATED_RESET ap_rst_n
}

proc ensure_reset_busif {core} {
    set busif [ipx::get_bus_interfaces -quiet ap_rst_n -of_objects $core]
    if {$busif eq ""} {
        set busif [ipx::add_bus_interface ap_rst_n $core]
    }
    set_property bus_type_vlnv xilinx.com:signal:reset:1.0 $busif
    set_property abstraction_type_vlnv xilinx.com:signal:reset_rtl:1.0 $busif
    set_property interface_mode slave $busif
    ensure_port_map $busif RST ap_rst_n
    ensure_bus_param $busif POLARITY ACTIVE_LOW
}

proc ensure_axil_busif {core name mode addr_width data_width} {
    set busif [ipx::get_bus_interfaces -quiet $name -of_objects $core]
    if {$busif eq ""} {
        set busif [ipx::add_bus_interface $name $core]
    }
    set_property bus_type_vlnv xilinx.com:interface:aximm:1.0 $busif
    set_property abstraction_type_vlnv xilinx.com:interface:aximm_rtl:1.0 $busif
    set_property interface_mode $mode $busif

    foreach logical {AWVALID AWREADY AWADDR WVALID WREADY WDATA WSTRB ARVALID ARREADY ARADDR RVALID RREADY RDATA RRESP BVALID BREADY BRESP} {
        ensure_port_map $busif $logical "${name}_${logical}"
    }

    ensure_bus_param $busif ADDR_WIDTH $addr_width
    ensure_bus_param $busif DATA_WIDTH $data_width
    ensure_bus_param $busif PROTOCOL AXI4LITE
    ensure_bus_param $busif READ_WRITE_MODE READ_WRITE
    ensure_bus_param $busif HAS_BURST 0
    ensure_bus_param $busif HAS_LOCK 0
    ensure_bus_param $busif HAS_PROT 0
    ensure_bus_param $busif HAS_CACHE 0
    ensure_bus_param $busif HAS_QOS 0
    ensure_bus_param $busif HAS_REGION 0
    ensure_bus_param $busif HAS_WSTRB 1
    ensure_bus_param $busif HAS_BRESP 1
    ensure_bus_param $busif HAS_RRESP 1
    ensure_bus_param $busif MAX_BURST_LENGTH 1
    ensure_bus_param $busif NUM_READ_OUTSTANDING 1
    ensure_bus_param $busif NUM_WRITE_OUTSTANDING 1
    ensure_bus_param $busif SUPPORTS_NARROW_BURST 0

    return $busif
}

proc ensure_aximm_busif {core name mode addr_width data_width id_width user_width} {
    set busif [ipx::get_bus_interfaces -quiet $name -of_objects $core]
    if {$busif eq ""} {
        set busif [ipx::add_bus_interface $name $core]
    }
    set_property bus_type_vlnv xilinx.com:interface:aximm:1.0 $busif
    set_property abstraction_type_vlnv xilinx.com:interface:aximm_rtl:1.0 $busif
    set_property interface_mode $mode $busif

    foreach logical {AWVALID AWREADY AWADDR AWID AWLEN AWSIZE AWBURST AWLOCK AWCACHE AWPROT AWQOS AWREGION AWUSER WVALID WREADY WDATA WSTRB WLAST WID WUSER ARVALID ARREADY ARADDR ARID ARLEN ARSIZE ARBURST ARLOCK ARCACHE ARPROT ARQOS ARREGION ARUSER RVALID RREADY RDATA RLAST RID RUSER RRESP BVALID BREADY BRESP BID BUSER} {
        ensure_port_map $busif $logical "${name}_${logical}"
    }

    ensure_bus_param $busif ADDR_WIDTH $addr_width
    ensure_bus_param $busif DATA_WIDTH $data_width
    ensure_bus_param $busif ID_WIDTH $id_width
    ensure_bus_param $busif AWUSER_WIDTH $user_width
    ensure_bus_param $busif ARUSER_WIDTH $user_width
    ensure_bus_param $busif WUSER_WIDTH $user_width
    ensure_bus_param $busif RUSER_WIDTH $user_width
    ensure_bus_param $busif BUSER_WIDTH $user_width
    ensure_bus_param $busif PROTOCOL AXI4
    ensure_bus_param $busif READ_WRITE_MODE READ_WRITE
    ensure_bus_param $busif HAS_BURST 1
    ensure_bus_param $busif HAS_LOCK 1
    ensure_bus_param $busif HAS_PROT 1
    ensure_bus_param $busif HAS_CACHE 1
    ensure_bus_param $busif HAS_QOS 1
    ensure_bus_param $busif HAS_REGION 1
    ensure_bus_param $busif HAS_WSTRB 1
    ensure_bus_param $busif HAS_BRESP 1
    ensure_bus_param $busif HAS_RRESP 1
    ensure_bus_param $busif MAX_BURST_LENGTH 256
    ensure_bus_param $busif NUM_READ_OUTSTANDING 32
    ensure_bus_param $busif NUM_WRITE_OUTSTANDING 32
    ensure_bus_param $busif SUPPORTS_NARROW_BURST 0

    return $busif
}

proc ensure_memory_map {core busif_name mm_name block_name range width usage} {
    set mm [ipx::get_memory_maps -quiet $mm_name -of_objects $core]
    if {$mm eq ""} {
        set mm [ipx::add_memory_map $mm_name $core]
    }

    set block [ipx::get_address_blocks -quiet $block_name -of_objects $mm]
    if {$block eq ""} {
        set block [ipx::add_address_block $block_name $mm]
    }

    set_property base_address 0 $block
    set_property range $range $block
    set_property width $width $block
    set_property usage $usage $block
    if {$usage eq "register"} {
        set_property access read-write $block
    } else {
        set_property access read-write $block
    }

    set busif [ipx::get_bus_interfaces -quiet $busif_name -of_objects $core]
    set_property slave_memory_map_ref $mm_name $busif
}

proc ensure_address_space {core busif_name as_name range width} {
    set addr_space [ipx::get_address_spaces -quiet $as_name -of_objects $core]
    if {$addr_space eq ""} {
        set addr_space [ipx::add_address_space $as_name $core]
    }

    set_property range $range $addr_space
    set_property width $width $addr_space

    set busif [ipx::get_bus_interfaces -quiet $busif_name -of_objects $core]
    set_property master_address_space_ref $as_name $busif
}

proc component_is_stale {component_xml source_files} {
    if {![file exists $component_xml]} {
        return 1
    }

    set component_mtime [file mtime $component_xml]
    foreach source_file $source_files {
        if {![file exists $source_file]} {
            error "Missing wrapper source file: $source_file"
        }
        if {[file mtime $source_file] > $component_mtime} {
            return 1
        }
    }
    return 0
}

proc package_wrapper_repo {} {
    variable script_folder

    set source_files [list \
        [file normalize [file join $script_folder mmult_axi_profile_wrapper.v]] \
        [file normalize [file join $script_folder mmult_axi_profile_regs.v]] \
    ]
    set dependency_files [concat [list [file normalize [info script]]] $source_files]
    set repo_root [file normalize [file join $script_folder ../rtl_ip_repo]]
    set ip_root [file normalize [file join $repo_root mmult_axi_profile_wrapper]]
    set component_xml [file join $ip_root component.xml]

    if {![component_is_stale $component_xml $dependency_files]} {
        return $ip_root
    }

    file mkdir $repo_root

    set saved_xpr ""
    if {[current_project -quiet] ne ""} {
        set project_name [get_property NAME [current_project]]
        set project_dir [get_property DIRECTORY [current_project]]
        set saved_xpr [file normalize [file join $project_dir "${project_name}.xpr"]]
        close_project -quiet
    }

    set temp_project_dir [file normalize [file join $repo_root .pkg_mmult_axi_profile_wrapper]]
    file delete -force $temp_project_dir
    file delete -force $ip_root

    create_project -force mmult_axi_profile_wrapper_pkg $temp_project_dir -part xcvu37p_CIV-fsvh2892-2-e
    add_files -norecurse $source_files
    set_property top mmult_axi_profile_wrapper [get_filesets sources_1]
    update_compile_order -fileset sources_1

    ipx::package_project -root_dir $ip_root -vendor skyward.com -library user -taxonomy {/UserIP} -import_files
    set core [ipx::current_core]
    set_property name mmult_axi_profile_wrapper $core
    set_property version 1.0 $core
    set_property display_name {MMult AXI Profile Wrapper} $core
    set_property description {AXI pass-through wrapper with protocol-level performance counters for the staged timed mixed GEMM kernel.} $core

    ensure_clock_busif $core
    ensure_reset_busif $core
    ensure_axil_busif $core s_axi_control slave 8 32
    ensure_axil_busif $core m_axi_control master 8 32
    ensure_axil_busif $core s_axi_profile slave 8 32
    ensure_aximm_busif $core s_axi_gmem_a slave 64 512 1 1
    ensure_aximm_busif $core m_axi_gmem_a master 64 512 1 1
    ensure_aximm_busif $core s_axi_gmem_b slave 64 512 1 1
    ensure_aximm_busif $core m_axi_gmem_b master 64 512 1 1
    ensure_aximm_busif $core s_axi_gmem_c slave 64 512 1 1
    ensure_aximm_busif $core m_axi_gmem_c master 64 512 1 1

    ensure_memory_map $core s_axi_control s_axi_control reg0 256 32 register
    ensure_memory_map $core s_axi_profile s_axi_profile reg0 256 32 register
    ensure_memory_map $core s_axi_gmem_a s_axi_gmem_a reg0 {0x10000000000000000} 512 memory
    ensure_memory_map $core s_axi_gmem_b s_axi_gmem_b reg0 {0x10000000000000000} 512 memory
    ensure_memory_map $core s_axi_gmem_c s_axi_gmem_c reg0 {0x10000000000000000} 512 memory

    ensure_address_space $core m_axi_control Data_m_axi_control 256 8
    ensure_address_space $core m_axi_gmem_a Data_m_axi_gmem_a {0x10000000000000000} 64
    ensure_address_space $core m_axi_gmem_b Data_m_axi_gmem_b {0x10000000000000000} 64
    ensure_address_space $core m_axi_gmem_c Data_m_axi_gmem_c {0x10000000000000000} 64

    ipx::create_xgui_files $core
    ipx::update_checksums $core
    ipx::check_integrity -quiet $core
    ipx::save_core $core
    close_project -quiet

    if {$saved_xpr ne ""} {
        open_project $saved_xpr
    }

    return $ip_root
}
}

namespace eval mmult_profile_wrapper_ip {
    variable script_folder [get_script_folder]
}
set MMULT_AXI_PROFILE_WRAPPER_REPO [mmult_profile_wrapper_ip::package_wrapper_repo]
