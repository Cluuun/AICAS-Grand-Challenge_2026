namespace eval project_flow {
    variable project_name "proj_opae_fim"
    variable project_dir "./proj_opae_fim"
    variable target_part "xcvu37p_CIV-fsvh2892-2-e"
    variable default_bd "../../src/ipi/cosim_mmult_accel_staged_timed_mixed_xdma_v2.bd.tcl"
    variable default_ip_repo "./add_cosim_ip_path.tcl"
    variable constraints_files [list "./xdma.xdc"]
    variable initialized 0
}

proc project_flow::arg_or_default {idx default_value} {
    global argv
    if {[llength $argv] > $idx} {
        set value [lindex $argv $idx]
        if {$value ne ""} {
            return $value
        }
    }
    return $default_value
}

proc project_flow::script_bd_path {} {
    variable default_bd
    return [arg_or_default 0 $default_bd]
}

proc project_flow::ip_repo_script_path {} {
    variable default_ip_repo
    return [arg_or_default 1 $default_ip_repo]
}

proc project_flow::create_base_project {} {
    variable project_name
    variable project_dir
    variable target_part
    variable initialized

    if {$initialized} {
        return
    }

    close_project -quiet
    puts "INFO: Creating project ${project_name} at ${project_dir}"
    create_project -force $project_name $project_dir -part $target_part

    set ip_repo_script [ip_repo_script_path]
    if {![file exists $ip_repo_script]} {
        error "ERROR: IP repo script not found: $ip_repo_script"
    }

    source $ip_repo_script
    set current_ip_repos [get_property ip_repo_paths [current_project]]
    set resolved_ip_repos [concat $AFU_IP_PATH $current_ip_repos]
    set unique_ip_repos [list]
    foreach repo_path $resolved_ip_repos {
        if {$repo_path eq ""} {
            continue
        }
        set normalized_repo [file normalize $repo_path]
        if {[lsearch -exact $unique_ip_repos $normalized_repo] < 0} {
            lappend unique_ip_repos $normalized_repo
        }
    }
    set_property ip_repo_paths $unique_ip_repos [current_project]
    update_ip_catalog

    set initialized 1
}

proc project_flow::add_constraints {} {
    variable constraints_files

    set has_constraints 0
    foreach xdc $constraints_files {
        if {[file exists $xdc]} {
            add_files -fileset constrs_1 -norecurse $xdc
            set has_constraints 1
        } else {
            puts "WARNING: constraints file not found: $xdc"
        }
    }
    if {!$has_constraints} {
        puts "WARNING: No constraints file found."
    }
}

proc project_flow::get_current_bd_file {} {
    set bd_name [get_property NAME [current_bd_design]]
    if {$bd_name eq ""} {
        error "ERROR: current_bd_design is empty after sourcing BD Tcl."
    }

    set bd_file [get_files -quiet "${bd_name}.bd"]
    if {$bd_file eq ""} {
        error "ERROR: Failed to locate generated BD file for ${bd_name}."
    }
    return $bd_file
}

proc project_flow::generate_bd_outputs {} {
    set bd_file [get_current_bd_file]
    set bd_name [get_property NAME [current_bd_design]]

    puts "INFO: BD file: $bd_file"

    validate_bd_design
    save_bd_design
    generate_target all $bd_file
    export_ip_user_files -of_objects $bd_file -no_script -sync -force -quiet

    set wrapper_files [make_wrapper -files $bd_file -top -quiet]
    if {$wrapper_files eq ""} {
        set wrapper_files [get_files -quiet -all -regexp ".*/${bd_name}_wrapper\\.(v|vhd)$"]
    }
    if {$wrapper_files eq ""} {
        error "ERROR: Could not find generated wrapper file."
    }

    set wrapper_top "${bd_name}_wrapper"
    puts "INFO: Wrapper top: $wrapper_top"
    foreach wrapper_file $wrapper_files {
        if {[llength [get_files -quiet $wrapper_file]] == 0} {
            add_files -norecurse $wrapper_file
        }
    }
    update_compile_order -fileset sources_1
    report_compile_order -fileset sources_1
    return $bd_file
}

proc project_flow::prepare_project_from_bd {} {
    create_base_project

    set script_bd [script_bd_path]
    puts "INFO: Sourcing BD Tcl: $script_bd"
    if {![file exists $script_bd]} {
        error "ERROR: BD Tcl not found: $script_bd"
    }

    source $script_bd
    add_constraints
    set bd_file [generate_bd_outputs]
    puts "INFO: Project prepared from BD: $bd_file"
    return $bd_file
}

project_flow::create_base_project
