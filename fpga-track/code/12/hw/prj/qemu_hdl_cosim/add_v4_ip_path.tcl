set script_dir [file dirname [file normalize [info script]]]
set repo_root [file normalize [file join $script_dir ../..]]
set hls_ip [file normalize [file join $repo_root src/hls/vlm_w8a8/vlm_engine_v4/vlm_engine_v4_kv260/solution1/impl/ip]]
set pe_ip  [file normalize [file join $repo_root src/rtl_ip_repo/pe_array_v4]]

set KV260_IP_REPOS [list]
foreach repo_dir [list $hls_ip $pe_ip] {
    set component_xml [file join $repo_dir component.xml]
    if {![file exists $component_xml]} {
        error "ERROR: Missing packaged v4 IP component.xml: $component_xml"
    }
    if {[lsearch -exact $KV260_IP_REPOS $repo_dir] < 0} {
        lappend KV260_IP_REPOS $repo_dir
    }
}

set AFU_IP_PATH $KV260_IP_REPOS

puts "INFO: Using KV260 v4 IP repositories:"
foreach repo $KV260_IP_REPOS {
    puts "  $repo"
}
