set script_dir [file dirname [file normalize [info script]]]
set repo_root  [file normalize [file join $script_dir ../..]]

set hls_ip [file normalize [file join $repo_root src/hls/vlm_w8a8/vlm_engine_v6/vlm_engine_v6_kv260/solution1/impl/ip]]
set pe_ip  [file normalize [file join $repo_root src/rtl_ip_repo/pe_array_v6]]

foreach {name path} [list hls $hls_ip pe $pe_ip] {
    set component_xml [file join $path component.xml]
    if {![file exists $component_xml]} {
        error "ERROR: Missing packaged v6 $name IP component.xml: $component_xml"
    }
}

set KV260_IP_REPOS [list $hls_ip $pe_ip]

puts "INFO: Using KV260 v6 IP repositories:"
foreach repo $KV260_IP_REPOS {
    puts "  $repo"
}
