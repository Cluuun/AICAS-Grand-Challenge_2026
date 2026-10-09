set script_dir [file dirname [file normalize [info script]]]
set repo_root [file normalize [file join $script_dir ../..]]
set hls_root [file normalize [file join $repo_root src/hls/vlm_w8a8]]

set KV260_IP_REPOS [list]

set final_ip_xmls [list \
    [file join $hls_root smolvlm_prefill_unified linear_accelerator_kv260 solution1 impl ip component.xml] \
]

foreach component_xml $final_ip_xmls {
    set normalized_xml [file normalize $component_xml]
    if {![file exists $normalized_xml]} {
        error "ERROR: Missing packaged IP component.xml: $normalized_xml"
    }
    set repo_dir [file dirname $normalized_xml]
    if {[lsearch -exact $KV260_IP_REPOS $repo_dir] < 0} {
        lappend KV260_IP_REPOS $repo_dir
    }
}

if {[llength $KV260_IP_REPOS] != 1} {
    error "ERROR: Expected exactly 1 linear accelerator IP repository, got [llength $KV260_IP_REPOS]"
}

set AFU_IP_PATH $KV260_IP_REPOS

puts "INFO: Using KV260 linear accelerator IP repository:"
foreach repo $KV260_IP_REPOS {
    puts "  $repo"
}
