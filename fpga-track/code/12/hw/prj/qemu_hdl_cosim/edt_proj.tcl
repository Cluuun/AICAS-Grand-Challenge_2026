source create_proj.tcl
set bd_file [project_flow::prepare_project_from_bd]
puts "SUCCESS: EDT project is ready at ./proj_opae_fim"
puts "BD: $bd_file"
close_project
exit
