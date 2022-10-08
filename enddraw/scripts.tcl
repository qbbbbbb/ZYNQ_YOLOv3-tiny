open_project draw_box
set_top main
add_files stb_image_write.h
add_files stb_image.h
add_files main.cpp
add_files input_image_short.h
add_files image.h
add_files image.cpp
add_files darknet.h
add_files -tb main.cpp -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
open_solution "solution1"
set_part {xc7k70t-fbv676-1}
create_clock -period 10 -name default
#source "./draw_box/solution1/directives.tcl"
csim_design
