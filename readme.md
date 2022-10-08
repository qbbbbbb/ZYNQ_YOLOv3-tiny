## Base on ZYNQ7020 YOLOv3-Tiny 
### Cited from
https://github.com/Yu-Zhewen/Tiny_YOLO_v3_ZYNQ
### How To Use
- Group0_input data for network computation is generated using ZYNQ_YOLO\runall\pygen0\runall.py. (Note that the py script deletes all.h files in the current path before use, to prevent overwriting errors, you can add delete scripts by yourself)
  - runall.py does the following:
    - load image data£¨loadimg.c ...£©
    - quantise the image£¨head_short.c ...£©
    - sort_input:Change the order of input image data from 416 * 416 * 3 to 3 * 416 * 416. (sort_input.c)
    - pad the input from 3 * 416 * 416 to 4 * 416 * 416

- Replace group0_input with the output data from the previous step in the SDK running folder, and send the input data to the network for calculation to get the output.
  - We need data for Group10 and Group13, which are two layers of YOLO data respectively. You can copy it out by going into sdk_terminal with the printf function. This is a stupid method.
- Vivado HLS Command Prompt can be used to input script instructions for subsequent operation (because some data types are involved and only work in HLS environment).
  - The copied output data is fed into endraw.py for processing. endraw.py contains the following operations:
    - The copied data may have blank lines, so use pythonTool /group.py to remove blank lines. The output results can be verified by the number of output lines.
    - Do some processing (including deleting , or adding } ) to the data document for subsequent program processing.
    - Converts the output data order to draw the anchor box. (including yolo0.h and yolo1.h)
    - Delete data that is not needed after the transformation order (usually 0)
    - Drop the ", "and send it to your draw_box for calculation.
  - Using ZYNQ_YOLO \ draw_box_c
    - Replace yolo0_cal.dat/yolo1_cal.dat and input_image_short.h in the main function