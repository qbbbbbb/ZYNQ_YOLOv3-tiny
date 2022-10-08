# coding=utf-8
import os,sys
# loadimage()
os.system("gcc -o testrunall0.exe loadimg.c stb_image.h stb_image_write.h")
os.system("testrunall0.exe")

# head_short
os.system("gcc -o testrunall1.exe head_short.c inputimg.h")
os.system("testrunall1.exe")

# sort_input
os.system("gcc -o testrunall2.exe sort_input.c input_image_short.h")
os.system("testrunall2.exe")

# padchannel
os.system("gcc -o testrunall3.exe padchannel.c input_img_sort.h")
os.system("testrunall3.exe")