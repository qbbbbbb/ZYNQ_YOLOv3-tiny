import os
import sys

import string
filename  = 'D:\\lab\\ZYNQ_YOLO\\pythontool\\1.txt'
fo = open('D:\\lab\\ZYNQ_YOLO\\pythontool\\2.txt', 'w')
cnts = open(filename).readlines()
for cnt in cnts:
     cnt = cnt.strip()

     cnt = float(cnt)*256
     fo.write(str(cnt) + '\n')
fo.close()