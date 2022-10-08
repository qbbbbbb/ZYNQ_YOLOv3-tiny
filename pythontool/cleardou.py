import os
import sys

import string
filename0 = 'yolo0_o.h'
fo0 = open('yolo0_cal.dat', 'w')
cnts = open(filename0).readlines()
for cnt in cnts:
     cnt = cnt.strip()
     if cnt.endswith(','):
        cnt = cnt[:-1]
        cnt = float(cnt)/256
     fo0.write(str(cnt) + '\n')
fo0.close()

filename1 = 'yolo1_o.h'
fo1 = open('yolo1_cal.dat', 'w')
cnts = open(filename1).readlines()
for cnt in cnts:
     cnt = cnt.strip()
     if cnt.endswith(','):
        cnt = cnt[:-1]
        cnt = float(cnt)/256
     fo1.write(str(cnt) + '\n')
fo1.close()