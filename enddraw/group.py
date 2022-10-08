# coding=utf-8
import sys
import time

# coding = utf-8
def clearBlankLine():
    file1 = open('10.txt', 'r') # 要去掉空行的文件
    file2 = open('yolo0.h', 'w') # 生成没有空行的文件
    try:
        for line in file1.readlines():
            if line == '\n':
                line = line.strip("\n")
            file2.write(line)
    finally:
        file1.close()
        file2.close()

    file3 = open('13.txt', 'r') # 要去掉空行的文件
    file4 = open('yolo1.h', 'w') # 生成没有空行的文件
    try:
        for line in file3.readlines():
            if line == '\n':
                line = line.strip("\n")
            file4.write(line)
    finally:
        file3.close()
        file4.close()


if __name__ == '__main__':
    clearBlankLine()
    flag = 1
