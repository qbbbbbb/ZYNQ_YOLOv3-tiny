
import os
# 改变网络输出数据格式，以便后续处理
os.system('python group.py')

# 要对输出的yolo0和yolo1进行处理，添加第一句话{和最后一句话}去掉,
with open('yolo0.h', 'r+') as f0:
    content = f0.read()
    f0.seek(0, 0)
    f0.write('short layer_output_ref[]={'+content)
    f0.close()

f1 = open('yolo0.h','rb+')
    # 删除最后一个，
f1.seek(-3, os.SEEK_END)
f1.truncate()
f1.close()

#在文档最后添加};
f2 = open('yolo0.h','a')
f2.write("};")
f2.close()

with open('yolo1.h', 'r+') as f3:
    content = f3.read()
    f3.seek(0, 0)
    f3.write('short layer_output_ref[]={'+content)
    f3.close()

f4 = open('yolo1.h','rb+')
    # 删除最后一个，
f4.seek(-3, os.SEEK_END)
f4.truncate()
f4.close()

#在文档最后添加};
f5 = open('yolo1.h','a')
f5.write("};")
f5.close()

# 转换输出数据格式，以便绘制锚框
# interleave_output_group-yolo0
os.system("gcc -o yolo0_inter.exe yolo0_main.c yolo0.h")
os.system("yolo0_inter.exe")
# interleave_output_group-yolo1
os.system("gcc -o yolo1_inter.exe yolo1_main.c yolo1.h")
os.system("yolo1_inter.exe")

f6 = open('yolo1_o.h','rb+')
# 删除最后一个，
f6.seek(-2706, os.SEEK_END)
f6.truncate()
f6.close()

f7 = open('yolo0_o.h','rb+')
# 删除最后一个，
f7.seek(-678, os.SEEK_END)
f7.truncate()
f7.close()

# 去掉逗号，并对其进行运算
os.system('python cleardou.py')



