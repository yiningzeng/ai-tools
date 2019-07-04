#! /usr/bin/env python
# -*- coding:UTF-8 -*-

from __future__ import print_function
import os
import sys
import shutil

#  通过终端参数获取img和xml的文件夹名
img_path = sys.argv[1]
xml_path = sys.argv[2]
res_path = sys.argv[3]           
# 获取当前目录名
cur_path = os.getcwd()
print("当前的路径为：", cur_path)
# 拼接img和xml的完整路径
img_path_abs = os.path.join(cur_path,img_path)
xml_path_abs = os.path.join(cur_path,xml_path)
img_res_path_abs = os.path.join(cur_path, res_path, img_path)
xml_res_path_abs = os.path.join(cur_path, res_path, xml_path)
# ------------------创建相对应的目录---------------------
# 创建文件夹函数
def makedir_path(the_path):
    if not os.path.exists(the_path):
        os.makedirs(the_path, mode=0o777)
    print("{} 该目录不存在，已创建成功".format(the_path))

    return 0

# 创建文件夹
makedir_path(img_res_path_abs)
makedir_path(xml_res_path_abs)

# ------------------------------------------------------
print("img所在目录为{0};xml所在目录为{1}".format(img_path_abs, xml_path_abs))
print("img待整理目录为{0};xml待整理目录为{1}".format(img_res_path_abs, xml_res_path_abs))
# 判断文件路径是否存在
if not os.path.exists(img_path_abs):
    print("img的路径不存在，请检查！")
if not os.path.exists(xml_path_abs):
    print("xml的路径不存在，请检查！")

if os.path.exists(img_path_abs) and os.path.exists(xml_path_abs):
# 或许两个目录下的文件列表
    img_file_list = os.listdir(img_path_abs)
    xml_file_list = os.listdir(xml_path_abs)
    img_list = []
    xml_list = []
# 因为img和xml文件的后缀不同，所以需要遍历，然后取后缀前的名
    for img_file in img_file_list:
        img_list.append(img_file[:-4])
    for xml_file in xml_file_list:
        xml_list.append(xml_file[:-4])
    print("img路径中共有{0}个文件，xml路径中共有{1}个文件".format(len(img_list),len(xml_list)))
    inter_set_list = list(set(img_list).intersection(set(xml_list)))
    print("两路径交集的文件数为：", len(inter_set_list))
    img_inter_list = []
    xml_inter_list = []
    for img_file_ in inter_set_list:
        img_inter_list.append((img_file_ + ".jpg"))
    for xml_file_ in inter_set_list:
        xml_inter_list.append((xml_file_ + ".xml"))
    print("交集中img路径中共有{0}个文件，xml路径中共有{1}个文件".format(len(img_inter_list),len(xml_inter_list)))
    # 复制交集部分的img到目标目录
    num = 35300
    for img_file_inter in img_inter_list:
        print((img_path_abs + '/' + img_file_inter),(img_res_path_abs + '/' + str(num).zfill(7)))
        shutil.copy((img_path_abs + '/' + img_file_inter),  (img_res_path_abs + '/' + str(num).zfill(7) + ".jpg")) 
        # 复制交集部分的xml到目标目录
        shutil.copy((xml_path_abs + '/' + img_file_inter[:-4] + '.xml'),  (xml_res_path_abs + '/' + str(num).zfill(7) + ".xml")) 
        num += 1
    print("交集中img路径中共有{0}个文件，xml路径中共有{1}个文件".format(len(img_inter_list),len(xml_inter_list)))
    
