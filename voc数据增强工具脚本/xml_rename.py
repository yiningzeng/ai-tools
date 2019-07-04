#! /usr/bin/env python
# -*- coding:UTF-8 -*-

from __future__ import print_function
import os
import sys
import cv2 as cv 
import numpy as np
import xml.dom.minidom
import math
import random

def makedir_path(the_path):
    if not os.path.exists(the_path):
        os.makedirs(the_path, mode=0o777)
    print("{} 该目录不存在，已创建成功".format(the_path))

    return 0
# 重载编码    
reload(sys)
sys.setdefaultencoding('utf8')
# 接收参数

src_xml_path = sys.argv[1]

# 绝对路径
cur_path = os.getcwd()
src_xml_path = os.path.join(cur_path, src_xml_path)

# 创建不存在的路径，只创建要保存的dst的相关路径，src不需要




def xml_process(xml_file):
       
    src_xml = xml_file
    print("要更改的xml文件为：",src_xml)
    # 实例化一个dom对象
    dom = xml.dom.minidom.parse(xml_file)
    root = dom.documentElement

    # 修改对应的图片的文件名
    itemlist = root.getElementsByTagName('filename')
    item = itemlist[0]
    file_new_name = xml_file.split("/")[-1]
    file_new_name = file_new_name[:-4] + ".jpg"
    item.firstChild.data = file_new_name
    print("修改后的文件对应的jpg的文件名为：", item.firstChild.data)

        
    dst_xml_file = xml_file

    with open(dst_xml_file, 'w') as fh:
        dom.writexml(fh)
        
for xml_file in os.listdir(src_xml_path ):
    if len(xml_file) == 11 :
        xml_name = src_xml_path  + '/' + xml_file
        print("当前修改的图像为：{}".format(xml_name ))
        xml_process(xml_name)