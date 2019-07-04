#! /usr/bin/env python
# -*- coding:UTF-8 -*-

from __future__ import print_function
import os
import sys
import cv2 as cv 
import numpy as np
import xml.dom.minidom
import math

def makedir_path(the_path):
    if not os.path.exists(the_path):
        os.makedirs(the_path, mode=0o777)
    print("{} 该目录不存在，已创建成功".format(the_path))

    return 0
# 重载编码    
reload(sys)
sys.setdefaultencoding('utf8')
# 接收参数
src_img_path = sys.argv[1]
src_xml_path = sys.argv[2]
dst_img_path = sys.argv[3]
dst_xml_path = sys.argv[4]
scale = float(sys.argv[5])                      #  resize的尺度
# 绝对路径
cur_path = os.getcwd()
src_img_path = os.path.join(cur_path, src_img_path)
src_xml_path = os.path.join(cur_path, src_xml_path)
dst_img_path = os.path.join(cur_path, dst_img_path)
dst_xml_path = os.path.join(cur_path, dst_xml_path)
# 创建不存在的路径，只创建要保存的dst的相关路径，src不需要
makedir_path(dst_img_path)
makedir_path(dst_xml_path)
crop_image = lambda img, x0, y0, w, h: img[y0:y0+h, x0:x0+w]

def resize_image(img, scale):
    h, w = img.shape[0:2]
    img_resize = cv.resize(img,(int(w*scale), int(h*scale)), interpolation=cv.INTER_NEAREST)
    return img_resize

def mx_my(a,b):
    min_x = min(a)
    min_y = min(b)
    max_x = max(a)
    max_y = max(b)

    return int(min_x[0]), int(min_y[0]), int(max_x[0]) ,int(max_y[0])

def xml_process(xml_file ,dst_xml_path, scale):
      
    src_xml = xml_file
    print("要更改的xml文件为：",src_xml)
    # 实例化一个dom对象
    dom = xml.dom.minidom.parse(src_xml)
    root = dom.documentElement

    # 修改对应的图片的文件名
    itemlist = root.getElementsByTagName('filename')
    item = itemlist[0]
    file_new_name = str(item.firstChild.data)[:-4] + '_resize_' + str(scale) + ".jpg"
    item.firstChild.data = file_new_name
    print("修改后的文件对应的jpg的文件名为：", item.firstChild.data)

    # 修改对应图片的尺寸大小  需要手动改
    itemlist_width = root.getElementsByTagName('width')
    item_width = itemlist_width[0]
    w = int(item_width.firstChild.data)
    item_width.firstChild.data = int(w*scale)
    itemlist_height = root.getElementsByTagName('height')
    item_height = itemlist_height[0]
    h = int(item_height.firstChild.data)
    item_height.firstChild.data = int(h*scale)

    # 修改坐标
    for i in range(len(root.getElementsByTagName("bndbox"))):
        itemlist_xmin = root.getElementsByTagName('xmin')
        item_xmin = itemlist_xmin[i]
        item_xmin.firstChild.data = str(int(int(item_xmin.firstChild.data)*scale))
        itemlist_ymin = root.getElementsByTagName('ymin')
        item_ymin = itemlist_ymin[i]
        item_ymin.firstChild.data = str(int(int(item_ymin.firstChild.data)*scale))
        itemlist_xmax = root.getElementsByTagName('xmax')
        item_xmax = itemlist_xmax[i]
        item_xmax.firstChild.data = str(int(int(item_xmax.firstChild.data)*scale))
        itemlist_ymax = root.getElementsByTagName('ymax')
        item_ymax = itemlist_ymax[i]
        item_ymax.firstChild.data = str(int(int(item_ymax.firstChild.data)*scale))
        
    dst_xml_file = dst_xml_path + '/' + xml_file.split("/")[-1][:-4] + "_resize_"  + str(scale) + ".xml"

    with open(dst_xml_file, 'w') as fh:
        dom.writexml(fh)
        
for image_file in os.listdir(src_img_path):
    img_name = src_img_path + '/' + image_file
    print("当前resize的图像为：{}".format(img_name))
    src_img = cv.imread(img_name, cv.IMREAD_COLOR)
    dst_resize_img = resize_image(src_img, scale)
    dst_resize_img_name = dst_img_path + '/' + image_file[:-4] + '_resize_' + str(scale) + ".jpg"
    cv.imwrite(dst_resize_img_name, dst_resize_img, [int(cv.IMWRITE_JPEG_QUALITY), 100])

for xml_file in os.listdir(src_xml_path):
    xml_name = src_xml_path + '/' + xml_file
    print("当前改写的xml文件为：{}".format(xml_name))
    print(xml_name, dst_xml_path,scale)
    xml_process(xml_name, dst_xml_path, scale)
