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
src_img_path = sys.argv[1]
src_xml_path = sys.argv[2]
dst_img_path = sys.argv[3]
dst_xml_path = sys.argv[4]

# 随机产生切割尺度坐标
def random_corp(image_src_data):
    h, w = image_src_data.shape[:2]
    width_random = random.uniform(0.50, 0.7)
    height_random = random.uniform(0.50, 0.7)
    new_width = int(w*width_random)
    new_height = int(h*height_random)
    x_start_section = [0,(w-new_width)]
    y_start_section = [0,(h-new_height)]
    x_start = random.randint(0,(w-new_width))
    y_start = random.randint(0,(h-new_height))
    
    return new_width, new_height, x_start, y_start, width_random ,height_random


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

def mx_my(a,b):
    min_x = min(a)
    min_y = min(b)
    max_x = max(a)
    max_y = max(b)

    return int(min_x[0]), int(min_y[0]), int(max_x[0]) ,int(max_y[0])

def xml_process(xml_file ,dst_xml_path,x_s, y_s, nw_, nh_):
      
    src_xml = xml_file
    x_start_coor = x_s
    y_start_coor = y_s
    x_end_coor = x_s + nw_
    y_end_coor = y_s + nh_
    num = []                                     #　　ｎｕｍ是ｌｉｓｔ，用来记录需要做删除操作的标签的ｉｎｄｅｘ
    flag = False                                 #  ｆｌａｇ是判断时候有需要做删除超界框的操作的标志变量
    print("要更改的xml文件为：",src_xml)
    # 实例化一个dom对象
    dom = xml.dom.minidom.parse(src_xml)
    root = dom.documentElement

    # 修改对应的图片的文件名
    itemlist = root.getElementsByTagName('filename')
    item = itemlist[0]
    file_new_name = (src_xml.split('/')[-1])[:-4] +  '_corp_' + str(x_s)  + "_" + str(y_s) + "_" + str(nw_)  + "_" + str(nh_) + ".jpg"
    item.firstChild.data = file_new_name
    print("修改后的文件对应的jpg的文件名为：", item.firstChild.data)

    # 修改对应图片的尺寸大小  需要手动改
    itemlist_width = root.getElementsByTagName('width')
    item_width = itemlist_width[0]
    item_width.firstChild.data = int(nw_)
    itemlist_height = root.getElementsByTagName('height')
    item_height = itemlist_height[0]
    item_height.firstChild.data = int(nh_)

    # 修改坐标
    print(len(root.getElementsByTagName("bndbox")))
    print(root.getElementsByTagName("bndbox"))
    for i in range(len(root.getElementsByTagName("bndbox"))):
        print(i)
        itemlist_xmin = root.getElementsByTagName('xmin')
        print(itemlist_xmin)
        item_xmin = itemlist_xmin[i]
        x_min = int(item_xmin.firstChild.data)
#        item_xmin.firstChild.data = str(int(item_xmin.firstChild.data)*scale)
        itemlist_ymin = root.getElementsByTagName('ymin')
        item_ymin = itemlist_ymin[i]
        y_min = int(item_ymin.firstChild.data)
#        item_ymin.firstChild.data = str(int(item_ymin.firstChild.data)*scale)
        itemlist_xmax = root.getElementsByTagName('xmax')
        item_xmax = itemlist_xmax[i]
        x_max = int(item_xmax.firstChild.data)
#        item_xmax.firstChild.data = str(int(item_xmax.firstChild.data)*scale)
        itemlist_ymax = root.getElementsByTagName('ymax')
        item_ymax = itemlist_ymax[i]
        y_max = int(item_ymax.firstChild.data)
#        item_ymax.firstChild.data = str(int(item_ymax.firstChild.data)*scale)

        if x_min <= x_start_coor or y_min <= y_start_coor or x_max >= x_end_coor or y_max >= y_end_coor:
            num.append(i)
            flag = True                                                                                  #　因为DOM的操作是实时的，所以当直接执行删除动作时会影响之后的操作，
                                                                                                         #　除非使用嵌套ｆｏｒ循环操作，此处采取最后统一删除操作 
        else:
            x_min = x_min - x_start_coor
            y_min = y_min - y_start_coor
            x_max = x_max - x_start_coor
            y_max = y_max - y_start_coor
            item_xmin.firstChild.data = str(x_min)
            item_ymin.firstChild.data = str(y_min)
            item_xmax.firstChild.data = str(x_max)
            item_ymax.firstChild.data = str(y_max)
        
    if flag:
        len_num = len(num)
        count = 0                                 # count作为进行删除操作的计数变量，对之后的ｉｎｄｅｘ进行修正 
        for j in range(len_num):
            index = j - count                                     # 因为进行了删除操作，ｉｎｄｅｘ值需要减去操作次数
            remove_node = root.getElementsByTagName("object")[num[index]]
            root.removeChild(remove_node)
            count += 1


    dst_xml_file = dst_xml_path + '/' + xml_file.split("/")[-1][:-4] +  '_corp_' + str(x_s)  + "_" + str(y_s) + "_" + str(nw_)  + "_" + str(nh_)  + ".xml"

    with open(dst_xml_file, 'w') as fh:
        dom.writexml(fh)
        
for image_file in os.listdir(src_img_path):
    img_name = src_img_path + '/' + image_file
    print("当前裁剪的图像为：{}".format(img_name))
    src_img = cv.imread(img_name, cv.IMREAD_COLOR)
    nw, nh, xs, ys, wr, hr = random_corp(src_img)
    dst_corp_img_name = dst_img_path + '/' + image_file[:-4] + '_corp_' + str(xs)  + "_" + str(ys) + "_" + str(nw)  + "_" + str(nh) + ".jpg"
    dst_corp_img = src_img[ys:(ys+nh), xs:(xs+nw)]
    cv.imwrite(dst_corp_img_name, dst_corp_img, [int(cv.IMWRITE_JPEG_QUALITY), 100])

    xml_name = src_xml_path + '/' + image_file[:-4] + ".xml"
    print("当前改写的xml文件为：{}".format(xml_name))
    print(xml_name, dst_xml_path)
    xml_process(xml_name, dst_xml_path, xs, ys, nw, nh)