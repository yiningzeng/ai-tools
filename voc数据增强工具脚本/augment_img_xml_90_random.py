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
angle_tran = int(sys.argv[5])
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

# 对图片进行旋转的函数

def rotate_image(img, angle, crop = False):                                   # 旋转，angle是角度值，不是弧度
    h, w = img.shape[:2]	
    angle %= 360	
    M_rotate = cv.getRotationMatrix2D((w/2, h/2), angle, 1)	
    img_rotated = cv.warpAffine(img, M_rotate, (w, h))
    if crop:
        angle_crop = angle % 180
        if angle_crop > 90:
            angle_crop = 180 - angle_crop			

        theta = angle_crop * np.pi / 180.0		
        hw_ratio = float(h) / float(w)
        tan_theta = np.tan(theta)
        numerator = np.cos(theta) + np.sin(theta) * tan_theta	
        r = hw_ratio if h > w else 1 / hw_ratio		
        denominator = r * tan_theta + 1		
        crop_mult = numerator / denominator		

        w_crop = int(round(crop_mult*w))
        h_crop = int(round(crop_mult*h))
        x0 = int((w-w_crop)/2)
        y0 = int((h-h_crop)/2)

        img_rotated = crop_image(img_rotated, x0, y0, w_crop, h_crop)

    return img_rotated

def pad_image(img, pad_w, pad_h):
    
    img_pad = np.zeros((pad_h, pad_w, 3))    
    h, w = img.shape[:2]    
    img_pad[int((pad_h - h)/2):int((pad_h - h)/2)+h ,int((pad_w - w)/2):int((pad_w - w)/2)+w ] = img
        
    return img_pad

def rot_compute(x0, y0, x1, y1, angle_):
    dx = x1 - x0
    dy = y0 - y1
#    print("笛卡尔坐标下的距离为：{0}{1}".format(dx, dy))
    magnitude, angle =	cv.cartToPolar(dx, dy)
#    print("距离{}角度{}".format(magnitude, angle))
    angle = angle + angle_
    dx_, dy_ = cv.polarToCart(magnitude, angle, True)
#    print(dx_, dy_)
    x2 = dx_ + x0
    y2 = y0 - dy_ 
#    print(x2[0], y2[0])    
    return x2[0], y2[0]

def mx_my(a,b):
    min_x = min(a)
    min_y = min(b)
    max_x = max(a)
    max_y = max(b)

    return int(min_x[0]), int(min_y[0]), int(max_x[0]) ,int(max_y[0])

def xml_process(xml_file ,dst_xml_path, new_width, new_higth, angle):
     
    angle_ = (angle/180.0) * math.pi    
    src_xml = xml_file
    print("要更改的xml文件为：",src_xml)
    # 实例化一个dom对象
    dom = xml.dom.minidom.parse(xml_file)
    root = dom.documentElement

    # 修改对应的图片的文件名
    itemlist = root.getElementsByTagName('filename')
    item = itemlist[0]
    file_new_name = str(item.firstChild.data)[:-4] + '_pad_' + str(angle) + ".jpg"
    item.firstChild.data = file_new_name
    print("修改后的文件对应的jpg的文件名为：", item.firstChild.data)

    # 修改对应图片的尺寸大小  需要手动改
    itemlist_width = root.getElementsByTagName('width')
    item_width = itemlist_width[0]
    w = item_width.firstChild.data
    item_width.firstChild.data = new_width
    itemlist_height = root.getElementsByTagName('height')
    item_height = itemlist_height[0]
    h = item_height.firstChild.data
    item_height.firstChild.data = new_higth

    dw = int((new_width - int(w))/2)
    dh = int((new_higth - int(h))/2)
    print("gfdhgfgjghkjhlk",dw,dh)
    # 修改坐标
    for i in range(len(root.getElementsByTagName("bndbox"))):
        itemlist_xmin = root.getElementsByTagName('xmin')
        item_xmin = itemlist_xmin[i]

        item_xmin.firstChild.data = str(int(item_xmin.firstChild.data) + dw)

        itemlist_ymin = root.getElementsByTagName('ymin')
        item_ymin = itemlist_ymin[i]
        item_ymin.firstChild.data = str(int(item_ymin.firstChild.data) + dh)

        x1, y1 = int(item_xmin.firstChild.data), int(item_ymin.firstChild.data)

        itemlist_xmax = root.getElementsByTagName('xmax')
        item_xmax = itemlist_xmax[i]
        item_xmax.firstChild.data = str(int(item_xmax.firstChild.data) + dw)
        itemlist_ymax = root.getElementsByTagName('ymax')
        item_ymax = itemlist_ymax[i]
        item_ymax.firstChild.data = str(int(item_ymax.firstChild.data) + dh)
        x4, y4 =  int(item_xmax.firstChild.data), int(item_ymax.firstChild.data)
        x2, y2 = x4, y1
        x3, y3 = x1, y4
        print("所以两个端点的坐标分别为：{0}{1}".format((x1, y1),(x4, y4)))
        print("计算模块：",int(new_width/2), int(new_higth/2), x1, y1, angle_)
        x1_, y1_ = rot_compute(int(new_width/2), int(new_higth/2), x1, y1, angle_)
        x2_, y2_ = rot_compute(int(new_width/2), int(new_higth/2), x2, y2, angle_)
        x3_, y3_ = rot_compute(int(new_width/2), int(new_higth/2), x3, y3, angle_)
        x4_, y4_ = rot_compute(int(new_width/2), int(new_higth/2), x4, y4, angle_)
        x_list = list([x1_, x2_, x3_, x4_])
        y_list = list([y1_, y2_, y3_, y4_])
        print('两个list为',list(x_list), list(y_list))
        min_x, min_y, max_x ,max_y = mx_my(x_list, y_list)
        item_xmin.firstChild.data = min_x+1
        item_ymin.firstChild.data = min_y+1
        item_xmax.firstChild.data = max_x-1
        item_ymax.firstChild.data = max_y-1
        
    dst_xml_file = dst_xml_path + '/' + xml_file.split("/")[-1][:-4] + "_pad_"  + str(angle) + ".xml"

    with open(dst_xml_file, 'w') as fh:
        dom.writexml(fh)
num = 0        
for image_file in os.listdir(src_img_path):
    num = num +1
    img_name = src_img_path + '/' + image_file
    print("当前扩增的图像为：{}".format(img_name))
    src_img = cv.imread(img_name, cv.IMREAD_COLOR)
    h, w = src_img.shape[:2]
    print("高 宽",h,w)
    if h == w :
        # pad_size =h
        # dst_pad_img = pad_image(src_img, pad_size, pad_size)
        dst_pad_rot_img = rotate_image(src_img, angle_tran)
        dst_pad_rot_img_name = dst_img_path + '/' + image_file[:-4] + '_pad_' + str(angle_tran) + ".jpg"
        cv.imwrite(dst_pad_rot_img_name, dst_pad_rot_img, [int(cv.IMWRITE_JPEG_QUALITY), 100])
        xml_name = src_xml_path + '/' + image_file[:-4] + ".xml"
        print("当前改写的xml文件为：{}".format(xml_name))
        #print(xml_name, dst_xml_path, pad_width, pad_higth, angle_tran)
        xml_process(xml_name, dst_xml_path, w, h, angle_tran)
    elif h > w:
        pad_size = h
        dst_pad_img = pad_image(src_img, pad_size, pad_size)
        dst_pad_rot_img = rotate_image(dst_pad_img, angle_tran)
        dst_pad_rot_img_name = dst_img_path + '/' + image_file[:-4] + '_pad_' + str(angle_tran) + ".jpg"
        dst_pad_rot_img = dst_pad_rot_img[(h-w)/2:(h+w)/2, :]
        cv.imwrite(dst_pad_rot_img_name, dst_pad_rot_img, [int(cv.IMWRITE_JPEG_QUALITY), 100])

        xml_name = src_xml_path + '/' + image_file[:-4] + ".xml"
        print("当前改写的xml文件为：{}".format(xml_name))
        #print(xml_name, dst_xml_path, pad_width, pad_higth, angle_tran)
        xml_process(xml_name, dst_xml_path, h, w, angle_tran)

    else:
        pad_size = w
        dst_pad_img = pad_image(src_img, pad_size, pad_size)
        dst_pad_rot_img = rotate_image(dst_pad_img, angle_tran)
        dst_pad_rot_img_name = dst_img_path + '/' + image_file[:-4] + '_pad_' + str(angle_tran) + ".jpg"
        dst_pad_rot_img = dst_pad_rot_img[:,(w-h)/2:(h+w)/2]
        cv.imwrite(dst_pad_rot_img_name, dst_pad_rot_img, [int(cv.IMWRITE_JPEG_QUALITY), 100])

        xml_name = src_xml_path + '/' + image_file[:-4] + ".xml"
        print("当前改写的xml文件为：{}".format(xml_name))
        #print(xml_name, dst_xml_path, pad_width, pad_higth, angle_tran)
        xml_process(xml_name, dst_xml_path, h, w, angle_tran)
    
    if num == 2000:
        break
