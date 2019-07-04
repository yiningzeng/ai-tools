#! /usr/bin/env python
# -*- coding:UTF-8 -*-

from __future__ import print_function
import cv2 as cv
import numpy as np
import os
import sys

def makedir_path(the_path):
    if not os.path.exists(the_path):
        os.makedirs(the_path, mode=0o777)
    print("{} 该目录不存在，已创建成功".format(the_path))

    return 0

cur_path = os.getcwd()
dir_abs_path = str(sys.argv[1])
dst_abs_path = str(sys.argv[2])
dir_path = os.path.join(cur_path, dir_abs_path)
dst_path = os.path.join(cur_path, dst_abs_path)

makedir_path(dst_path)

for i in os.listdir(dir_path):
    src_img_dir = dir_path + "/" + i
    for j in os.listdir(src_img_dir):
        src_img_name =  src_img_dir + "/" + j
        src_img = cv.imread(src_img_name, cv.IMREAD_COLOR)
        makedir_path(dst_path + "/" + i + "/")
        dst_img_name = dst_path + "/" + i + "/" + j[:-4] + ".jpg"
        cv.imwrite(dst_img_name, src_img)