#! /usr/bin/env python
# -*- coding:UTF-8 -*-

from __future__ import print_function
import cv2 as cv
import numpy as np
import os
import sys

cur_path = os.getcwd()
dir_abs_path = str(sys.argv[1])
dir_path = os.path.join(cur_path, dir_abs_path)

for i in os.listdir(dir_path):
    src_img_dir = dir_path + "/" + i
    for j in os.listdir(src_img_dir):
        if not j[-4:] == ".jpg":
            src_img_name =  src_img_dir + "/" + j
            os.remove(src_img_name)
            pritn(j)