# -*- coding:utf-8 -*-
# !/usr/bin/env python

import argparse
import json
import os
import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np
import glob
import PIL.Image
import PIL.ImageDraw

path = "/home/baymin/test/"
new_path="change_1_ok"
old_path="/media/baymin/"
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

class ChangePath(object):
    def __init__(self,labelme_json=[]):
        self.labelme_json=labelme_json
        self.save_json()

    def save_json(self):

        for num, json_file in enumerate(self.labelme_json):
            with open(json_file, 'r') as fp:
                data = json.load(fp)  # 加载json文件
                data['asset']['path'] = str(data['asset']['path']).replace(old_path, new_path)
            json.dump(data, open(json_file, 'w'), indent=4, cls=MyEncoder)  # indent=4 更加美观显示

if __name__ == '__main__':
    labelme_json=glob.glob(path+'*.json')
    ChangePath(labelme_json)
