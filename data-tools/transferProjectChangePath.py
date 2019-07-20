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

path = "/home/baymin/daily-work/new-work/素材/yunsheng_date/6and4/"
new_path="/home/baymin/daily-work"
old_path="/media/baymin/c731be01-5353-4600-8df0-b766fc1f9b80"
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
            print("dodo" + json_file)
            try:
                with open(json_file, 'r') as fp:
                    data = json.load(fp)  # 加载json文件
                    data['asset']['path'] = str(data['asset']['path']).replace(old_path, new_path)
                json.dump(data, open(json_file, 'w'), indent=4, cls=MyEncoder)  # indent=4 更加美观显示
            except:
                print("err" + json_file)

if __name__ == '__main__':
    labelme_json=glob.glob(path+'*.json')
    ChangePath(labelme_json)
