# -*- coding:utf-8 -*-
# !/usr/bin/env python

import argparse
import json
import os
import matplotlib.pyplot as plt
import skimage.io as io
import cv2
import numpy as np
import glob
import PIL.Image
import PIL.ImageDraw

# path="/media/baymin/项目/daily-data/熊猫第一批全部数据-待格式转换--0614/All_TrainningDate-1-----0613/训练大杂烩/"
path = "/media/baymin/QTING2.0T/熊猫第一批全部数据-待格式转换--0614/曾伟数据/a/"
new_label="change_1_ok"
old_label="1_ok"
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

class otherJson2Labelme(object):
    def __init__(self,labelme_json=[]):
        '''
        :param labelme_json: 所有labelme的json文件路径组成的列表
        :param save_json_path: json保存位置
        '''
        self.labelme_json=labelme_json
        self.shapes=[]
        # self.imgHeight=0
        # self.imgWidth=0
        self.save_json()

    def save_json(self):

        for num, json_file in enumerate(self.labelme_json):
            with open(json_file, 'r') as fp:
                data = json.load(fp)  # 加载json文件
                for shape in data['shapes']:
                    label = shape['label']
                    if(label == old_label):
                        shape['label'] = new_label


                    # twoPoint=object['data']
                    # # region 2点换4点
                    # points=[]
                    # points.append([int(twoPoint[0]), int(twoPoint[1])])
                    # points.append([int(twoPoint[2]), int(twoPoint[1])])
                    # points.append([int(twoPoint[0]), int(twoPoint[2])])
                    # points.append([int(twoPoint[2]), int(twoPoint[3])])
                    # # endregion

                    # object['points']=points
                    self.shapes.append(shape)
            # 保存json文件
            data_coco = {}
            # data_coco['imgHeight'] = self.imgHeight
            # data_coco['imgWidth'] = self.imgWidth
            data_coco['shapes'] = self.shapes
            json.dump(data_coco, open(json_file, 'w'), indent=4, cls=MyEncoder)  # indent=4 更加美观显示

if __name__ == '__main__':
    labelme_json=glob.glob(path+'*.json')
    # labelme_json = glob.glob('./bird.json')
    # labelme_json=['./1.json']

    otherJson2Labelme(labelme_json)