# -*- coding:utf-8 -*-
# !/usr/bin/env python

import argparse
import json
import os
import matplotlib.pyplot as plt
import skimage.io as io
import cv2
import numpy as np
import copy
import glob
import PIL.Image
import PIL.ImageDraw

# path="/media/baymin/项目/daily-data/熊猫第一批全部数据-待格式转换--0614/All_TrainningDate-1-----0613/训练大杂烩/"
sourceJson = "/home/baymin/work/6-17/source.json"
targetJson='/home/baymin/work/6-17/target.json'
saveJson='/home/baymin/work/6-17/final.json'



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

class append2coco(object):
    def __init__(self, sourceJson, targetJson):
        '''
        :param labelme_json: 所有labelme的json文件路径组成的列表
        :param save_json_path: json保存位置
        '''
        self.sourceJson=sourceJson
        self.sourceImages=[]
        self.sourceCategories=[]
        self.sourceAnnotations=[]

        self.targetJson=targetJson
        self.targetImages=[]
        self.targetCategories=[]
        self.targetAnnotations=[]

        self.images = [] # 单独文件名string数组
        self.categories = [] # 单独标签string数组

        self.append()

    def data_transfer(self):
        print("正在处理：", self.sourceJson)
        with open(self.sourceJson, 'r') as fp:
            data = json.load(fp)  # 加载json文件
            self.sourceImages = data['images']
            self.sourceCategories = data['categories']
            self.sourceAnnotations = data['annotations']
        with open(self.targetJson, 'r') as fp:
            data = json.load(fp)  # 加载json文件
            self.targetImages = data['images']

            for num, images in enumerate(self.targetImages):
                self.images.append(images['file_name'])

            self.targetCategories = data['categories']

            for num, category in enumerate(self.targetCategories):
                self.categories.append(category['name'])

            self.targetAnnotations = data['annotations']

        self.doAppend()

    def doAppend(self):
        for image in self.sourceImages:
            tempImage = image.copy()
            fileName = tempImage['file_name']
            sourceImageId = tempImage['id']
            # region 重复文件名的不合并，保留target的文件
            if fileName not in self.images:
                self.images.append(fileName)

                # 更新源image的id +1 并加入目标
                tempImage['id'] = self.targetImages[len(self.targetImages)-1]['id'] + 1
                self.targetImages.append(tempImage)

                # 更新源Annotation的imageId
                sourceAnnotation = self.getSourceAnnotationByImageId(sourceImageId)
                if sourceAnnotation is not None:
                    tempAnnotation = sourceAnnotation.copy()
                    tempAnnotation['image_id'] = tempImage['id']

                    # 更新源categories的id +1 并加入目标
                    # 1.通过标签name查找目标categories是否存在，不存在新增，存在不新增
                    sourceCategory = self.getSourceCategoryByTagId(tempAnnotation['category_id'])

                    if sourceCategory is not None: # 已经存在标签
                        tempCategory = sourceCategory.copy()
                        targetCategoryId = self.getTargetCategoryIdByTagName(tempCategory['name'])
                        if targetCategoryId is not None:
                            tempCategory['id'] = self.getTargetCategoryIdByTagName(tempCategory['name'])
                        else:
                            tempCategory['id'] = self.targetCategories[len(self.targetCategories) - 1]['id'] + 1
                            self.targetCategories.append(tempCategory)

                    tempAnnotation['category_id'] = tempCategory['id']
                    tempAnnotation['id'] = self.targetAnnotations[len(self.targetAnnotations) - 1]['id'] + 1
                    # 更新源Annotation的category_id 并加入目标
                    self.targetAnnotations.append(tempAnnotation)
            self.data_coco = self.data2coco()
            # 保存json文件
            json.dump(self.data_coco, open(saveJson, 'w'), indent=4, cls=MyEncoder)  # indent=4 更加美观显示
            # endregion

    '''
    通过imageid 查找源json的annotation
    '''
    def getSourceAnnotationByImageId(self, imageId):
        for num, annotation in enumerate(self.sourceAnnotations):
            if annotation['image_id'] == imageId:
                return annotation
        return None

    '''
    
    '''
    def getSourceCategoryByTagId(self, tagId):
        for num, category in enumerate(self.sourceCategories):
            if category['id'] == tagId:
                return category
        return None

    '''
    通过源json的tagName查找目标已存在的标签id
    '''
    def getTargetCategoryIdByTagName(self, tagName):
        for num, category in enumerate(self.targetCategories):
            if category['name'] == tagName:
                return category['id']
        return None



    def data2coco(self):
        data_coco={}
        data_coco['images']=self.targetImages
        data_coco['categories']=self.targetCategories
        data_coco['annotations']=self.targetAnnotations
        return data_coco

    def append(self):
        self.data_transfer()



if __name__ == '__main__':

    # labelme_json = glob.glob('./bird.json')
    # labelme_json=['./1.json']

    append2coco(sourceJson, targetJson)