# -*- coding:utf-8 -*-
# !/usr/bin/env python
import argparse
import json
import os
import matplotlib.pyplot as plt
import skimage.io as io
import cv2
import numpy as np
from math import *
import copy
import glob
import PIL.Image
import PIL.ImageDraw

# path="/media/baymin/项目/daily-data/熊猫第一批全部数据-待格式转换--0614/All_TrainningDate-1-----0613/训练大杂烩/"
sourceJson = "/media/baymin/c731be01-5353-4600-8df0-b766fc1f9b80/new-work/coco/annotations/instances_train2014.json"
sourceImgPath = '/media/baymin/c731be01-5353-4600-8df0-b766fc1f9b80/new-work/coco/coco_train2014/'
saveJson = '/media/baymin/c731be01-5353-4600-8df0-b766fc1f9b80/new-work/coco/coco_train2014/save.json'
saveImgPath = '/media/baymin/c731be01-5353-4600-8df0-b766fc1f9b80/new-work/coco/saveimg/'
angel = 179

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
    def __init__(self):
        '''
        :param labelme_json: 所有labelme的json文件路径组成的列表
        :param save_json_path: json保存位置
        '''
        self.sourceJson = sourceJson
        self.sourceImages = []
        self.sourceCategories = []
        self.sourceAnnotations = []

        self.targetAnnotations = []
        self.height = 0
        self.width = 0

        self.append()

    # 对图片进行旋转的函数

    def rotate_image(self, img, angle, scale):  # 旋转，angle是角度值，scale 缩放比例
        h, w = img.shape[:2]
        angle %= 360

        # (2)旋转后的尺寸
        # @radians(),角度转换为弧度
        heightNew = int(w * fabs(sin(radians(angle))) + h * fabs(cos(radians(angle))))
        widthNew = int(h * fabs(sin(radians(angle))) + w * fabs(cos(radians(angle))))
        # (3)求旋转矩阵，以图片中心点为旋转中心
        matRotation = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        matRotation[0, 2] += (widthNew - w) / 2
        matRotation[1, 2] += (heightNew - h) / 2
        imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(0, 0, 0))
        return imgRotation, widthNew, heightNew, w, h

    def rot_compute(self, x0, y0, x1, y1, angle_):
        dx = x1 - x0
        dy = y1 - y0
        #    print("笛卡尔坐标下的距离为：{0}{1}".format(dx, dy))
        magnitude, angle = cv2.cartToPolar(float(dx), float(dy))
        #    print("距离{}角度{}".format(magnitude, angle))
        angle = angle - angle_
        dx_, dy_ = cv2.polarToCart(magnitude, angle, True)
        #    print(dx_, dy_)
        x2 = dx_ + x0
        y2 = y0 + dy_
        #    print(x2[0], y2[0])
        return x2[0], y2[0]

    def update_annotation(self, annotation, points):
        annotation['segmentation'] = [list(np.asarray(points).flatten())]
        annotation['bbox'] = list(map(float, self.getbbox(points)))
        annotation['area'] = annotation['bbox'][2] * annotation['bbox'][3]
        return annotation

    def getbbox(self, points):
        # img = np.zeros([self.height,self.width],np.uint8)
        # cv2.polylines(img, [np.asarray(points)], True, 1, lineType=cv2.LINE_AA)  # 画边界线
        # cv2.fillPoly(img, [np.asarray(points)], 1)  # 画多边形 内部像素值为1
        polygons = points

        mask = self.polygons_to_mask([self.height, self.width], polygons)
        return self.mask2box(mask)

    def mask2box(self, mask):
        '''从mask反算出其边框
        mask：[h,w]  0、1组成的图片
        1对应对象，只需计算1对应的行列号（左上角行列号，右下角行列号，就可以算出其边框）
        '''
        # np.where(mask==1)
        index = np.argwhere(mask == 1)
        rows = index[:, 0]
        clos = index[:, 1]
        # 解析左上角行列号
        left_top_r = np.min(rows)  # y
        left_top_c = np.min(clos)  # x

        # 解析右下角行列号
        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)

        # return [(left_top_r,left_top_c),(right_bottom_r,right_bottom_c)]
        # return [(left_top_c, left_top_r), (right_bottom_c, right_bottom_r)]
        # return [left_top_c, left_top_r, right_bottom_c, right_bottom_r]  # [x1,y1,x2,y2]
        return [left_top_c, left_top_r, right_bottom_c - left_top_c,
                right_bottom_r - left_top_r]  # [x1,y1,w,h] 对应COCO的bbox格式

    def polygons_to_mask(self, img_shape, polygons):
        mask = np.zeros(img_shape, dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        return mask

    def data_transfer(self):
        print("正在处理：", self.sourceJson)
        with open(self.sourceJson, 'r') as fp:
            data = json.load(fp)  # 加载json文件
            self.sourceImages = data['images']
            self.sourceCategories = data['categories']
            self.sourceAnnotations = data['annotations']
        # with open(self.targetJson, 'r') as fp:
        #     data = json.load(fp)  # 加载json文件
        #     self.targetImages = data['images']
        #
        #     for num, images in enumerate(self.targetImages):
        #         self.images.append(images['file_name'])
        #
        #     self.targetCategories = data['categories']
        #
        #     for num, category in enumerate(self.targetCategories):
        #         self.categories.append(category['name'])
        #
        #     self.targetAnnotations = data['annotations']

        self.doAppend()

    def doAppend(self):
        for i, image in enumerate(self.sourceImages):
            image_name = image['file_name']
            h = image['height']
            w = image['width']
            n = 1
            while n < angel:
                if n % 90 == 0:
                    angle_ = (n/180) * pi
                    (filepath, tempfilename) = os.path.split(image_name)
                    (filename, extension) = os.path.splitext(tempfilename)
                    final_image_name = filepath + "/" + filename + "_" + str(n) + extension
                    print(final_image_name)
                    img = cv2.imread(sourceImgPath + image_name, 0)
                    rotate_image, widthNew, heightNew, widthOld, heightOld = self.rotate_image(img, n, 1)

                    self.width = widthNew
                    self.height = heightNew
                    # region 坐标转换
                    annotation = self.getSourceAnnotationByImageId(image['id'])
                    if annotation is None:
                        n = n + 1
                        continue
                    seg = np.array(annotation['segmentation'])

                    points = np.reshape(seg, (-1, 2))
                    points = np.delete(points, [len(points)-2, len(points)-1], 0) # 删除最后两个闭合点
                    new_points = [].copy()
                    for point in points:
                        # 在平面坐标上，任意点P(x1,y1)，绕一个坐标点Q(x2,y2)旋转θ角度后,新的坐标设为(x, y)
                        # pointx, pointy = widthOld/2, heightOld/2
                        x, y = point[0], point[1]
                        x0, y0 = w/2, h/2
                        magnitude, angle = cv2.cartToPolar(float(x), float(y))
                        angle = angle + angle_
                        srx, sry = cv2.polarToCart(magnitude, angle, True)
                        print(x0, y0)
                        # x1_, y1_ = self.rot_compute(int(widthOld / 2), int(heightOld / 2), x, y, angle_)
                        # x = (x1 - x2) * cos(n) - (y1 - y2) * sin(n) + x2
                        # y = (x1 - x2) * sin(n) + (y1 - y2) * cos(n) + y2
                        # new_points.append([int(round(x)), int(round(y))])
                        # (x, y)为要转的点，（pointx, pointy)为中心点，如果顺时针角度为angle
                        # dx = (x - pointx) * cos(n) - (y - pointy) * sin(n)
                        # dy = (x - pointx) * sin(n) + (y - pointy) * cos(n)
                        # dx = (x - pointx) * cos(n) + (y - pointy) * sin(n)
                        # dy = (y - pointy) * cos(n) - (x - pointx) * sin(n)

                        # magnitude, angle = cv2.cartToPolar(float(x), float(y))
                        # angle = angle + n
                        # dx_, dy_ = cv2.polarToCart(magnitude, angle, True)
                        # srx = dx_ + pointx
                        # sry = pointy - dy_
                        new_points.append([int(srx[0]), int(sry[0])])
                        # x, y = self.rot_compute(int(widthNew / 2), int(heightNew / 2), x1, y1, n)
                    new_points.append([new_points[0][0], new_points[1][1]])
                    new_points.append([new_points[1][0], new_points[0][1]])
                    self.targetAnnotations.append(self.update_annotation(annotation, points))
                    print(new_points)
                    cv2.rectangle(rotate_image, (new_points[0][0], new_points[0][1]),
                                  (new_points[2][0], new_points[2][1]), (0, 0, 255), 2)
                    cv2.imwrite(saveImgPath + final_image_name, rotate_image)
                    # print([list(np.asarray([[415, 501], [438, 617], [503, 638], [537, 529], [534, 531]]).flatten())])
                    # print('aaaaaa===========================')
                    # seg = np.array([415, 501, 438, 617, 503, 638, 537, 529, 534, 531, 415, 617, 438, 501])
                    # print(seg.shape)
                    # print('===========================')
                    # print(np.reshape(seg, (-1, 2)))
                    # print('===========================')
                    # endregion
                n = n + 1

    '''
    通过imageid 查找源json的annotation
    '''

    def getSourceAnnotationByImageId(self, imageId):
        for num, annotation in enumerate(self.sourceAnnotations):
            if annotation['image_id'] == imageId:
                return annotation.copy()
        return None

    '''

    '''

    def getSourceCategoryByTagId(self, tagId):
        for num, category in enumerate(self.sourceCategories):
            if category['id'] == tagId:
                return category
        return None


    def data2coco(self):
        data_coco = {}
        data_coco['images'] = self.sourceImages
        data_coco['categories'] = self.sourceCategories
        data_coco['annotations'] = self.targetAnnotations
        return data_coco

    def append(self):
        self.data_transfer()
        self.data_coco = self.data2coco()
        # 保存json文件
        json.dump(self.data_coco, open(saveJson, 'w'), indent=4, cls=MyEncoder)  # indent=4 更加美观显示
        # endregion


if __name__ == '__main__':
    # labelme_json = glob.glob('./bird.json')
    # labelme_json=['./1.json']

    append2coco()
