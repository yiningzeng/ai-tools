# -*- coding:utf-8 -*-
# !/usr/bin/env python
from ctypes import *
import math
import random
import glob
import os
import sys
import json
import time
import cv2
import base64
from flask import Flask, request
import numpy as np
reload(sys)
sys.setdefaultencoding("utf8")
app = Flask(__name__)

cfg = "/home/baymin/daily-work/new-work/ab-darknet-localhost/yunsheng/yolov3-voc-test.cfg"
model = "/home/baymin/daily-work/new-work/ab-darknet-localhost/yunsheng/backup/yolov3-voc_110000.weights"
data = "/home/baymin/daily-work/new-work/ab-darknet-localhost/yunsheng/voc.data"
save_path = "/home/baymin/daily-work/new-work/ab-darknet-localhost/yunsheng/test7-10/result/" # 检测结果保存路径 以 / 结尾
img_path = "/home/baymin/daily-work/new-work/ab-darknet-localhost/yunsheng/test7-10/" # 检测的图片路径 以 / 结尾
score = 0.01

img_ext = "*.jpg" # 图片格式
lib_dll = "/home/baymin/daily-work/new-work/ab-darknet-localhost/libdark.so"

is_detecton = False

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    

lib = CDLL(lib_dll, RTLD_GLOBAL)
#lib = CDLL("libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    global is_detecton
    is_detecton = True
    im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    is_detecton = False
    return res
    

@app.route('/pandas/', methods=['GET', 'POST'])
def main():    
    if request.method == 'POST' or request.method == 'GET':
        global is_detecton
        img = base64.b64decode(str(request.form['photo']))       
        f_name = str(request.form['name'])
        # print("sdfsdafasdfsdafa",f_name)
        img = np.fromstring(img, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        out_put_path = save_path + f_name
        cv2.imwrite(out_put_path, img)

        ret = {"num": 0, "label_str": "OK,", "points": [], "img_name": f_name, "process_time": "s"}    #######

        while 1:
            if is_detecton:
                time.sleep(0.1)
            else:
                break

        start_time = time.time()
        r = detect(net, meta, out_put_path)
        end_time = time.time()
        total_time = end_time - start_time
        ret["num"] = len(r)
        print("--------------------------")
        for i in range(len(r)):
            print(i+1)
            x1 = r[i][2][0] - r[i][2][2] / 2
            y1 = r[i][2][1] - r[i][2][3] / 2
            x2 = r[i][2][0] + r[i][2][2] / 2
            y2 = r[i][2][1] + r[i][2][3] / 2
            print r[i]
            if r[i][1] > score:
                print "save > " + out_put_path
                # ret["label_str"] = ret["label_str"] + str(r[i][0]) + " " + str(r[i][1]) + ","
                ret["label_str"] = "NG"
                ret["points"].append(r[i])
                cv2.putText(img, str(r[i][0]), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                cv2.imwrite(out_put_path, img)
        print("\ntotal_time: " + str(total_time) + "\nnumber: " + str(ret["num"]))
        print("--------------------------")
        ret["process_time"] = total_time
        ret["points"] = str(ret["points"])
        ret = json.dumps(ret)
        return ret

if __name__ == "__main__":
    #net = load_net("cfg/densenet201.cfg", "/home/pjreddie/trained/densenet201.weights", 0)
    #im = load_image("data/wolf.jpg", 0, 0)
    #meta = load_meta("cfg/imagenet1k.data")
    #r = classify(net, meta, im)
    #print r[:10]


    net = load_net(cfg, model, 0)
    meta = load_meta(data)
    # app.run(host="0.0.0.0", port=8889)
    app.run(host="0.0.0.0", port=sys.argv[1])

    # img_list=glob.glob(img_path+img_ext)

    # for num, img_file in enumerate(img_list):
        # r = detect(net, meta, img_file)
    #     for i in range(len(r)):
    #         x1 = r[i][2][0] - r[i][2][2] / 2
    #         y1 = r[i][2][1] - r[i][2][3] / 2
    #         x2 = r[i][2][0] + r[i][2][2] / 2
    #         y2 = r[i][2][1] + r[i][2][3] / 2
    #         print("----")
    #         print r[i]
    #         font = cv2.FONT_HERSHEY_SIMPLEX
    #         if r[i][1] > 0.1:
    #             filepath, tmpfilename = os.path.split(img_file)
    #             out_put_path = save_path + tmpfilename
    #             print img_file + " > " + out_put_path
    #             cv2.putText(numpy_image, str(r[i][0]), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    #             cv2.rectangle(numpy_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
    #             cv2.imwrite(out_put_path, numpy_image)
    #             print("----")
