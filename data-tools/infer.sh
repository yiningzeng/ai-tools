#!/bin/bash
python tools/infer_simple.py --cfg configs/getting_started/tutorial_1gpu_e2e_faster_rcnn_R-50-FPN.yaml --output-dir detectron/datasets/data/result/infer --output-ext jpg --image-ext bmp --wts detectron/datasets/data/result/train/coco_2014_train/generalized_rcnn/model_final.pkl detectron/datasets/data/Gray
