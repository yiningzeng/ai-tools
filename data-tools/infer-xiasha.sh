#!/bin/bash
python tools/infer_simple.py --cfg detectron/datasets/data/aodelu/aodelu.yaml --output-dir detectron/datasets/data/result/xiasha --output-ext jpg --image-ext bmp --wts detectron/datasets/data/result/models5-17/coco_2014_train/generalized_rcnn/model_final.pkl detectron/datasets/data/new/xiasha

