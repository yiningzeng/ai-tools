#!/bin/bash
python tools/infer_simple_new.py --cfg detectron/datasets/data/xiasha-5-23/coco-json-export---0523/train-config.yaml --output-dir detectron/datasets/data/xiasha-5-23/infer --output-ext jpg --image-ext bmp --wts detectron/datasets/data/xiasha-5-23/train/coco_2014_train/generalized_rcnn/model_final.pkl --ng-output-dir detectron/datasets/data/xiasha-5-23/infer/ng detectron/datasets/data/new/xiasha

