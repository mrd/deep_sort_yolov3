#!/bin/bash
python3 live-counter.py --model=ssd_mobilenet.tflite --labels=coco_labelmap.txt $*
