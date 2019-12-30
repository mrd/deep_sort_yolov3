#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('line', help='line x1,y1,x2,y2')
parser.add_argument('file1', help="first file")
parser.add_argument('file2', help="second file")
args = parser.parse_args()

v1 = cv2.VideoCapture(args.file1)
v2 = cv2.VideoCapture(args.file2)

r1, f1 = v1.read()
r2, f2 = v2.read()

if r1 and r2:
    g1 = cv2.cvtColor(f1,cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(f2,cv2.COLOR_BGR2GRAY)
    
    # det1 = cv2.xfeatures2d_SURF.create(hessianThreshold=400)
    # det2 = cv2.xfeatures2d_SURF.create(hessianThreshold=400)
    # key1 = det1.detect(g1)
    # key2 = det2.detect(g2)

    det1 = cv2.xfeatures2d_SIFT.create()
    key1 = det1.detect(g1)
    key2 = det1.detect(g2)

    pts1 = cv2.KeyPoint_convert(key1)
    pts2 = cv2.KeyPoint_convert(key2)
    l = min(len(pts1), len(pts2))
    pts1 = pts1[:l]
    pts2 = pts2[:l]

    #    import pdb; pdb.set_trace()
    
    trans,_ = cv2.estimateAffinePartial2D(pts1, pts2)
    #trans = np.matmul(trans,np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float))
    #print(trans)

    line = np.expand_dims(np.array(list(map(int,args.line.strip().split(','))),dtype=int).reshape(2,2),axis=0)
    res = np.array(cv2.transform(line, trans),dtype=int)[0]
    print("{},{},{},{}".format(res[0,0],res[0,1],res[1,0],res[1,1]))

v1.release()
v2.release()

    
