#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
from timeit import time
import warnings
import sys
import argparse
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO
from intersection import any_intersection

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
warnings.filterwarnings('ignore')


def main(yolo):
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help="input file")
    parser.add_argument('output', help="output file")
    parser.add_argument('--show', help="show intermediate results on screen",
                        action='store_true')
    parser.add_argument('--line', '-L', help="counting line: x1,y1,x2,y2",
                        default=None)
    args = parser.parse_args()

    basedir = os.getenv('DEEPSORTHOME','.')

    input_filename = args.input
    output_filename = args.output

   # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    
   # deep_sort 
    model_filename = '{}/model_data/mars-small128.pb'.format(basedir)
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    delcount=0
    db = {}
    def check_track(i, frame=None):
        nonlocal delcount
        if i in db and len(db[i]) > 1:
            if any_intersection(countline[0], countline[1], np.array(db[i])):
                if frame is not None:
                    pts = np.array(db[i]).reshape((-1,1,2))
                    cv2.polylines(frame, [np.int32(pts)], False, [0,0,255], 6)
                delcount+=1
                print("delcount={}".format(delcount))
            db[i] = []

    writeVideo_flag = True 
    
    video_capture = cv2.VideoCapture(input_filename)
    #video_capture = cv2.VideoCapture('jaarbeursplein.mp4')
    image_width = 0
    image_height = 0
    if writeVideo_flag:
    # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        image_width = w
        h = int(video_capture.get(4))
        image_height = h
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        out = cv2.VideoWriter(output_filename, fourcc, 15, (w, h))
        #list_file = open('detection.txt', 'w')
        frame_index = -1 

    if args.line is None:
        countline = np.array([[0,h/2],[w,h/2]],dtype=int)
    else:
        countline = np.array(list(map(int,args.line.strip().split(','))),dtype=int).reshape(2,2)

    fps = 0.0

    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break
        t1 = time.time()

       # image = Image.fromarray(frame)
        t1yolo = time.time()
        # following line accounts for about 15-20ms
        image = Image.fromarray(frame[...,::-1]) #bgr to rgb
        # YOLO accounts for about 40ms on ponder.cl
        boxs = yolo.detect_image(image)
        t2yolo = time.time()
        # features and tracking get more expensive as there are more things to track
        # writes to disk are fairly substantial as well

        # print("box_num",len(boxs))
        t1feat = t2yolo
        features = encoder(frame,boxs)
        t2feat = time.time()
        
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        
        # Run non-maxima suppression.
        t1prep = time.time()
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        t2prep = time.time()
        
        # Call the tracker
        t1trac = time.time()
        tracker.predict()
        tracker.update(detections)
        a = (countline[0,0], countline[0,1])
        b = (countline[1,0], countline[1,1])
        cv2.line(frame, a, b, (0,0,255), 3)
        for track in tracker.deleted_tracks:
            i = track.track_id
            if track.is_deleted():
                check_track(track.track_id, frame)

        for track in tracker.tracks:
            i = track.track_id
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            if i not in db:
                db[i] = []
            db[i].append(track.mean[:2].copy())
            if len(db[i]) > 1:
                pts = np.array(db[i]).reshape((-1,1,2))
                cv2.polylines(frame, [np.int32(pts)], False, [255,0,255], 3)
            bbox = track.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
            cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)
            
        cv2.putText(frame, str(delcount),(0, image_height-5),0, 5e-3 * 200, (0,0,255),2)
        if args.show:
            cv2.imshow('', frame)
        
        t2trac = time.time()
        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1
            #list_file.write(str(frame_index)+' ')
            #if len(boxs) != 0:
            #   for i in range(0,len(boxs)):
            #       list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ')
            #list_file.write('\n')
            
        #fps  = ( fps + (1./(time.time()-t1)) ) / 2
        #print("fps= %f"%(fps))
        t2 = time.time()
        print("Frame processing time={:.0f}ms (yolo={:.0f}ms prep={:.0f}ms feat={:.0f}ms trac={:.0f}ms)".format(1000*(t2 - t1), 1000*(t2yolo - t1yolo), 1000*(t2prep - t1prep), 1000*(t2feat - t1feat), 1000*(t2trac - t1trac)))
        
        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for track in tracker.tracks:
        check_track(track.track_id)

    print(delcount)
    video_capture.release()
    if writeVideo_flag:
        out.release()
        #list_file.close()
    if args.show:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main(YOLO())
