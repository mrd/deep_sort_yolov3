#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

from st7735_ijl20.st7735 import ST7735
import os
import io
from timeit import time
import warnings
import sys
import argparse
import numpy as np
import signal
import picamera
from picamera import Color
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageColor
from PIL import Image
#from yolo import YOLO
#from yolo_v3_tflite import YOLO_TFLITE
from ssd_mobilenet import SSD_MOBILENET
from intersection import any_intersection, intersection

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
warnings.filterwarnings('ignore')

class SigTerm(SystemExit): pass
def sigterm(sig,frm): raise SigTerm
signal.signal(15,sigterm)

FONT_SMALL = ImageFont.truetype('fonts/truetype/freefont/FreeSansBold.ttf', 10)
FONT_LARGE = ImageFont.truetype('fonts/truetype/freefont/FreeSansBold.ttf', 12)

DISPLAY_WIDTH = 160                # LCD panel width in pixels
DISPLAY_HEIGHT = 128               # LCD panel height

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

def clamp(minvalue, value, maxvalue):
    return max(minvalue, min(value, maxvalue))

def main():
    ##################################################
    # Initialise LCD
    ## LCD = LCD_1in8.LCD()
    LCD = ST7735()
    ## Lcd_ScanDir = LCD_1in8.SCAN_DIR_DFT
    ## LCD.LCD_Init(Lcd_ScanDir)
    LCD.begin()
    ## screenbuf = Image.new("RGB", (LCD.LCD_Dis_Column, LCD.LCD_Dis_Page), "WHITE")
    screenbuf = Image.new("RGBA", (DISPLAY_WIDTH, DISPLAY_HEIGHT), (0,0,0,0))
    bgbuf = Image.new("RGBA", (DISPLAY_WIDTH, DISPLAY_HEIGHT), (0,0,0,0))
    draw = ImageDraw.Draw(screenbuf)
    draw.text((33, 22), 'Initialising...', fill = "BLUE", font = FONT_LARGE)
    ## LCD.LCD_PageImage(screenbuf)
    LCD.display(screenbuf)

    ##################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--line', '-L', help="counting line: x1,y1,x2,y2",
                        default=None)
    parser.add_argument('--model', help='File path of .tflite file.', required=True)
    parser.add_argument('--labels', help='File path of labels file.', required=True)
    args = parser.parse_args()

    objd = SSD_MOBILENET(wanted_label='person', model_file=args.model, label_file=args.labels)
    basedir = os.getenv('DEEPSORTHOME','.')

   # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    
   # deep_sort 
    model_filename = '{}/model_data/mars-small128.pb'.format(basedir)
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric,max_age=10)
    delcount=0
    intcount=0
    poscount=0
    negcount=0
    db = {}
    ratios = np.array([DISPLAY_WIDTH/CAMERA_WIDTH, DISPLAY_HEIGHT/CAMERA_HEIGHT],dtype=float)
    def check_track(i, draw=None):
        nonlocal delcount
        nonlocal ratios
        nonlocal cameracountline
        if i in db and len(db[i]) > 1:
            if any_intersection(cameracountline[0], cameracountline[1], np.array(db[i])):
                if draw is not None:
                    pts = np.array(db[i]).reshape((-1,1,2)) * ratios
                    #draw.line(list(np.int32(pts).reshape(-1)), fill=(0,0,255), width=6)
                delcount+=1
                print("delcount={}".format(delcount))
            db[i] = []

    if args.line is None:
        w, h = DISPLAY_WIDTH, DISPLAY_HEIGHT
        countline = np.array([[w/2,0],[w/2,h]],dtype=int)
    else:
        countline = np.array(list(map(int,args.line.strip().split(','))),dtype=int).reshape(2,2)

    cameracountline = countline.astype(float) / ratios

    fps = 0.0
    frameTime = time.time()*1000
    with picamera.PiCamera(resolution=(CAMERA_WIDTH, CAMERA_HEIGHT), framerate=30) as camera:
        camera.start_preview(alpha=255)
        camera.annotate_foreground = Color('black')
        camera.annotate_background = Color('white')
        screenarray = np.array(screenbuf.resize((CAMERA_WIDTH, CAMERA_HEIGHT), Image.ANTIALIAS))
        overlay = camera.add_overlay(screenarray.tobytes(), layer=3, alpha=64)
        try:
            stream = io.BytesIO()
            for _ in camera.capture_continuous(stream, format='jpeg', use_video_port=True):
                stream.seek(0)
            
                image = Image.open(stream)
                t1 = time.time()
                t1objd = time.time()
                boxs = objd.detect_image(image)
                # boxs is in tlwh format
                t2objd = time.time()

                stream.seek(0)
                stream.truncate()

                # set background image and clear screen
                bgbuf.paste(image.resize((DISPLAY_WIDTH, DISPLAY_HEIGHT)))
                draw.rectangle([0,0,DISPLAY_WIDTH,DISPLAY_HEIGHT], fill=0, outline=0)

                # features and tracking get more expensive as there are more things to track
                # writes to disk are fairly substantial as well

                # print("box_num",len(boxs))
                t1feat = t2objd
                frame = np.array(image)
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
                draw.line([a, b], fill=(0,0,255), width=3)

                for track in tracker.deleted_tracks:
                    i = track.track_id
                    if track.is_deleted():
                        check_track(track.track_id, draw)

                for track in tracker.tracks:
                    i = track.track_id
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    if i not in db:
                        db[i] = []
                    db[i].append(track.mean[:2].copy())
                    if len(db[i]) > 1:
                        pts = (np.array(db[i]).reshape((-1,1,2)) * ratios).reshape(-1)
                        draw.line(list(np.int32(pts)), fill=(255,0,255), width=3)
                        p1 = cameracountline[0]
                        q1 = cameracountline[1]
                        p2 = np.array(db[i][-1])
                        q2 = np.array(db[i][-2])
                        cp = np.cross(q1 - p1,q2 - p2)
                        if intersection(p1,q1,p2,q2):
                            intcount+=1
                            print("track_id={} just intersected camera countline; cross-prod={}; intcount={}".format(i,cp,intcount))
                            draw.line(list(np.int32(pts)[-4:]), fill=(0,0,255), width=5)
                            if cp >= 0:
                                poscount+=1
                            else:
                                negcount+=1

                    bbox = list((track.to_tlbr().reshape(2,2) * ratios).astype(int).reshape(-1))
                    draw.rectangle(bbox,outline=(255,255,255))
                    draw.text(bbox[:2],str(track.track_id), fill=(0,255,0), font=FONT_SMALL)

                for det in detections:
                    bbox = list((det.to_tlbr().reshape(2,2) * ratios).astype(int).reshape(-1))
                    draw.rectangle(bbox,outline=(255,0,0))
                
                t2trac = time.time()

                draw.text((0, DISPLAY_HEIGHT-10), str(negcount), fill=(255,0,0), font=FONT_LARGE)
                draw.text((DISPLAY_WIDTH/2, DISPLAY_HEIGHT-10), str(abs(negcount-poscount)), fill=(0,255,0), font=FONT_LARGE)
                draw.text((DISPLAY_WIDTH-10, DISPLAY_HEIGHT-10), str(poscount), fill=(0,0,255), font=FONT_LARGE)
                bgbuf.paste(screenbuf)
                LCD.display(bgbuf)
                screenarray = np.array(screenbuf.resize((CAMERA_WIDTH, CAMERA_HEIGHT), Image.ANTIALIAS))
                overlay.update(screenarray.tobytes())
            
                t2 = time.time()

                print("Frame processing time={:.0f}ms (objd={:.0f}ms prep={:.0f}ms feat={:.0f}ms trac={:.0f}ms)".format(1000*(t2 - t1), 1000*(t2objd - t1objd), 1000*(t2prep - t1prep), 1000*(t2feat - t1feat), 1000*(t2trac - t1trac)))
            

            for track in tracker.tracks:
                check_track(track.track_id)

            print(delcount)

        finally:
          camera.stop_preview()
          LCD.LCD_Clear()

if __name__ == '__main__':
    main()
