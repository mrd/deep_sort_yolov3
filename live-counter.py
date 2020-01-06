#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
no_lcd = True

if not no_lcd:
    from st7735_ijl20.st7735 import ST7735
import os
import io
from timeit import time
import warnings
import sys
import argparse
import numpy as np
import signal
import cv2
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

FONT_SMALL_LCD = ImageFont.truetype('fonts/truetype/freefont/FreeSansBold.ttf', 10)
FONT_SMALL_PRV = ImageFont.truetype('fonts/truetype/freefont/FreeSansBold.ttf', 40)
FONT_LARGE_LCD = ImageFont.truetype('fonts/truetype/freefont/FreeSansBold.ttf', 12)
FONT_LARGE_PRV = ImageFont.truetype('fonts/truetype/freefont/FreeSansBold.ttf', 48)

FONT_SMALL = {'lcd': FONT_SMALL_LCD, 'fbf': FONT_SMALL_PRV}
FONT_LARGE = {'lcd': FONT_LARGE_LCD, 'fbf': FONT_LARGE_PRV}

DISPLAY_WIDTH = 160                # LCD panel width in pixels
DISPLAY_HEIGHT = 128               # LCD panel height

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

FRAMEBUF_WIDTH = 480
FRAMEBUF_HEIGHT = 320

no_framebuf = False

def clamp(minvalue, value, maxvalue):
    return max(minvalue, min(value, maxvalue))

def main():
    global no_framebuf, no_lcd
    basedir = os.getenv('DEEPSORTHOME','.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--line', '-L', help="counting line: x1,y1,x2,y2",
                        default=None)
    parser.add_argument('--model', help='File path of .tflite file.', required=True)
    parser.add_argument('--labels', help='File path of labels file.', required=True)
    parser.add_argument('--no-framebuf', help='Disable framebuffer display',
                        required=False, action='store_true')
    parser.add_argument('--framebuffer', '-F', help='Framebuffer device',
                        default='/dev/fb1', metavar='DEVICE')
    parser.add_argument('--max-cosine-distance', help='Max cosine distance', metavar='N',
                        default=0.3)
    parser.add_argument('--nms-max-overlap', help='Non-Max-Suppression max overlap', metavar='N',
                        default=1.0)
    parser.add_argument('--max-iou-distance', help='Max Intersection-Over-Union distance',
                        metavar='N', default=0.7)
    parser.add_argument('--max-age', help='Max age of lost track', metavar='N',
                        default=10)
    parser.add_argument('--deepsorthome', help='Location of model_data directory',
                        metavar='PATH', default=basedir)
    args = parser.parse_args()
    basedir = args.deepsorthome

    no_framebuf = args.no_framebuf

    ##################################################
    # Initialise LCD
    ## LCD = LCD_1in8.LCD()
    if not no_lcd:
        LCD = ST7735()
        ## Lcd_ScanDir = LCD_1in8.SCAN_DIR_DFT
        ## LCD.LCD_Init(Lcd_ScanDir)
        LCD.begin()
        ## screenbuf = Image.new("RGB", (LCD.LCD_Dis_Column, LCD.LCD_Dis_Page), "WHITE")
        screenbuf = Image.new("RGBA", (DISPLAY_WIDTH, DISPLAY_HEIGHT), (0,0,0,0))
        drawLCD = ImageDraw.Draw(screenbuf)
    else:
        drawLCD = None

    framebuf = Image.new("RGBA", (FRAMEBUF_WIDTH, FRAMEBUF_HEIGHT), (0,0,0,0))

    drawFBF = ImageDraw.Draw(framebuf)
    ratios = np.array([DISPLAY_WIDTH/CAMERA_WIDTH, DISPLAY_HEIGHT/CAMERA_HEIGHT],dtype=float)
    ratiosLCD = ratios
    ratiosFBF = np.array([FRAMEBUF_WIDTH/CAMERA_WIDTH, FRAMEBUF_HEIGHT/CAMERA_HEIGHT],dtype=float)

    # draw* functions work in CAMERA coordinates
    def drawtext(xy, txt, fill = None, font = FONT_SMALL):
        nonlocal drawLCD, drawFBF, ratiosLCD, ratiosFBF
        if drawLCD is not None:
            xyl = list(np.int32(np.array(xy) * ratiosLCD))
            drawLCD.text(xyl, txt, fill=fill, font=font['lcd'])
        if drawFBF is not None: 
            xyf = list(np.int32(np.array(xy) * ratiosFBF))
            drawFBF.text(xyf, txt, fill=fill, font=font['fbf'])

    def drawline(pts, fill = None, width = 0):
        nonlocal drawLCD, drawFBF, ratiosLCD, ratiosFBF
        if drawLCD is not None:
            ptsl = list(np.int32(np.array(pts).reshape(-1,2) * ratiosLCD).reshape(-1))
            drawLCD.line(ptsl, fill = fill, width = width)
        if drawFBF is not None: 
            ptsf = list(np.int32(np.array(pts).reshape(-1,2) * ratiosFBF).reshape(-1))
            drawFBF.line(ptsf, fill = fill, width = width)

    def drawrect(pts, outline = None, fill = None):
        nonlocal drawLCD, drawFBF, ratiosLCD, ratiosFBF
        if drawLCD is not None:
            ptsl = list(np.int32(np.array(pts).reshape(-1,2) * ratiosLCD).reshape(-1))
            drawLCD.rectangle(ptsl, fill = fill, outline = outline)
        if drawFBF is not None: 
            ptsf = list(np.int32(np.array(pts).reshape(-1,2) * ratiosFBF).reshape(-1))
            drawFBF.rectangle(ptsf, fill = fill, outline = outline)

    def textsize(txt, font = FONT_SMALL):
        nonlocal ratiosFBF
        return font['fbf'].getsize(txt) / ratiosFBF

    drawtext((100, 100), 'Initialising...', fill = "BLUE", font = FONT_LARGE)
    if not no_lcd:
        LCD.display(screenbuf) # force initial drawing on LCD

    ##################################################
    if no_framebuf:
        drawFBF = None

    objd = SSD_MOBILENET(wanted_label='person', model_file=args.model, label_file=args.labels)

   # Definition of the parameters
    max_cosine_distance = args.max_cosine_distance
    nn_budget = None
    nms_max_overlap = args.nms_max_overlap
    
   # deep_sort 
    model_filename = '{}/model_data/mars-small128.pb'.format(basedir)
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance,
                                                       nn_budget)
    tracker = Tracker(metric,max_iou_distance=args.max_iou_distance,
                      max_age=args.max_age)
    delcount=0
    intcount=0
    poscount=0
    negcount=0
    db = {}
    def check_track(i):
        nonlocal delcount
        nonlocal ratios
        nonlocal cameracountline
        if i in db and len(db[i]) > 1:
            if any_intersection(cameracountline[0], cameracountline[1], np.array(db[i])):
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

    cap = cv2.VideoCapture(0)

    try:
        while True:
            t1 = time.time()
            t1read = time.time()
            ret, frame = cap.read()
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA))
            t2read = time.time()
            if not ret:
                break
        
            t1objd = time.time()
            boxs = objd.detect_image(image)
            # boxs is in tlwh format
            t2objd = time.time()

            # set background image and clear screen
            t1bclr = time.time()
            # clear both screens
            drawrect([0,0,CAMERA_WIDTH,CAMERA_HEIGHT], fill=0, outline=0)
            # also blit camera image onto LCD screen
            if not no_lcd:
                screenbuf.paste(image.resize((DISPLAY_WIDTH, DISPLAY_HEIGHT)))
            if not no_framebuf:
                framebuf.paste(image.resize((FRAMEBUF_WIDTH, FRAMEBUF_HEIGHT)))

            t2bclr = time.time()

            # features and tracking get more expensive as there are more things to track
            # writes to disk are fairly substantial as well

            # print("box_num",len(boxs))
            t1feat = t2objd
            #frame = np.array(image)
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
            a = (cameracountline[0,0], cameracountline[0,1])
            b = (cameracountline[1,0], cameracountline[1,1])
            drawline([a, b], fill=(0,0,255), width=3)

            for track in tracker.deleted_tracks:
                i = track.track_id
                if track.is_deleted():
                    check_track(track.track_id)

            for track in tracker.tracks:
                i = track.track_id
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                if i not in db:
                    db[i] = []
                db[i].append(track.mean[:2].copy())
                if len(db[i]) > 1:
                    pts = (np.array(db[i]).reshape((-1,1,2))).reshape(-1)
                    drawline(pts, fill=(255,0,255), width=3)
                    p1 = cameracountline[0]
                    q1 = cameracountline[1]
                    p2 = np.array(db[i][-1])
                    q2 = np.array(db[i][-2])
                    cp = np.cross(q1 - p1,q2 - p2)
                    if intersection(p1,q1,p2,q2):
                        intcount+=1
                        print("track_id={} just intersected camera countline; cross-prod={}; intcount={}".format(i,cp,intcount))
                        drawline(pts[-4:], fill=(0,0,255), width=5)
                        if cp >= 0:
                            poscount+=1
                        else:
                            negcount+=1

                bbox = track.to_tlbr()
                drawrect(bbox,outline=(255,255,255))
                drawtext(bbox[:2],str(track.track_id), fill=(0,255,0), font=FONT_SMALL)

            for det in detections:
                bbox = det.to_tlbr()
                drawrect(bbox,outline=(255,0,0))
            
            t2trac = time.time()

            t1draw = time.time()

            # draw counters along bottom of screen(s)
            (_, dy) = textsize(str(negcount), font = FONT_LARGE)
            drawtext((0, CAMERA_HEIGHT-dy), str(negcount), fill=(255,0,0), font=FONT_LARGE)
            (dx, dy) = textsize(str(abs(negcount-poscount)), font = FONT_LARGE)
            drawtext(((CAMERA_WIDTH-dx)/2, CAMERA_HEIGHT-dy), str(abs(negcount-poscount)), fill=(0,255,0), font=FONT_LARGE)
            (dx, dy) = textsize(str(poscount), font = FONT_LARGE)
            drawtext((CAMERA_WIDTH-dx, CAMERA_HEIGHT-dy), str(poscount), fill=(0,0,255), font=FONT_LARGE)
            if not no_lcd:
                LCD.display(screenbuf) #50ms (RPi3)
            if not no_framebuf:
                framearray=np.array(framebuf)
                fbframe16 = cv2.cvtColor(framearray, cv2.COLOR_RGBA2BGR565)
                with open(args.framebuffer, 'wb') as buf:
                   buf.write(fbframe16)
            t2draw = time.time()
            t2 = time.time()

            print("Frame processing time={:.0f}ms (objd={:.0f}ms prep={:.0f}ms feat={:.0f}ms trac={:.0f}ms draw={:.0f}ms)".format(1000*(t2 - t1), 1000*(t2objd - t1objd), 1000*(t2prep - t1prep), 1000*(t2feat - t1feat), 1000*(t2trac - t1trac), 1000*(t2draw - t1draw)))

            if cv2.waitKey(1) & 0xff == ord('q'):
                break
        

        for track in tracker.tracks:
            check_track(track.track_id)

        print(delcount)

    finally:
        if not no_lcd:
            LCD.LCD_Clear()
        if not no_framebuf:
            framearray=np.array(framebuf)
            framearray[:,:] = 0
            fbframe16 = cv2.cvtColor(framearray, cv2.COLOR_RGBA2BGR565)
            with open(args.framebuffer, 'wb') as buf:
               buf.write(fbframe16)

if __name__ == '__main__':
    main()