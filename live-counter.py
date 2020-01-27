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
import threading
from multiprocessing import Pool
import socket
import selectors

warnings.filterwarnings('ignore')

class Encoder(object):
    def __init__(self, encoder, frame):
        self.encoder = encoder
        self.frame = frame

    def __call__(self, box):
        return self.encoder(self.frame, [self.box])[0]

def init_worker(fn, model_filename):
    fn.encoder = gdet.create_box_encoder(model_filename,batch_size=1)
def do_work(args):
    frame, box = args
    return do_work.encoder(frame, [box])[0]

class SigTerm(SystemExit): pass
def sigterm(sig,frm): raise SigTerm
signal.signal(15,sigterm)

class MBox:
    def __init__(self):
        self.message = None
        self.lock = threading.Lock()

    def get_message(self):
        self.lock.acquire()
        message = self.message
        self.lock.release()
        return message

    def set_message(self, message):
        self.lock.acquire()
        self.message = message
        self.lock.release()

FONT_TINY_LCD  = ImageFont.truetype('fonts/truetype/freefont/FreeSansBold.ttf', 6)
FONT_TINY_FBF  = ImageFont.truetype('fonts/truetype/freefont/FreeSansBold.ttf', 24)
FONT_LARGE_LCD = ImageFont.truetype('fonts/truetype/freefont/FreeSansBold.ttf', 12)
FONT_SMALL_LCD = ImageFont.truetype('fonts/truetype/freefont/FreeSansBold.ttf', 10)
FONT_SMALL_FBF = ImageFont.truetype('fonts/truetype/freefont/FreeSansBold.ttf', 40)
FONT_LARGE_LCD = ImageFont.truetype('fonts/truetype/freefont/FreeSansBold.ttf', 12)
FONT_LARGE_FBF = ImageFont.truetype('fonts/truetype/freefont/FreeSansBold.ttf', 48)

FONT_TINY  = {'lcd': FONT_TINY_LCD, 'fbf': FONT_TINY_FBF}
FONT_SMALL = {'lcd': FONT_SMALL_LCD, 'fbf': FONT_SMALL_FBF}
FONT_LARGE = {'lcd': FONT_LARGE_LCD, 'fbf': FONT_LARGE_FBF}

DISPLAY_WIDTH = 160                # LCD panel width in pixels
DISPLAY_HEIGHT = 128               # LCD panel height

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

FRAMEBUF_WIDTH = 640
FRAMEBUF_HEIGHT = 480

#COLOR_MODE = cv2.COLOR_RGBA2BGR565
COLOR_MODE = None

no_framebuf = False

running = True
sleeping = False
delcount=0
intcount=0
poscount=0
negcount=0
perfmsgs=[]

def capthread_f(cap, box):
    global sleeping
    while True:
        if sleeping:
            time.sleep(0.2)
            continue
        ret, frame = cap.read()
        box.set_message(frame)

def clamp(minvalue, value, maxvalue):
    return max(minvalue, min(value, maxvalue))

def readcmd(conn, mask):
    global sleeping, running, delcount, intcount, poscount, negcount, perfmsgs
    data, client = conn.recvfrom(1000)
    if not data:
        return
    cmd = data.decode('utf-8').strip().split()
    if len(cmd) == 0:
        return

    cmd0 = cmd[0].lower()

    response = 'OK'
    if cmd0 == 'quit':
        running = False
    elif cmd0 == 'sleep':
        sleeping = True
    elif cmd0 == 'wake':
        sleeping = False
    elif cmd0 == 'reset':
        delcount=0
        intcount=0
        poscount=0
        negcount=0
    elif cmd0 == 'stat' or cmd0 == 'stats' or cmd0 == 'status':
        response  = "Deleted track count: {}\n".format(delcount)
        response += "Intersected track count: |{} - {}| = {}\n".format(poscount,negcount,intcount)
        response += 'OK'
    elif cmd0 == 'perf':
        response = '\n'.join(perfmsgs)
    else:
        response = 'Command not recognised.'

    response += '\n'

    conn.sendto(response.encode('utf-8'), client)


def main():
    global running, sleeping, perfmsgs
    global delcount, intcount, poscount, negcount
    global no_framebuf, no_lcd
    basedir = os.getenv('DEEPSORTHOME','.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--line', '-L', help="counting line: x1,y1,x2,y2",
                        default=None)
    parser.add_argument('--model', help='File path of object detection .tflite file.',
                        required=True)
    parser.add_argument('--encoder-model', help='File path of feature encoder .pb file.',
                        required=False)
    parser.add_argument('--labels', help='File path of labels file.', required=True)
    parser.add_argument('--no-framebuf', help='Disable framebuffer display',
                        required=False, action='store_true')
    parser.add_argument('--framebuffer', '-F', help='Framebuffer device',
                        default='/dev/fb0', metavar='DEVICE')
    parser.add_argument('--color-mode', help='Color mode for framebuffer, default: RGBA (see OpenCV docs)',
                        default=None, metavar='MODE')
    parser.add_argument('--max-cosine-distance', help='Max cosine distance', metavar='N',
                        default=0.6, type=float)
    parser.add_argument('--nms-max-overlap', help='Non-Max-Suppression max overlap', metavar='N',
                        default=1.0, type=float)
    parser.add_argument('--max-iou-distance', help='Max Intersection-Over-Union distance',
                        metavar='N', default=0.7, type=float)
    parser.add_argument('--max-age', help='Max age of lost track', metavar='N',
                        default=10, type=int)
    parser.add_argument('--num-threads', '-N', help='Number of threads for tensorflow lite',
                        metavar='N', default=4, type=int)
    parser.add_argument('--deepsorthome', help='Location of model_data directory',
                        metavar='PATH', default=basedir)
    parser.add_argument('--camera-flip', help='Flip the camera image vertically',
                        default=True, type=bool)
    parser.add_argument('--control-socket', help='Path to UNIX socket for control console.',
                        default='/tmp/live-counter.sock')
    args = parser.parse_args()
    basedir = args.deepsorthome

    socket_path = args.control_socket
    if os.path.exists(socket_path):
        os.remove(socket_path)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    server.bind(socket_path)
    server.setblocking(False)
    sel = selectors.DefaultSelector()
    sel.register(server, selectors.EVENT_READ, readcmd)

    no_framebuf = args.no_framebuf

    if args.color_mode == 'RGBA' or args.color_mode is None:
        COLOR_MODE = None
    elif args.color_mode == 'BGRA':
        COLOR_MODE = cv2.COLOR_RGBA2BGRA
    elif args.color_mode == 'BGR565':
        COLOR_MODE = cv2.COLOR_RGBA2BGR565
    else:
        COLOR_MODE = None
        print("Unsupported --color-mode={}".format(args.color_mode))

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

    def copybackbuf():
        global no_lcd, no_framebuf, COLOR_MODE
        nonlocal screenbuf, framebuf, args
        if not no_lcd:
            LCD.display(screenbuf) #50ms (RPi3)
        if not no_framebuf:
            framearray=np.array(framebuf)
            if COLOR_MODE is not None:
                fbframe16 = cv2.cvtColor(framearray, COLOR_MODE)
            else:
                fbframe16 = framearray
            with open(args.framebuffer, 'wb') as buf:
               buf.write(fbframe16)


    drawtext((100, 100), 'Initialising...', fill = "BLUE", font = FONT_LARGE)
    if not no_lcd:
        LCD.display(screenbuf) # force initial drawing on LCD

    ##################################################
    if no_framebuf:
        drawFBF = None

    objd = SSD_MOBILENET(wanted_label='person', model_file=args.model, label_file=args.labels, num_threads=int(args.num_threads))
    #objd = YOLO_TFLITE()

   # Definition of the parameters
    max_cosine_distance = args.max_cosine_distance
    nn_budget = None
    nms_max_overlap = args.nms_max_overlap
    
    # deep_sort 
    if args.encoder_model is None:
        model_filename = '{}/mars-64x32x3.pb'.format(basedir)
    else:
        model_filename = args.encoder_model

    #model_filename = '{}/model_data/mars-small128.pb'.format(basedir)
    encoder = gdet.create_box_encoder(model_filename,batch_size=32)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance,
                                                       nn_budget)
    tracker = Tracker(metric,max_iou_distance=args.max_iou_distance,
                      max_age=args.max_age)
    db = {}
    def check_track(i):
        global delcount
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
    frameTime = 0

    cap = cv2.VideoCapture(0)

    box = MBox()
    capthread = threading.Thread(target=capthread_f, args=(cap,box), daemon=True)
    capthread.start()

    #pool = Pool(processes=4,initializer=init_worker, initargs=(do_work, model_filename))

    sleeptext = False
    try:
        while running:
            events = sel.select(0)
            for key, mask in events:
                callback = key.data
                callback(key.fileobj, mask)
            if sleeping:
                if not sleeptext:
                    drawtext((10,int(DISPLAY_HEIGHT/2)),'sleeping...', fill=(255,255,255), font=FONT_LARGE)
                    copybackbuf()
                    sleeptext = True
                time.sleep(0.2)
                continue
            sleeptext = False

            t1 = time.time()
            t1read = time.time()
            #while True:
                # eat the stale, buffered frames
                #t1a = time.time()
                #ret, frame = cap.read()
                #t1b = time.time()
                #if not ret or (t1b - t1a) > 0.02:
                    #break
            frame = None
            while frame is None:
                frame = box.get_message()

            # the camera is mounted upside down currently
            if args.camera_flip:
                frame = cv2.flip(frame, 0)

            #ret, frame = cap.read()
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA))
            t2read = time.time()
        
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
            t1feat = time.time()
            #frame = np.array(image)

            # default
            features = encoder(frame,boxs)

            # attempted parallel version
            #frameBoxs = [(frame, box) for box in boxs]
            #features = np.array(pool.map(do_work, frameBoxs))

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

            if frameTime != 0:
                frameTimeMsg = "{:.0f}ms".format(frameTime*1000)
                (dx, dy) = textsize(frameTimeMsg, font = FONT_TINY)
                drawtext((CAMERA_WIDTH-dx, 0), frameTimeMsg, fill=(0,0,255), font=FONT_TINY)
            copybackbuf()

            t2draw = time.time()
            t2 = time.time()
            frameTime = t2 - t1

            msg = ("Frame processing time={:.0f}ms (read={:.0f}ms objd={:.0f}ms prep={:.0f}ms feat={:.0f}ms trac={:.0f}ms draw={:.0f}ms)".format(1000*(t2 - t1), 1000*(t2read - t1read), 1000*(t2objd - t1objd), 1000*(t2prep - t1prep), 1000*(t2feat - t1feat), 1000*(t2trac - t1trac), 1000*(t2draw - t1draw)))
            print(msg)
            perfmsgs.append(msg)
            if len(perfmsgs) > 10:
                perfmsgs = perfmsgs[-10:]

            if cv2.waitKey(1) & 0xff == ord('q'):
                break
        

        for track in tracker.tracks:
            check_track(track.track_id)

        print(delcount)

    finally:
        copybackbuf()
        if not no_lcd:
            LCD.LCD_Clear()

if __name__ == '__main__':
    main()
