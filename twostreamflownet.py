# Adapted from run-flownet-many.py at https://github.com/lmb-freiburg/flownet2
#         and  denseFlow.cpp       at https://github.com/wanglimin/dense_flow

import os
import numpy as np
import argparse
import caffe
import tempfile
import math
import progressbar
import cv2
import glob

parser = argparse.ArgumentParser()
parser.add_argument('caffemodel', help='path to model')
parser.add_argument('deployproto', help='path to deploy prototxt template')
parser.add_argument('videofolder', help='path to video folder')
parser.add_argument('outputfolder', help='output folder')
parser.add_argument('--gpu',  help='gpu id to use (0, 1, ...)', default=0, type=int)
parser.add_argument('--verbose',  help='whether to output all caffe logging', action='store_true')
parser.add_argument('--width', help='width of all videos', default=-1, type=int)
parser.add_argument('--height', help='height of all videos', default=-1, type=int)
parser.add_argument('--extension', help='video extension', default='.avi', type=str)
parser.add_argument('--bound', help='absolute bound of optical flow', default=15, type=int)

args = parser.parse_args()

if(not os.path.isfile(args.caffemodel)):
    raise BaseException('Caffe model does not exist or is not a file: ' + args.caffemodel)
if(not os.path.isfile(args.deployproto)):
    raise BaseException('Deploy definition file does not exist or is not a file: ' + args.deployproto)
if(not os.path.isdir(args.videofolder)):
    raise BaseException('Video folder does not exist or is not a directory: ' + args.videofolder)
if(not os.path.isdir(args.outputfolder)): 
    os.mkdir(args.outputfolder)

# Recursivity requires Python >= 3.5
videofiles = glob.glob(os.path.join(args.videofolder, '**', '*' + args.extension), recursive=True)
width = args.width
height = args.height
if width == -1 or height == -1:
    cap = cv2.VideoCapture(videofiles[0])
    _, frame = cap.read()
    (height, width) = (frame.shape[0], frame.shape[1])

vars = {}
vars['TARGET_WIDTH'] = width
vars['TARGET_HEIGHT'] = height
divisor = 64.
vars['ADAPTED_WIDTH'] = int(math.ceil(width/divisor) * divisor)
vars['ADAPTED_HEIGHT'] = int(math.ceil(height/divisor) * divisor)
vars['SCALE_WIDTH'] = width / float(vars['ADAPTED_WIDTH']);
vars['SCALE_HEIGHT'] = height / float(vars['ADAPTED_HEIGHT']);
tmp = tempfile.NamedTemporaryFile(mode='w', delete=False)
proto = open(args.deployproto).readlines()
for line in proto:
    for key, value in vars.items():
        tag = "$%s$" % key
        line = line.replace(tag, str(value))
    tmp.write(line)
tmp.flush()

if not args.verbose:
    caffe.set_logging_disabled()
caffe.set_device(args.gpu)
caffe.set_mode_gpu()
net = caffe.Net(tmp.name, args.caffemodel, caffe.TEST)

bar = progressbar.ProgressBar()
for f in bar(files):
    cap = cv2.VideoCapture(f)
    _, img1 = cap.read()
    if img1.shape[0] != height or img1.shape[1] != width:
        img1 = cv2.resize(img1, (width, height))

    framenum = 1
    imgs = [None, img1]
    while True:
        imgs[0] = np.copy(imgs[1])
        retval, imgs[1] = cap.read()
        if not retval: break
        if imgs[1].shape[0] != height or imgs[1].shape[1] != width:
            imgs[1] = cv2.resize(imgs[1], (width, height))

        input_dict = {}
        for i in range(2):
            if len(imgs[i].shape) < 3:
                input_dict[net.inputs[i]] = imgs[i][np.newaxis, np.newaxis, :, :]
            else:
                input_dict[net.inputs[i]] = imgs[i][np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :]

        net.forward(**input_dict)
        blob = np.squeeze(net.blobs['predict_flow_final'].data).transpose(1, 2, 0)
        flow = blob.astype(np.float32)
        flowimg = np.zeros(flow.shape, np.uint8)

        # Inspired from the optical flow tool used for two-stream Caffe
        # Loops are very inefficient in NumPy, so we use masks
        for i in range(2):
            x, y = (abs(flow[:, :, i]) <= args.bound).nonzero()
            flowimg[x, y, i] = np.rint(255 * (flow[x, y, i] + args.bound) / (2 * args.bound))
            x, y = (flow[:, :, i] > args.bound).nonzero()
            flowimg[x, y, i] = 255
            x, y = (flow[:, :, i] < -args.bound).nonzero()
            flowimg[x, y, i] = 0
            # Output filename format is "(video filename)_(x or y)_(frame number).jpg", change this according to your needs
            filename = '_'.join(os.path.basename(f), chr(ord('x' + i)), str(framenum) + '.jpg')
            cv2.imwrite(os.path.join(args.outputfolder, filename), flowimg[:, :, i])
    framenum += 1
