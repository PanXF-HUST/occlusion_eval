import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms

import torch.nn as nn
import torch.utils.data
import numpy as np
from opt import opt

from dataloader import ImageLoader, DetectionLoader, DetectionProcessor, DataWriter, Mscoco
from yolo.util import write_results, dynamic_write_results
from SPPE.src.main_fast_inference import *

import os
import sys
from tqdm import tqdm
import time
from fn import getTime

from pPose_nms import pose_nms, write_json

args = opt
args.dataset = 'coco'
if not args.sp:
    torch.multiprocessing.set_start_method('forkserver', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')
    # torch.queue.set_start_method('forkserver', force=True)
    # torch.queue.set_sharing_strategy('file_system')

if __name__ == "__main__":
    inputpath = args.inputpath
    inputlist = args.inputlist
    mode = args.mode
    if not os.path.exists(args.outputpath):
        os.mkdir(args.outputpath)

    if len(inputlist):
        im_names = open(inputlist, 'r').readlines()
    elif len(inputpath) and inputpath != '/':
        for root, dirs, files in os.walk(inputpath):
            im_names = files
    else:
        raise IOError('Error: must contain either --indir/--list')

    # Load input images
    data_loader = ImageLoader(im_names, batchSize=args.detbatch, format='yolo').start()

    # Load detection loader
    print('Loading YOLO model..')
    sys.stdout.flush()
    det_loader = DetectionLoader(data_loader, batchSize=args.detbatch).start()



    print('here will show the det_loader information')
    data_loader_length = data_loader.length()
    for i in range(data_loader_length):
        (orig_img, im_name, boxes, scores, inps, pt1, pt2) = det_loader.read()
        print('image_name',im_name)
        print('boxes',boxes)
        print('scores',scores)
        print('inps',inps)
        print('pt1',pt1)
        print('pt2',pt2)
        print('------------------------------------------------------------')
    print('data_loader finish+++++++++++++++++++++++++++++++++++')


    det_processor = DetectionProcessor(det_loader).start()
    print('successful processed')

    print('here is the result which is processed')
    data_len = data_loader.length()
    print('start shown ')
    # im_names_desc =range(data_len+1)
    im_names_desc = tqdm(range(data_len))
    for i in im_names_desc:
        print('+++++++++++++++++++++++','this is the ',i,'th img+++++++++++++++++')
        with torch.no_grad():
            (inps1, orig_img1, im_name1, boxes1, scores1, pt11, pt21) = det_processor.read()
            if boxes is None or boxes.nelement() == 0:
                print('there is none')
                continue

            print('image_name', im_name1)
            print('inps', inps1)
            print('boxes', boxes1)
            print('scores', scores1)
            print('pt1', pt11)
            print('pt2', pt21)

    print ('finish test+++++++++++++++++++++++++++++++++')
'''
def save_detloader_json(all_results, outputpath):

    json_results = []
    for im_res in all_results:
        im_name = im_res['imgname']

    with open(os.path.join(outputpath, 'detloader-results.json'), 'w') as json_file:
        json_file.write(json.dumps(json_results))
'''