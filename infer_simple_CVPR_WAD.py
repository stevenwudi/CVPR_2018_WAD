from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import distutils.util
import os
import sys
from collections import defaultdict
from datetime import *
# Use a non-interactive backend
import matplotlib
import pandas as pd
from six.moves import xrange
from tqdm import tqdm

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

matplotlib.use('TkAgg')
import _init_paths  # pylint: disable=unused-import
import cv2
import torch
import nn as mynn
from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
# from core.test import im_detect_all
from test_rle import im_detect_all
from modeling.model_builder import Generalized_RCNN
import datasets.dummy_datasets as datasets
from datasets.dataloader_wad_cvpr2018 import WAD_CVPR2018
import utils.misc as misc_utils
import utils.net as net_utils
import utils.vis as vis_utils
from utils.detectron_weight_helper import load_detectron_weight
from utils.timer import Timer

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Demonstrate mask-rcnn results')
    parser.add_argument('--cfg', dest='cfg_file', required=True, help='optional config file')
    parser.add_argument('--load_ckpt', help='path of checkpoint to load')
    parser.add_argument('--dataset_dir', help='directory to load images for demo')
    parser.add_argument('--cls_boxes_confident_threshold', type=float, default=0.5, help='threshold for detection boundingbox')
    args = parser.parse_args()

    return args


def main():
    """main function"""

    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    args = parse_args()
    print('Called with args:')
    print(args)

    dataset = WAD_CVPR2018(args.dataset_dir)
    cfg.MODEL.NUM_CLASSES = len(dataset.eval_class) + 1  # with a background class

    print('load cfg from file: {}'.format(args.cfg_file))
    cfg_from_file(args.cfg_file)

    cfg.RESNETS.IMAGENET_PRETRAINED = False  # Don't need to load imagenet pretrained weights
    assert_and_infer_cfg()

    maskRCNN = Generalized_RCNN()
    maskRCNN.cuda()

    if args.load_ckpt:
        load_name = args.load_ckpt
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
        net_utils.load_ckpt(maskRCNN, checkpoint['model'])

    maskRCNN = mynn.DataParallel(maskRCNN, cpu_keywords=['im_info', 'roidb'], minibatch=True, device_ids=[0])  # only support single GPU

    maskRCNN.eval()
    imglist = misc_utils.get_imagelist_from_dir(dataset.test_image_dir)
    num_images = len(imglist)

    output_dir = os.path.join(('/').join(args.load_ckpt.split('/')[:-2]), 'Images')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in tqdm(xrange(num_images)):
        im = cv2.imread(imglist[i])
        assert im is not None
        timers = defaultdict(Timer)
        im_name, _ = os.path.splitext(os.path.basename(imglist[i]))
        args.current_im_name = im_name
        cls_boxes, cls_segms, prediction_row = im_detect_all(args, maskRCNN, im, dataset, timers=timers)

        thefile = open(os.path.join(output_dir, im_name+'.txt'), 'w')
        for item in prediction_row:
            thefile.write("%s\n" % item)


if __name__ == '__main__':
    main()
