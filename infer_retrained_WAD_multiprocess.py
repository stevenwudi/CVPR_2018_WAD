from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
from collections import defaultdict

# Use a non-interactive backend
import matplotlib
from six.moves import xrange
from tqdm import tqdm


import _init_paths  # pylint: disable=unused-import

matplotlib.use('TkAgg')
import cv2
import torch
import nn as mynn
from core.config import cfg, cfg_from_file, assert_and_infer_cfg
# from core.test import im_detect_all
from test_rle import im_detect_all
from modeling.model_builder import Generalized_RCNN
from datasets.dataloader_wad_cvpr2018 import WAD_CVPR2018
import utils.misc as misc_utils
import utils.net as net_utils
import utils.vis as vis_utils
from infer_retrained_WAD import parse_args

from utils.timer import Timer
import utils.env as envu
import utils.subprocess as subprocess_utils
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def main():
    args = parse_args()
    print('load cfg from file: {}'.format(args.cfg_file))
    cfg_from_file(args.cfg_file)
    if args.nms_soft:
        cfg.TEST.SOFT_NMS.ENABLED = True
    else:
        cfg.TEST.NMS = args.nms

    if args.nms_soft:
        output_dir = os.path.join(('/').join(args.load_ckpt.split('/')[:-2]),'Images_' + str(cfg.TEST.SCALE) + '_SOFT_NMS')
    elif args.nms:
        output_dir = os.path.join(('/').join(args.load_ckpt.split('/')[:-2]),'Images_' + str(cfg.TEST.SCALE) + '_NMS_%.2f' % args.nms)
    else:
        output_dir = os.path.join(('/').join(args.load_ckpt.split('/')[:-2]), 'Images_' + str(cfg.TEST.SCALE))

    if cfg.TEST.BBOX_AUG.ENABLED:
        output_dir += '_TEST_AUG'
    # if args.cls_boxes_confident_threshold < 0.5:
    output_dir += '_cls_boxes_confident_threshold_%.1f' % args.cls_boxes_confident_threshold

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    args.output_vis_dir = os.path.join(output_dir, 'Image_Vis')
    if not os.path.exists(args.output_vis_dir):
        os.makedirs(args.output_vis_dir)

    args.output_img_dir = os.path.join(output_dir, 'Image_Masks')
    if not os.path.exists(args.output_img_dir):
        os.makedirs(args.output_img_dir)

    args.output_list_dir = os.path.join(output_dir, 'List_Masks')
    if not os.path.exists(args.output_list_dir):
        os.makedirs(args.output_list_dir)
    print('Called with args:')
    print(args)

    if args.range is None:
        args.test_net_file, _ = os.path.splitext(__file__)
        dataset = WAD_CVPR2018(args.dataset_dir)

        img_produced = os.listdir(args.output_list_dir)
        imglist_all = misc_utils.get_imagelist_from_dir(dataset.test_image_dir)
        imglist = [x for x in imglist_all if x.split('/')[-1][:-4] + '.txt' not in img_produced]
        num_images = len(imglist)
        multi_gpu_test_net_on_dataset(args, num_images)
    else:
        test_net_on_dataset_multigpu(args)


def multi_gpu_test_net_on_dataset(args, num_images):
    """Multi-gpu inference on a dataset."""
    binary_dir = envu.get_runtime_dir()
    binary_ext = envu.get_py_bin_ext()
    binary = os.path.join(binary_dir, args.test_net_file + binary_ext)
    assert os.path.exists(binary), 'Binary \'{}\' not found'.format(binary)

    # Run inference in parallel in subprocesses
    # Outputs will be a list of outputs from each subprocess, where the output
    # of each subprocess is the dictionary saved by test_net().
    subprocess_utils.process_in_parallel_wad(args, num_images, binary)


def test_net_on_dataset_multigpu(args):

    dataset = WAD_CVPR2018(args.dataset_dir)
    cfg.MODEL.NUM_CLASSES = len(dataset.eval_class) + 1  # with a background class
    print('load cfg from file: {}'.format(args.cfg_file))
    cfg_from_file(args.cfg_file)
    cfg.RESNETS.IMAGENET_PRETRAINED = False  # Don't need to load imagenet pretrained weights
    if args.nms_soft:
        cfg.TEST.SOFT_NMS.ENABLED = True
    else:
        cfg.TEST.NMS = args.nms
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
    #imglist = misc_utils.get_imagelist_from_dir(dataset.test_image_dir)
    img_produced = os.listdir(args.output_list_dir)
    imglist_all = misc_utils.get_imagelist_from_dir(dataset.test_image_dir)
    imglist = [x for x in imglist_all if x.split('/')[-1][:-4] + '.txt' not in img_produced]

    #for i in tqdm(xrange(args.range[0], args.range[1])):
    start_idx = len(imglist_all) - args.range[0]
    end_idx = len(imglist_all) - args.range[0] + args.range[1]
    for i in tqdm(xrange(start_idx, end_idx)):
        im = cv2.imread(imglist_all[i])
        assert im is not None
        timers = defaultdict(Timer)
        im_name, _ = os.path.splitext(os.path.basename(imglist[i]))
        args.current_im_name = im_name
        cls_boxes, cls_segms, prediction_row = im_detect_all(args, maskRCNN, im, dataset, timers=timers)
        im_name, _ = os.path.splitext(os.path.basename(imglist[i]))
        print(im_name)
        if args.vis:
            vis_utils.vis_one_image_cvpr2018_wad(
                im[:, :, ::-1],  # BGR -> RGB for visualization
                im_name,
                args.output_vis_dir,
                cls_boxes,
                cls_segms,
                None,
                dataset=dataset,
                box_alpha=0.3,
                show_class=True,
                thresh=0.5,
                kp_thresh=2
            )

        thefile = open(os.path.join(args.output_list_dir, im_name + '.txt'), 'w')
        for item in prediction_row:
            thefile.write("%s\n" % item)


if __name__ == '__main__':
    main()
