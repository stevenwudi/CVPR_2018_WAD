"""Perform inference on one or more datasets."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import argparse
import cv2
import pprint
import sys
import torch

import _init_paths  # pylint: disable=unused-import
from core.config import cfg, merge_cfg_from_file, assert_and_infer_cfg
from core.test_engine import run_inference_wad
import utils.logging

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')

    parser.add_argument('--load_ckpt', default='./Outputs/e2e_mask_rcnn_R-101-FPN_2x/May30-12-10-19_n606_step/ckpt/model_step39999.pth', help='path of checkpoint to load')
    parser.add_argument('--multi-gpu-testing', default=False, help='using multiple gpus for inference', action='store_true')
    parser.add_argument('--vis', default=False, dest='vis', help='visualize detections', action='store_true')
    parser.add_argument('--output_dir', help='output directory to save the testing results. If not provided defaults to [args.load_ckpt|args.load_detectron]/../test.')
    parser.add_argument('--cfg', dest='cfg_file', default='./configs/e2e_mask_rcnn_R-101-FPN_2x.yaml', help='Config file for training (and optionally testing)')
    parser.add_argument('--range', default=(0, 3), help='(0, 38653) --> start (inclusive) and end (exclusive) indices', type=int, nargs=2)
    parser.add_argument('--dataset_dir', default='/media/samsumg_1tb/CVPR2018_WAD')
    parser.add_argument('--nms_soft', default=True, help='Using Soft NMS')
    parser.add_argument('--nms', default=0.3, help='default value for NMS')

    return parser.parse_args()


if __name__ == '__main__':

    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    logger = utils.logging.setup_logging(__name__)
    args = parse_args()
    logger.info('Called with args:')
    logger.info(args)

    assert (torch.cuda.device_count() == 1) ^ bool(args.multi_gpu_testing)
    if args.output_dir is None:
        ckpt_path = args.load_ckpt if args.load_ckpt else args.load_detectron
        args.output_dir = os.path.join(
            os.path.dirname(os.path.dirname(ckpt_path)), 'test')
        logger.info('Automatically set output directory to %s', args.output_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    cfg.VIS = args.vis

    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)

    cfg.TEST.DATASETS = ('wad',)
    cfg.MODEL.NUM_CLASSES = 8
    if args.nms_soft:
        cfg.TEST.SOFT_NMS.ENABLED = True
    else:
        cfg.TEST.NMS = args.nms
    assert_and_infer_cfg()

    logger.info('Testing with config:')
    logger.info(pprint.pformat(cfg))

    # For test_engine.multi_gpu_test_net_on_dataset
    args.test_net_file, _ = os.path.splitext(__file__)
    # manually set args.cuda
    args.cuda = True

    run_inference_wad(args, ind_range=args.range, multi_gpu_testing=args.multi_gpu_testing)

