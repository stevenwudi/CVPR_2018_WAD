"""Perform inference on one or more datasets."""

import argparse
import cv2
import os
import pprint
import sys
import time

import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import _init_paths  # pylint: disable=unused-import
from core.config import cfg, merge_cfg_from_file, merge_cfg_from_list, assert_and_infer_cfg
from core.test_engine import run_inference
import utils.logging

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument(
        '--dataset',
        help='training dataset')
    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='optional config file')

    parser.add_argument(
        '--load_ckpt', help='path of checkpoint to load')
    parser.add_argument(
        '--load_detectron', help='path to the detectron weight pickle file')

    parser.add_argument(
        '--output_dir',
        help='output directory to save the testing results. If not provided, '
             'defaults to [args.load_ckpt|args.load_detectron]/../test.')

    parser.add_argument(
        '--set', dest='set_cfgs',
        help='set config keys, will overwrite config in the cfg_file.'
             ' See lib/core/config.py for all options',
        default=[], nargs='*')

    parser.add_argument(
        '--range',
        help='start (inclusive) and end (exclusive) indices',
        type=int, nargs=2)
    parser.add_argument(
        '--multi-gpu-testing', help='using multiple gpus for inference', default=False,
        action='store_true')
    parser.add_argument(
        '--vis', dest='vis', help='visualize detections', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':

    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    logger = utils.logging.setup_logging(__name__)
    args = parse_args()
    logger.info('Called with args:')
    logger.info(args)

    assert (torch.cuda.device_count() == 1) ^ bool(args.multi_gpu_testing)

    assert bool(args.load_ckpt) ^ bool(args.load_detectron), \
        'Exactly one of --load_ckpt and --load_detectron should be specified.'
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
    if args.set_cfgs is not None:
        merge_cfg_from_list(args.set_cfgs)

    if args.dataset == "coco2017":
        cfg.TEST.DATASETS = ('coco_2017_val',)
        cfg.MODEL.NUM_CLASSES = 81
    elif args.dataset == "keypoints_coco2017":
        cfg.TEST.DATASETS = ('keypoints_coco_2017_val',)
        cfg.MODEL.NUM_CLASSES = 2
    else:  # For subprocess call
        assert cfg.TEST.DATASETS, 'cfg.TEST.DATASETS shouldn\'t be empty'
    assert_and_infer_cfg()

    logger.info('Testing with config:')
    logger.info(pprint.pformat(cfg))

    # For test_engine.multi_gpu_test_net_on_dataset
    args.test_net_file, _ = os.path.splitext(__file__)
    # manually set args.cuda
    args.cuda = True

    run_inference(
        args,
        ind_range=args.range,
        multi_gpu_testing=args.multi_gpu_testing,
        check_expected_results=True)



"""
INFO test_engine.py: 161: Total inference time: 1074.752s
INFO task_evaluation.py:  75: Evaluating detections
INFO json_dataset_evaluator.py: 162: Writing bbox results json to: /home/stevenwudi/PycharmProjects/mask-rcnn.pytorch/tools/Outputs/e2e_mask_rcnn_R-50-C4_1x/May10-17-08-44_n606_step/test/bbox_coco_2017_val_results.json
Loading and preparing results...
DONE (t=1.17s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=47.42s).
Accumulating evaluation results...
DONE (t=7.07s).
INFO json_dataset_evaluator.py: 222: ~~~~ Mean and per-category AP @ IoU=[0.50,0.95] ~~~~
INFO json_dataset_evaluator.py: 223: 35.8
INFO json_dataset_evaluator.py: 231: 51.4
INFO json_dataset_evaluator.py: 231: 27.1
INFO json_dataset_evaluator.py: 231: 38.2
INFO json_dataset_evaluator.py: 231: 38.6
INFO json_dataset_evaluator.py: 231: 58.5
INFO json_dataset_evaluator.py: 231: 62.0
INFO json_dataset_evaluator.py: 231: 56.8
INFO json_dataset_evaluator.py: 231: 30.7
INFO json_dataset_evaluator.py: 231: 23.3
INFO json_dataset_evaluator.py: 231: 22.2
INFO json_dataset_evaluator.py: 231: 60.1
INFO json_dataset_evaluator.py: 231: 61.2
INFO json_dataset_evaluator.py: 231: 42.2
INFO json_dataset_evaluator.py: 231: 20.6
INFO json_dataset_evaluator.py: 231: 29.3
INFO json_dataset_evaluator.py: 231: 60.7
INFO json_dataset_evaluator.py: 231: 54.2
INFO json_dataset_evaluator.py: 231: 53.5
INFO json_dataset_evaluator.py: 231: 43.3
INFO json_dataset_evaluator.py: 231: 50.0
INFO json_dataset_evaluator.py: 231: 57.8
INFO json_dataset_evaluator.py: 231: 61.8
INFO json_dataset_evaluator.py: 231: 65.3
INFO json_dataset_evaluator.py: 231: 64.7
INFO json_dataset_evaluator.py: 231: 13.4
INFO json_dataset_evaluator.py: 231: 33.1
INFO json_dataset_evaluator.py: 231: 9.6
INFO json_dataset_evaluator.py: 231: 25.4
INFO json_dataset_evaluator.py: 231: 26.9
INFO json_dataset_evaluator.py: 231: 57.1
INFO json_dataset_evaluator.py: 231: 18.6
INFO json_dataset_evaluator.py: 231: 32.1
INFO json_dataset_evaluator.py: 231: 38.5
INFO json_dataset_evaluator.py: 231: 34.2
INFO json_dataset_evaluator.py: 231: 19.9
INFO json_dataset_evaluator.py: 231: 30.2
INFO json_dataset_evaluator.py: 231: 44.9
INFO json_dataset_evaluator.py: 231: 32.2
INFO json_dataset_evaluator.py: 231: 40.9
INFO json_dataset_evaluator.py: 231: 33.0
INFO json_dataset_evaluator.py: 231: 29.2
INFO json_dataset_evaluator.py: 231: 36.3
INFO json_dataset_evaluator.py: 231: 26.7
INFO json_dataset_evaluator.py: 231: 10.8
INFO json_dataset_evaluator.py: 231: 10.4
INFO json_dataset_evaluator.py: 231: 37.5
INFO json_dataset_evaluator.py: 231: 21.5
INFO json_dataset_evaluator.py: 231: 17.7
INFO json_dataset_evaluator.py: 231: 27.0
INFO json_dataset_evaluator.py: 231: 27.3
INFO json_dataset_evaluator.py: 231: 20.3
INFO json_dataset_evaluator.py: 231: 18.9
INFO json_dataset_evaluator.py: 231: 30.4
INFO json_dataset_evaluator.py: 231: 47.7
INFO json_dataset_evaluator.py: 231: 36.8
INFO json_dataset_evaluator.py: 231: 28.7
INFO json_dataset_evaluator.py: 231: 22.6
INFO json_dataset_evaluator.py: 231: 36.4
INFO json_dataset_evaluator.py: 231: 21.8
INFO json_dataset_evaluator.py: 231: 35.2
INFO json_dataset_evaluator.py: 231: 23.7
INFO json_dataset_evaluator.py: 231: 55.3
INFO json_dataset_evaluator.py: 231: 51.1
INFO json_dataset_evaluator.py: 231: 55.0
INFO json_dataset_evaluator.py: 231: 52.5
INFO json_dataset_evaluator.py: 231: 22.1
INFO json_dataset_evaluator.py: 231: 46.2
INFO json_dataset_evaluator.py: 231: 27.3
INFO json_dataset_evaluator.py: 231: 49.5
INFO json_dataset_evaluator.py: 231: 29.4
INFO json_dataset_evaluator.py: 231: 36.2
INFO json_dataset_evaluator.py: 231: 31.0
INFO json_dataset_evaluator.py: 231: 48.6
INFO json_dataset_evaluator.py: 231: 10.9
INFO json_dataset_evaluator.py: 231: 46.0
INFO json_dataset_evaluator.py: 231: 32.0
INFO json_dataset_evaluator.py: 231: 23.5
INFO json_dataset_evaluator.py: 231: 42.0
INFO json_dataset_evaluator.py: 231: 2.1
INFO json_dataset_evaluator.py: 231: 12.6
INFO json_dataset_evaluator.py: 232: ~~~~ Summary metrics ~~~~
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.358
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.563
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.389
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.189
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.400
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.497
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.305
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.470
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.487
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.285
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.539
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.649
INFO json_dataset_evaluator.py: 199: Wrote json eval results to: /home/stevenwudi/PycharmProjects/mask-rcnn.pytorch/tools/Outputs/e2e_mask_rcnn_R-50-C4_1x/May10-17-08-44_n606_step/test/detection_results.pkl
INFO task_evaluation.py:  61: Evaluating bounding boxes is done!
INFO task_evaluation.py: 104: Evaluating segmentations
INFO json_dataset_evaluator.py:  83: Writing segmentation results json to: /home/stevenwudi/PycharmProjects/mask-rcnn.pytorch/tools/Outputs/e2e_mask_rcnn_R-50-C4_1x/May10-17-08-44_n606_step/test/segmentations_coco_2017_val_results.json
Loading and preparing results...
DONE (t=2.63s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *segm*
DONE (t=51.29s).
Accumulating evaluation results...
DONE (t=6.91s).
INFO json_dataset_evaluator.py: 222: ~~~~ Mean and per-category AP @ IoU=[0.50,0.95] ~~~~
INFO json_dataset_evaluator.py: 223: 31.3
INFO json_dataset_evaluator.py: 231: 39.9
INFO json_dataset_evaluator.py: 231: 14.4
INFO json_dataset_evaluator.py: 231: 33.1
INFO json_dataset_evaluator.py: 231: 29.4
INFO json_dataset_evaluator.py: 231: 39.0
INFO json_dataset_evaluator.py: 231: 59.1
INFO json_dataset_evaluator.py: 231: 57.5
INFO json_dataset_evaluator.py: 231: 29.7
INFO json_dataset_evaluator.py: 231: 18.7
INFO json_dataset_evaluator.py: 231: 19.5
INFO json_dataset_evaluator.py: 231: 54.7
INFO json_dataset_evaluator.py: 231: 59.1
INFO json_dataset_evaluator.py: 231: 42.7
INFO json_dataset_evaluator.py: 231: 13.3
INFO json_dataset_evaluator.py: 231: 21.7
INFO json_dataset_evaluator.py: 231: 64.3
INFO json_dataset_evaluator.py: 231: 51.7
INFO json_dataset_evaluator.py: 231: 34.5
INFO json_dataset_evaluator.py: 231: 36.2
INFO json_dataset_evaluator.py: 231: 38.8
INFO json_dataset_evaluator.py: 231: 52.4
INFO json_dataset_evaluator.py: 231: 62.5
INFO json_dataset_evaluator.py: 231: 52.1
INFO json_dataset_evaluator.py: 231: 44.8
INFO json_dataset_evaluator.py: 231: 13.2
INFO json_dataset_evaluator.py: 231: 39.6
INFO json_dataset_evaluator.py: 231: 8.1
INFO json_dataset_evaluator.py: 231: 21.7
INFO json_dataset_evaluator.py: 231: 29.0
INFO json_dataset_evaluator.py: 231: 52.5
INFO json_dataset_evaluator.py: 231: 1.0
INFO json_dataset_evaluator.py: 231: 18.3
INFO json_dataset_evaluator.py: 231: 34.7
INFO json_dataset_evaluator.py: 231: 19.5
INFO json_dataset_evaluator.py: 231: 17.4
INFO json_dataset_evaluator.py: 231: 29.9
INFO json_dataset_evaluator.py: 231: 24.2
INFO json_dataset_evaluator.py: 231: 25.2
INFO json_dataset_evaluator.py: 231: 46.5
INFO json_dataset_evaluator.py: 231: 29.4
INFO json_dataset_evaluator.py: 231: 24.0
INFO json_dataset_evaluator.py: 231: 35.3
INFO json_dataset_evaluator.py: 231: 10.2
INFO json_dataset_evaluator.py: 231: 7.1
INFO json_dataset_evaluator.py: 231: 5.7
INFO json_dataset_evaluator.py: 231: 35.4
INFO json_dataset_evaluator.py: 231: 17.1
INFO json_dataset_evaluator.py: 231: 15.2
INFO json_dataset_evaluator.py: 231: 28.8
INFO json_dataset_evaluator.py: 231: 25.8
INFO json_dataset_evaluator.py: 231: 18.8
INFO json_dataset_evaluator.py: 231: 15.7
INFO json_dataset_evaluator.py: 231: 24.6
INFO json_dataset_evaluator.py: 231: 45.8
INFO json_dataset_evaluator.py: 231: 36.9
INFO json_dataset_evaluator.py: 231: 30.0
INFO json_dataset_evaluator.py: 231: 13.7
INFO json_dataset_evaluator.py: 231: 30.3
INFO json_dataset_evaluator.py: 231: 18.1
INFO json_dataset_evaluator.py: 231: 26.1
INFO json_dataset_evaluator.py: 231: 13.6
INFO json_dataset_evaluator.py: 231: 55.3
INFO json_dataset_evaluator.py: 231: 52.4
INFO json_dataset_evaluator.py: 231: 55.2
INFO json_dataset_evaluator.py: 231: 51.2
INFO json_dataset_evaluator.py: 231: 17.8
INFO json_dataset_evaluator.py: 231: 44.5
INFO json_dataset_evaluator.py: 231: 24.3
INFO json_dataset_evaluator.py: 231: 50.9
INFO json_dataset_evaluator.py: 231: 26.6
INFO json_dataset_evaluator.py: 231: 37.2
INFO json_dataset_evaluator.py: 231: 28.1
INFO json_dataset_evaluator.py: 231: 50.4
INFO json_dataset_evaluator.py: 231: 5.2
INFO json_dataset_evaluator.py: 231: 45.3
INFO json_dataset_evaluator.py: 231: 29.3
INFO json_dataset_evaluator.py: 231: 18.8
INFO json_dataset_evaluator.py: 231: 42.0
INFO json_dataset_evaluator.py: 231: 0.1
INFO json_dataset_evaluator.py: 231: 8.9
INFO json_dataset_evaluator.py: 232: ~~~~ Summary metrics ~~~~
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.313
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.527
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.329
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.127
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.346
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.498
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.276
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.415
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.429
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.224
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.478
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.610
INFO json_dataset_evaluator.py: 122: Wrote json eval results to: /home/stevenwudi/PycharmProjects/mask-rcnn.pytorch/tools/Outputs/e2e_mask_rcnn_R-50-C4_1x/May10-17-08-44_n606_step/test/segmentation_results.pkl
INFO task_evaluation.py:  65: Evaluating segmentations is done!
INFO task_evaluation.py: 180: copypaste: Dataset: coco_2017_val
INFO task_evaluation.py: 182: copypaste: Task: box
INFO task_evaluation.py: 185: copypaste: AP,AP50,AP75,APs,APm,APl
INFO task_evaluation.py: 186: copypaste: 0.3580,0.5631,0.3894,0.1893,0.3999,0.4974
INFO task_evaluation.py: 182: copypaste: Task: mask
INFO task_evaluation.py: 185: copypaste: AP,AP50,AP75,APs,APm,APl
INFO task_evaluation.py: 186: copypaste: 0.3127,0.5273,0.3294,0.1265,0.3455,0.4979
"""