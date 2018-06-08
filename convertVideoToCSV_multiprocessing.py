#!/usr/bin/python
"""
This script is dependent upon convertVideoToCSV_custom
"""
from __future__ import print_function
import os
from six.moves import xrange
import argparse
import _init_paths  # pylint: disable=unused-import

from convertVideoToCSV_custom import getPrediction, convertImages_fast, parse_args, convertImages_with_postprocessing
import multiprocessing


def convertCsvWorker(video_path, args):
    videoname = video_path.split('.')[0]
    args.csv_file = os.path.join(args.submission_path, videoname + '.csv')
    print('videoname:' + videoname)
    groundTruthImgList = []
    predictionImgList = []
    groundTruthList = open(os.path.join(args.test_video_list_dir, video_path), "r")
    groundTruthListlines = groundTruthList.readlines()
    for groundTruthListline in groundTruthListlines:
        gtfilename = groundTruthListline.split('\n')[0].split('\t')[0]
        groundTruthImgList.append(gtfilename)

    # We also read groudtruth md5 mapping
    mapping_file = open(os.path.join(args.mapping_dir, 'md5_mapping_' + video_path), "r")
    mappingLines = mapping_file.readlines()
    mapping_dict = {}
    for mapping in mappingLines:
        mapping_dict[mapping.split('\t')[1].replace('\n', '')] = mapping.split('\t')[0]

    for gt in groundTruthImgList:
        predictionImgList.append(getPrediction(gt, args, mapping_dict))

    if args.del_overlap:
        convertImages_with_postprocessing(predictionImgList, groundTruthImgList, args, mapping_dict)
    else:
        convertImages_fast(predictionImgList, groundTruthImgList, args, mapping_dict)


def main():
    args = parse_args()
    args.pred_list_dir = os.path.join(args.result_dir, 'List_Masks')
    if args.del_overlap:
        args.submission_path = os.path.join(args.result_dir, 'csv_files_del_overlap')
    else:
        args.submission_path = os.path.join(args.result_dir, 'csv_files')
    if not os.path.exists(args.submission_path):
        os.mkdir(args.submission_path)

    args.pred_image_list = os.listdir(args.pred_list_dir)
    test_video_list = os.listdir(args.test_video_list_dir)

    jobs = []
    for i in xrange(len(test_video_list)):
        p = multiprocessing.Process(target=convertCsvWorker, args=(test_video_list[i], args))
        jobs.append(p)
        p.start()

    # waiting for all jobs to finish
    for j in jobs:
        j.join()
    print("Finishing converting all the csv fils")


if __name__ == '__main__':
    main()