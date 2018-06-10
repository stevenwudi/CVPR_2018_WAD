#!/usr/bin/python
"""
This script is dependent upon convertVideoToCSV_custom
"""
from __future__ import print_function
import os
from six.moves import xrange
import _init_paths  # pylint: disable=unused-import

from convertVideoToCSV_custom import getPrediction, parse_args, convertImages_with_postprocessing_image, convertImages_with_image
import multiprocessing


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
        video_path = test_video_list[i]
        videoname = video_path.split('.')[0]
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

        for list_index, filename in enumerate(groundTruthImgList):
            args.csv_file_image = os.path.join(args.submission_path,  videoname + '_FRAME_%d' % list_index + '_' + filename.split('/')[-1][:-4] + '.csv')
            #if not os.path.exists(args.csv_file_image):
            if True:
                if args.del_overlap:
                    p = multiprocessing.Process(target=convertImages_with_postprocessing_image,
                                                args=(filename, list_index, predictionImgList, args, mapping_dict))
                else:
                    p = multiprocessing.Process(target=convertImages_with_image,
                                                args=(filename, list_index, predictionImgList, args, mapping_dict))
                jobs.append(p)
                p.start()

                if len(jobs) > args.num_threads:
                    jobs[-args.num_threads].join()

    # waiting for all jobs to finish
    for j in jobs:
        j.join()
    print("Finishing converting all the csv fils")


if __name__ == '__main__':
    main()