#!/usr/bin/python
from __future__ import print_function
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse
from six.moves import xrange
from test_rle import rle_encoding
from datasets.dataloader_wad_cvpr2018 import WAD_CVPR2018


def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Customized mapping')
    parser.add_argument('--result_dir', default='./Outputs/e2e_mask_rcnn_R-101-FPN_2x/May30-12-10-19_n606_step/Images_0')
    parser.add_argument('--mapping_dir', default="/media/samsumg_1tb/CVPR2018_WAD/list_test_mapping", help='md5 test image mapping dir')
    parser.add_argument('--test_video_list_dir', default='/media/samsumg_1tb/CVPR2018_WAD/list_test')
    parser.add_argument('--test_img_dir', default='/media/samsumg_1tb/CVPR2018_WAD/test')
    parser.add_argument('--dataset_dir', default='/media/samsumg_1tb/CVPR2018_WAD')

    args = parser.parse_args()
    return args


def getPrediction(groundTruthFile, args, mapping_dict):
    # determine the prediction path, if the method is first called
    md5_file_name = mapping_dict[groundTruthFile] + '.txt'

    if md5_file_name in args.pred_image_list:
        predictionFile = os.path.join(args.pred_list_dir, md5_file_name)
        return predictionFile
    else:
        print("Found no prediction for ground truth {}".format(groundTruthFile))
        return None


def convertImages(predictionList, groundTruthList, args, mapping_dict):
    csv = open(args.csv_file, 'a')
    for list_index, filename in enumerate(groundTruthList):
        imageID = mapping_dict[filename]
        predicitionFile = open(predictionList[list_index], "r")
        predictionlines = predicitionFile.readlines()
        for predictionline in predictionlines:
            predictionInfo = predictionline.split(' ')
            img = Image.open(predictionInfo[0])
            InstanceMap = np.array(img)
            csv.write("{},".format(imageID))
            csv.write("{},".format(predictionInfo[1]))
            csv.write("{},".format(predictionInfo[2].split('\n')[0]))
            idmap1d = np.reshape(InstanceMap == 1, (-1))
            Totalcount = np.sum(idmap1d)
            csv.write("{},".format(Totalcount))
            EncodedPixed = rle_encoding(InstanceMap)
            csv.write(EncodedPixed)
            csv.write("\n")
    csv.close()
    return


def convertImages_fast(predictionList, groundTruthList, args, mapping_dict):
    csv = open(args.csv_file, 'a')
    dataset = WAD_CVPR2018('/media/samsumg_1tb/CVPR2018_WAD')
    for list_index, filename in enumerate(groundTruthList):
        imageID = mapping_dict[filename]
        if predictionList[list_index]:
            predicitionFile = open(predictionList[list_index], "r")
            predictionlines = predicitionFile.readlines()
            for predictionline in predictionlines:
                predictionInfo = predictionline.split(' ')
                img = Image.open(predictionInfo[0])
                InstanceMap = np.array(img)
                csv.write("{},".format(imageID))
                # Historical code
                if int(predictionInfo[1]) in dataset.contiguous_category_id_to_json_id.keys():
                    csv.write("{},".format(dataset.contiguous_category_id_to_json_id[int(predictionInfo[1])]))
                else:
                    csv.write("{},".format(predictionInfo[1]))
                csv.write("{},".format(predictionInfo[2].split('\n')[0]))
                idmap1d = np.reshape(InstanceMap == 1, (-1))
                Totalcount = np.sum(idmap1d)
                csv.write("{},".format(Totalcount))
                EncodedPixed = rle_encoding(InstanceMap)
                csv.write(EncodedPixed)
                csv.write('\n')
    csv.close()
    print('Finish converting file: %s' % args.csv_file)
    return


def main():
    args = parse_args()
    args.pred_list_dir = os.path.join(args.result_dir, 'List_Masks')
    args.submission_path = os.path.join(args.result_dir, 'csv_files')
    args.pred_image_list = os.listdir(args.pred_list_dir)
    test_video_list = os.listdir(args.test_video_list_dir)
    number_of_img_in_videos = []
    for i in tqdm(xrange(len(test_video_list))):
        video_path = test_video_list[i]
        videoname = video_path.split('.')[0]

        if not os.path.exists(args.submission_path):
            os.mkdir(args.submission_path)
        args.csv_file = os.path.join(args.submission_path, videoname + '.csv')
        print('videoname:' + videoname)
        groundTruthImgList = []
        predictionImgList = []
        groundTruthList = open(os.path.join(args.test_video_list_dir, video_path), "r")
        groundTruthListlines = groundTruthList.readlines()
        for groundTruthListline in groundTruthListlines:
            gtfilename = groundTruthListline.split('\n')[0].split('\t')[0]
            groundTruthImgList.append(gtfilename)

        number_of_img_in_videos.append(len(groundTruthImgList))
        # We also read groudtruth md5 mapping
        mapping_file = open(os.path.join(args.mapping_dir, 'md5_mapping_'+video_path), "r")
        mappingLines = mapping_file.readlines()
        mapping_dict = {}
        for mapping in mappingLines:
            mapping_dict[mapping.split('\t')[1].replace('\n', '')] = mapping.split('\t')[0]

        for gt in groundTruthImgList:
            predictionFile = getPrediction(gt, args, mapping_dict)
            if len(predictionFile):
                predictionImgList.append(predictionFile)

        # This will convert the png images to RLE format
        convertImages_fast(predictionImgList, groundTruthImgList, args, mapping_dict)


if __name__ == '__main__':
    main()