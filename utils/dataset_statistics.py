#!/usr/bin/python
from __future__ import print_function
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse
from six.moves import xrange
from test_rle import rle_encoding


def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Customized mapping')
    parser.add_argument('--mapping_dir', default="/media/samsumg_1tb/CVPR2018_WAD/list_test_mapping",
                        help='md5 test image mapping dir')
    parser.add_argument('--pred_list_dir', default='/home/stevenwudi/PycharmProjects/CVPR_2018_WAD/results/mask_rcnn_original/pred_list')
    parser.add_argument('--test_video_list_dir', default='/media/samsumg_1tb/CVPR2018_WAD/list_test')
    parser.add_argument('--submission_path', default='/home/stevenwudi/PycharmProjects/CVPR_2018_WAD/results/mask_rcnn_original/csv_files/')
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
    csv.close()
    return


def main():
    args = parse_args()
    args.pred_image_list = os.listdir(args.pred_list_dir)
    if False:
        args.test_video_list_dir = '/media/samsumg_1tb/CVPR2018_WAD/train_video_list/'

    test_video_list = os.listdir(args.test_video_list_dir)
    number_of_img_in_videos = []
    for i in tqdm(xrange(len(test_video_list))):
        video_path = test_video_list[i]
        videoname = video_path.split('.')[0]
        if videoname == 'road02_cam_5_video_5_image_list_test':
            args.csv_file = args.submission_path + videoname+'.csv'
            print('videoname:' + videoname)
            groundTruthImgList = []
            groundTruthList = open(os.path.join(args.test_video_list_dir, video_path), "r")
            groundTruthListlines = groundTruthList.readlines()
            for groundTruthListline in groundTruthListlines:
                gtfilename = groundTruthListline.split('\n')[0].split('\t')[0]
                groundTruthImgList.append(gtfilename)

            number_of_img_in_videos.append(len(groundTruthImgList))

        # we plot the rgb image here
        # image_rgb = os.path.join(rgb_img_dir, imageID + '.jpg')
        # im = Image.open(image_rgb)
        # ax.imshow(im)
        # dst = '/home/stevenwudi/PycharmProjects/CVPR_2018_WAD/notebooks/images/rgb/' + str("%3d" % a) + '.jpg'
        # copyfile(image_rgb, dst)

    img_num = np.array(number_of_img_in_videos)
    print("Average images in a video is: %d with min: %d and max: %d" %
          (np.mean(img_num), np.min(img_num), np.max(img_num)))
    print(img_num)


if __name__ == '__main__':
    main()