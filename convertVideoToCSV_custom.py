#!/usr/bin/python
from __future__ import print_function
import os
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse
from six.moves import xrange
from test_rle import rle_encoding
from datasets.dataloader_wad_cvpr2018 import WAD_CVPR2018
from pycocotools import mask as maskUtils


def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Customized mapping')
    parser.add_argument('--result_dir', default='./Outputs/e2e_mask_rcnn_R-101-FPN_2x/May30-12-10-19_n606_step/Images_0')
    #parser.add_argument('--result_dir', default='./Outputs/e2e_mask_rcnn_R-101-FPN_2x/May30-12-10-19_n606_step/Images_0_NMS_0.50_cls_boxes_confident_threshold_0.1')
    parser.add_argument('--mapping_dir', default="/media/samsumg_1tb/CVPR2018_WAD/list_test_mapping", help='md5 test image mapping dir')
    parser.add_argument('--test_video_list_dir', default='/media/samsumg_1tb/CVPR2018_WAD/list_test')
    parser.add_argument('--test_img_dir', default='/media/samsumg_1tb/CVPR2018_WAD/test')
    parser.add_argument('--dataset_dir', default='/media/samsumg_1tb/CVPR2018_WAD')
    parser.add_argument('--del_overlap', default=0.1, help='None or a float number')
    parser.add_argument('--num_threads', default=15, help='multiprocessing thread')

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
    df = pd.DataFrame(columns=['ImageId', 'LabelId', 'Confidence', 'PixelCount', 'EncodedPixels'])
    df_count = 0
    for list_index, filename in enumerate(groundTruthList):
        imageID = mapping_dict[filename]
        if predictionList[list_index]:
            predicitionFile = open(predictionList[list_index], "r")
            predictionlines = predicitionFile.readlines()
            for predictionline in predictionlines:
                predictionInfo = predictionline.split(' ')
                img = Image.open(predictionInfo[0])
                InstanceMap = np.array(img)
                InstanceMap = InstanceMap / np.max(InstanceMap)
                # Historical code
                LabelId = int(predictionInfo[1])
                Confidence = float(predictionInfo[2].split('\n')[0])
                idmap1d = np.reshape(InstanceMap > 0, (-1))
                PixelCount = np.sum(idmap1d)
                EncodedPixels = rle_encoding(InstanceMap)

                df.loc[df_count] = [imageID, LabelId, Confidence, PixelCount, EncodedPixels]
                df_count += 1

    df.to_csv(args.csv_file, header=True, index=False)
    print('Finish converting file: %s' % args.csv_file)
    return


def convertImages_with_postprocessing(predictionList, groundTruthList, args, mapping_dict):
    df = pd.DataFrame(columns=['ImageId', 'LabelId', 'Confidence', 'PixelCount', 'EncodedPixels'])
    df_count = 0
    del_count = 0

    for list_index, filename in enumerate(groundTruthList):
        imageID = mapping_dict[filename]
        if predictionList[list_index]:
            predicitionFile = open(predictionList[list_index], "r")
            predictionlines = predicitionFile.readlines()
            # We keep a mask for the whole image, and fill it with masks. The maskes are filled
            # with descending order and if there is an overlap, we discard such instance.
            img_mask_list = []
            img_mask_valid = []
            conf_list = []
            label_list = []
            for predictionline in predictionlines:
                predictionInfo = predictionline.split(' ')
                img_mask_list.append(predictionInfo[0])
                label_list.append(int(predictionInfo[1]))
                conf_list.append(float(predictionInfo[2].split('\n')[0]))

            conf_order = np.argsort(-np.array(conf_list))
            image_mask_all = np.zeros(shape=(2710, 3384))
            for conf_idx in conf_order:
                # skip flag indicates whether we will skip this instance
                skip_flag = False
                img = Image.open(img_mask_list[conf_idx])
                InstanceMap = np.array(img)
                image_mask_all += InstanceMap > 0
                if len(np.unique(image_mask_all)) > 2:
                    # we will check whether we will keep this instance by looping over all previous instances:
                    for im_mask in img_mask_valid:
                        rle1 = maskUtils.encode(np.array(im_mask[:, :, np.newaxis], order='F'))[0]
                        rle2 = maskUtils.encode(np.array(InstanceMap[:, :, np.newaxis], order='F'))[0]
                        iou = maskUtils.iou([rle1], [rle2], [0])
                        if iou[0][0] > args.del_overlap:
                            image_mask_all -= InstanceMap
                            del_count += 1
                            skip_flag = True
                            continue
                if not skip_flag:
                    img_mask_valid.append(InstanceMap)
                    LabelId = label_list[conf_idx]
                    Confidence = conf_list[conf_idx]
                    idmap1d = np.reshape(InstanceMap > 0, (-1))
                    PixelCount = np.sum(idmap1d)
                    EncodedPixels = rle_encoding(InstanceMap)

                    df.loc[df_count] = [imageID, LabelId, Confidence, PixelCount, EncodedPixels]
                    df_count += 1

    df.to_csv(args.csv_file, header=True, index=False)
    print('Finish converting file: %s with %d deleting overlaps' % (args.csv_file, del_count))
    return


def convertImages_with_postprocessing_image(filename, list_index,  predictionList, args, mapping_dict):

    df = pd.DataFrame(columns=['ImageId', 'LabelId', 'Confidence', 'PixelCount', 'EncodedPixels'])
    df_count = 0
    del_count = 0
    imageID = mapping_dict[filename]
    if os.path.exists(predictionList[list_index]):
        predicitionFile = open(predictionList[list_index], "r")
        predictionlines = predicitionFile.readlines()
        # We keep a mask for the whole image, and fill it with masks. The maskes are filled
        # with descending order and if there is an overlap, we discard such instance.
        img_mask_list = []
        img_mask_valid = []
        conf_list = []
        label_list = []
        for predictionline in predictionlines:
            predictionInfo = predictionline.split(' ')
            img_mask_list.append(predictionInfo[0])
            label_list.append(int(predictionInfo[1]))
            conf_list.append(float(predictionInfo[2].split('\n')[0]))

        conf_order = np.argsort(-np.array(conf_list))
        image_mask_all = np.zeros(shape=(2710, 3384))
        for conf_idx in conf_order:
            # skip flag indicates whether we will skip this instance
            skip_flag = False
            img = Image.open(img_mask_list[conf_idx])
            InstanceMap = np.array(img)
            image_mask_all += InstanceMap > 0
            if len(np.unique(image_mask_all)) > 2:
                # we will check whether we will keep this instance by looping over all previous instances:
                for im_mask in img_mask_valid:
                    rle1 = maskUtils.encode(np.array(im_mask[:, :, np.newaxis], order='F'))[0]
                    rle2 = maskUtils.encode(np.array(InstanceMap[:, :, np.newaxis], order='F'))[0]
                    iou = maskUtils.iou([rle1], [rle2], [0])
                    if iou[0][0] > args.del_overlap:
                        image_mask_all -= InstanceMap
                        del_count += 1
                        skip_flag = True
                        continue
            if not skip_flag:
                img_mask_valid.append(InstanceMap)
                LabelId = label_list[conf_idx]
                Confidence = conf_list[conf_idx]
                idmap1d = np.reshape(InstanceMap > 0, (-1))
                PixelCount = np.sum(idmap1d)
                EncodedPixels = rle_encoding(InstanceMap)

                df.loc[df_count] = [imageID, LabelId, Confidence, PixelCount, EncodedPixels]
                df_count += 1

    df.to_csv(args.csv_file_image, header=True, index=False)
    print('Finish converting file: %s with %d deleting overlaps' % (args.csv_file_image, del_count))


def convertImages_with_image(filename, list_index,  predictionList, args, mapping_dict):

    df = pd.DataFrame(columns=['ImageId', 'LabelId', 'Confidence', 'PixelCount', 'EncodedPixels'])
    df_count = 0
    imageID = mapping_dict[filename]
    if os.path.exists(predictionList[list_index]):
        predicitionFile = open(predictionList[list_index], "r")
        predictionlines = predicitionFile.readlines()
        for predictionline in predictionlines:
            LabelId = int(predictionInfo[1])
            Confidence = float(predictionInfo[2].split('\n')[0])

            predictionInfo = predictionline.split(' ')
            img = Image.open(predictionInfo[0])
            InstanceMap = np.array(img)
            idmap1d = np.reshape(InstanceMap > 0, (-1))
            PixelCount = np.sum(idmap1d)
            EncodedPixels = rle_encoding(InstanceMap)

            df.loc[df_count] = [imageID, LabelId, Confidence, PixelCount, EncodedPixels]
            df_count += 1

    df.to_csv(args.csv_file_image, header=True, index=False)
    print('Finish converting file: %s.' % (args.csv_file_image))


def convertImages_fast_old(predictionList, groundTruthList, args, mapping_dict):
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
        if args.del_overlap:
            args.csv_file = os.path.join(args.submission_path, videoname + '_del_overlap.csv')
        else:
            args.csv_file = os.path.join(args.submission_path, videoname + '.csv')
        if not os.path.exists(args.submission_path):
            os.mkdir(args.submission_path)
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
        if args.del_overlap:
            convertImages_with_postprocessing(predictionImgList, groundTruthImgList, args, mapping_dict)
        else:
            convertImages_fast(predictionImgList, groundTruthImgList, args, mapping_dict)


if __name__ == '__main__':
    main()