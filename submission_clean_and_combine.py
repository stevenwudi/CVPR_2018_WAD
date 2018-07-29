import os
import pandas as pd
import _init_paths  # pylint: disable=unused-import

from submission_file_check import submission_sanity_check
from tqdm import tqdm
from convertVideoToCSV_custom import parse_args
import numpy as np
from datasets.dataloader_wad_cvpr2018 import WAD_CVPR2018


def main():
    args = parse_args()
    frames = []
    dataset = WAD_CVPR2018(args.dataset_dir)
    if args.del_overlap:
        args.submission_path = os.path.join(args.result_dir, 'csv_files_del_overlap')
    else:
        args.submission_path = os.path.join(args.result_dir, 'csv_files')

    print(args.submission_path)
    names = ['ImageId', 'LabelId', 'Confidence', 'PixelCount', 'EncodedPixels']
    for csv_file in os.listdir(args.submission_path):
        df = pd.read_csv(os.path.join(args.submission_path, csv_file), names=names)
        df_new = df[df.PixelCount != 0]
        frames.append(df_new)
        if len(df) != len(df_new):
            print("We have zeros elements in the file: %s" % csv_file)

    sub_csv = pd.concat(frames, axis=0)
    sub_csv = sub_csv.dropna()
    # sub_csv.Confidence = 1      # THIS LINE NEEDS TO BE DELETED!
    #confs = [0.1, 0.2, 0.3, 0.4, 0.5]
    confs = [0.6, 0.7, 0.8, 0.9]
    for i in tqdm(range(len(confs))):
        conf = confs[i]
        sub_csv = sub_csv.reset_index(drop=True)
        sub_csv_new = submission_sanity_check(sub_csv, args.test_img_dir)
        sub_csv_new.Confidence = sub_csv_new.Confidence.astype(float)
        sub_csv_new = sub_csv_new.loc[sub_csv_new['Confidence'] > conf]
        sub_csv_new = sub_csv_new.reset_index(drop=True)
        sub_csv_new = sub_csv_new.reindex(columns=['ImageId', 'LabelId', 'PixelCount', 'Confidence', 'EncodedPixels'])
        sub_csv_new.LabelId = sub_csv_new.LabelId.astype(int)
        sub_csv_new.PixelCount = sub_csv_new.PixelCount.astype(int)
        sub_csv_new.Confidence = sub_csv_new.Confidence.round(6)
        # print some statistics
        count = []
        for l in np.unique(sub_csv_new.LabelId):
            count.append(np.sum(sub_csv_new['LabelId'] == l))

        count_ratio = np.array([x/np.sum(count) for x in count])
        for i, l in enumerate(np.unique(sub_csv_new.LabelId)):
            print("%s --> %d --> %0.3f" %(dataset.id_map_to_cat[l], count[i], count_ratio[i]))
        print('Total instance number for conf %.1f is %d' % (conf, len(sub_csv_new)))

        if args.del_overlap:
            csv_file_name = os.path.join(args.result_dir, 'combined_%.2f_del_overlap.csv' % conf)
        else:
            csv_file_name = os.path.join(args.result_dir, 'combined_%.2f.csv' % conf)
        sub_csv_new.to_csv(csv_file_name, header=True, index=False)
        print("Finish saving file t: %s" % csv_file_name)


if __name__ == '__main__':
    main()


