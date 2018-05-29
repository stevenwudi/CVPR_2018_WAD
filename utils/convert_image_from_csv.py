import argparse
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm


def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Customized mapping')
    parser.add_argument('--submission_path',
                        default='/home/stevenwudi/PycharmProjects/CVPR_2018_WAD/results/mask_rcnn_original')
    parser.add_argument('--decode_img_path', default='/home/stevenwudi/PycharmProjects/CVPR_2018_WAD/results/mask_rcnn_original/decode_img')
    args = parser.parse_args()
    return args


def rle_decoding(mask_rle, shape=(2710, 3384)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split('|')
    starts  = [int(x.split(' ')[0]) for x in s[:-1]]
    lengths = [int(x.split(' ')[1]) for x in s[:-1]]
    ends = np.array(starts) + np.array(lengths)
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


def save_to_image(imageId, data, args):
    mask_path = args.decode_img_path
    if not os.path.exists(args.decode_img_path):
        os.makedirs(mask_path)

    target = os.path.join(mask_path, imageId + '.png')

    # Rescale to 0-255 and convert to uint8
    rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)

    # save
    img = Image.fromarray(rescaled)
    img.save(target)
    #print("save_to_image >>>", mask_path)


def main():

    args = parse_args()
    df = pd.read_csv(os.path.join(args.submission_path, 'combined_0.90.csv'))
    prev_im_name = None
    prev_im = np.zeros((2710, 3384), dtype=np.uint8)
    instance_count = 1
    for i in tqdm(range(1, len(df))):
        current_im_name = df.iloc[i]['ImageId']
        mask_info = df.iloc[i]['EncodedPixels']
        im = rle_decoding(mask_info, shape=(2710, 3384))
        if current_im_name == prev_im_name:
            instance_count += 1
            prev_im += im * instance_count
        elif prev_im_name is not None:
            # plt.imshow(prev_im)
            save_to_image(prev_im_name, prev_im, args)
            instance_count = 1
            prev_im = im

        prev_im_name = current_im_name


if __name__ == '__main__':
    main()