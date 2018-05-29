import copy
import itertools
import json
import os
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from pycocotools import mask as maskUtils


class WAD_CVPR2018:
    def __init__(self, dataset_dir):
        """
        Constructor of WAD_CVPR2018 helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset_dir = dataset_dir
        self.image_shape = (2710, 3384)  # Height, Width
        self.train_image_dir = os.path.join(dataset_dir, 'train_color')
        self.test_image_dir = os.path.join(dataset_dir, 'test')
        self.train_label_dir = os.path.join(dataset_dir, 'train_label')
        self.train_video_list_dir = os.path.join(self.dataset_dir, 'train_video_list')
        self.img_video_id = self.getImageVideoIds()

        self.category_to_id_map = {
            'car': 33,
            'motorcycle': 34,
            'bicycle': 35,
            'pedestrian': 36,
            'rider': 37,
            'truck': 38,
            'bus': 39,
            'tricycle': 40,
            'others': 0,
            'rover': 1,
            'sky': 17,
            'car_groups': 161,
            'motorbicycle_group': 162,
            'bicycle_group': 163,
            'person_group': 164,
            'rider_group': 165,
            'truck_group': 166,
            'bus_group': 167,
            'tricycle_group': 168,
            'road': 49,
            'siderwalk': 50,
            'traffic_cone': 65,
            'road_pile': 66,
            'fence': 67,
            'traffic_light': 81,
            'pole': 82,
            'traffic_sign': 83,
            'wall': 84,
            'dustbin': 85,
            'billboard': 86,
            'building': 97,
            'bridge': 98,
            'tunnel': 99,
            'overpass': 100,
            'vegatation': 113,
            'unlabeled': 255,
        }

        self.id_map_to_cat = dict(zip(self.category_to_id_map.values(), self.category_to_id_map.keys()))

        self.eval_cat = {'car', 'motorcycle', 'bicycle', 'pedestrian', 'truck', 'bus', 'tricycle'}

        self.eval_class = [self.category_to_id_map[x] for x in self.eval_cat]
        # Due to previous training, we need to set the order as follows
        self.eval_class = [39, 40, 34, 33, 38, 36, 35]
        self.json_category_id_to_contiguous_id = {
            v: i + 1
            for i, v in enumerate(self.eval_class)
        }

        self.contiguous_category_id_to_json_id = {
            v: k
            for k, v in self.json_category_id_to_contiguous_id.items()
        }

    def getImageVideoIds(self):
        image_video_ids = []
        for fname in os.listdir(self.train_video_list_dir):
            image_video = []
            f = open(os.path.join(self.train_video_list_dir, fname), 'r')
            img_list = f.readlines()
            f.close()
            for line in img_list:
                img_id_line = line.split('\t')[0]
                img_id = img_id_line.split('\\')[-1]
                image_video.append(img_id)
            image_video_ids.append(image_video)

        return image_video_ids
