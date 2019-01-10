# Copyright 2018 Zihua Zeng (edvard_hua@live.com)
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===================================================================================
# -*- coding: utf-8 -*-

import tensorflow as tf

from dataset_augment import pose_random_scale, pose_rotation, pose_flip, pose_resize_shortestedge_random, \
    pose_crop_random, pose_to_img
from dataset_prepare import CocoMetadata
from os.path import join
from pycocotools.coco import COCO
import multiprocessing

BASE = "/root/hdd"
BASE_PATH = ""
# TRAIN_JSON = "aug_256x192_single_keypoint_validation_annotations.json"
TRAIN_JSON = 'single_keypoint_train_annotations.json'
VALID_JSON = "single_keypoint_valid_annotations.json"

TRAIN_ANNO = None
CONFIG = None


def set_config(config):
    global CONFIG, BASE, BASE_PATH
    CONFIG = config
    BASE = CONFIG['imgpath']
    BASE_PATH = CONFIG['datapath']

def _parse_function_donothing(imgId, ann=None):
    """
    :param imgId:
    :return:
    """
    global TRAIN_ANNO

    if ann is not None:
        TRAIN_ANNO = ann

    img_meta = TRAIN_ANNO.loadImgs([imgId])[0]
    anno_ids = TRAIN_ANNO.getAnnIds(imgIds=imgId)
    img_anno = TRAIN_ANNO.loadAnns(anno_ids)
    idx = img_meta['id']
    img_path = join(BASE, img_meta['file_name'])

    img_meta_data = CocoMetadata(idx, img_path, img_meta, img_anno, sigma=6.0)
    # img_meta_data = pose_random_scale(img_meta_data)
    # img_meta_data = pose_rotation(img_meta_data)
    # img_meta_data = pose_flip(img_meta_data)
    #img_meta_data = pose_resize_shortestedge_random(img_meta_data)
    # img_meta_data = pose_crop_random(img_meta_data)
    return pose_to_img(img_meta_data)
    
def _parse_function(imgId, ann=None):
    """
    :param imgId:
    :return:
    """
    global TRAIN_ANNO

    if ann is not None:
        TRAIN_ANNO = ann

    img_meta = TRAIN_ANNO.loadImgs([imgId])[0]
    anno_ids = TRAIN_ANNO.getAnnIds(imgIds=imgId)
    img_anno = TRAIN_ANNO.loadAnns(anno_ids)
    idx = img_meta['id']
    img_path = join(BASE, img_meta['file_name'])

    img_meta_data = CocoMetadata(idx, img_path, img_meta, img_anno, sigma=6.0)
    img_meta_data = pose_random_scale(img_meta_data)
    img_meta_data = pose_rotation(img_meta_data)
    img_meta_data = pose_flip(img_meta_data)
    img_meta_data = pose_resize_shortestedge_random(img_meta_data)
    img_meta_data = pose_crop_random(img_meta_data)
    return pose_to_img(img_meta_data)


def _set_shapes(img, heatmap):
    img.set_shape([CONFIG['input_height'], CONFIG['input_width'], 3])
    heatmap.set_shape(
        [CONFIG['input_height'] / CONFIG['scale'], CONFIG['input_width'] / CONFIG['scale'], CONFIG['n_kpoints']])
    return img, heatmap


def _get_dataset_pipline(json_filename, batch_size, epoch, buffer_size):
    global TRAIN_ANNO

    TRAIN_ANNO = COCO(
        join(BASE_PATH, json_filename)
    )
    imgIds = TRAIN_ANNO.getImgIds()

    dataset = tf.data.Dataset.from_tensor_slices(imgIds)

    dataset.shuffle(buffer_size)
    dataset = dataset.map(
        lambda imgId: tuple(
            tf.py_func(
                func=_parse_function,
                inp=[imgId],
                Tout=[tf.float32, tf.float32]
            )
        ), num_parallel_calls=CONFIG['multiprocessing_num'])

    dataset = dataset.map(_set_shapes, num_parallel_calls=CONFIG['multiprocessing_num'])
    dataset = dataset.batch(batch_size).repeat(epoch)
    dataset = dataset.prefetch(100)

    return dataset


def get_train_dataset_pipline(batch_size=32, epoch=10, buffer_size=1):
    return _get_dataset_pipline(TRAIN_JSON, batch_size, epoch, buffer_size, )
