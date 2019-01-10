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
import os
import time
import numpy as np
import configparser
import dataset

from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
from dataset import get_train_dataset_pipline
from networks import get_network
from dataset_prepare import CocoPose
from dataset_augment import set_network_input_wh, set_network_scale


def get_train_input(batchsize, epoch):
    train_ds = get_train_dataset_pipline(batch_size=batchsize, epoch=epoch, buffer_size=100)
    iter = train_ds.make_one_shot_iterator()
    _ = iter.get_next()
    return _[0], _[1]


def get_loss_and_output(model, batchsize, input_image, input_heat, reuse_variables=None):
    losses = []

    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        _, pred_heatmaps_all = get_network(model, input_image, True)

    for idx, pred_heat in enumerate(pred_heatmaps_all):
        loss_l2 = tf.nn.l2_loss(tf.concat(pred_heat, axis=0) - input_heat, name='loss_heatmap_stage%d' % idx)
        losses.append(loss_l2)

    total_loss = tf.reduce_sum(losses) / batchsize
    total_loss_ll_heat = tf.reduce_sum(loss_l2) / batchsize
    return total_loss, total_loss_ll_heat, pred_heat


def average_gradients(tower_grads):
    """
    Get gradients of all variables.
    :param tower_grads:
    :return:
    """
    average_grads = []

    # get variable and gradients in differents gpus
    for grad_and_vars in zip(*tower_grads):
        # calculate the average gradient of each gpu
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def main(argv=None):
    # load config file and setup
    params = {}
    config = configparser.ConfigParser()
    config_file = "experiments/mv2_hourglass.cfg"
    if len(argv) != 1:
        config_file = argv[1]
    config.read(config_file)
    for _ in config.options("Train"):
        params[_] = eval(config.get("Train", _))

    os.environ['CUDA_VISIBLE_DEVICES'] = params['visible_devices']

    gpus_index = params['visible_devices'].split(",")
    params['gpus'] = len(gpus_index)

    dataset.set_config(params)
    set_network_input_wh(params['input_width'], params['input_height'])
    set_network_scale(params['scale'])

    training_name = '{}_batch-{}_lr-{}_gpus-{}_{}x{}_{}'.format(
        params['model'],
        params['batchsize'],
        params['lr'],
        params['gpus'],
        params['input_width'], params['input_height'],
        config_file.replace("/", "-").replace(".cfg", "")
    )

    with tf.Graph().as_default(), tf.device("/cpu:0"):
        input_image, input_heat = get_train_input(1, params['max_epoch'])


        tower_grads = []
        reuse_variable = False

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        # occupy gpu gracefully
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            print("Start training...")
            fig = plt.figure()
            for step in tqdm(range(0, 100)):
                in_image= sess.run(
                    [input_image]
                )
                print(in_image[0])
                data=CocoPose.get_bgimg(in_image[0,:,:])
                im = Image.fromarray(data)
                im.save("./input/crop/%d_crop.jpg" % step )
                a = fig.add_subplot(1, 2, 1)
                a.set_title('Image')
                plt.imshow(CocoPose.get_bgimg(in_image[index,:,:,:])) 
                plt.show()


if __name__ == '__main__':
    tf.app.run()