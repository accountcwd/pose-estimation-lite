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
import argparse
from networks import get_network
import os
from tensorflow.python import pywrap_tensorflow
from pprint import pprint

os.environ['CUDA_VISIBLE_DEVICES'] = ''
parser = argparse.ArgumentParser(description='Tensorflow Pose Estimation Graph Extractor')
parser.add_argument('--model', type=str, default='network_SHG_lv2_conv2', help='')
parser.add_argument('--width', type=int, default=128)
parser.add_argument('--height', type=int, default=256)
parser.add_argument('--checkpoint', type=str, default='/home/cwd/project/PoseEstimationForMobile/trained_v6/network_SHG_lv2_conv2_batch-32_lr-0.001_gpus-1_128x256_experiments-multi_network_SHG_lv2_conv2/model-216000', help='checkpoint path')
# parser.add_argument('--output_node_names', type=str, default='Convolutional_Pose_Machine/stage_5_out')
parser.add_argument('--output_node_names', type=str, default='l1_out/BiasAdd')
parser.add_argument('--output_graph', type=str, default='./graph/multi_SHG_lv2_conv2.pb', help='output_freeze_path')

args = parser.parse_args()

input_node = tf.placeholder(tf.float32, shape=[1, args.height, args.width, 3], name="image")

with tf.Session() as sess:
    net = get_network(args.model, input_node, trainable=False)
    saver = tf.train.Saver()
    saver.restore(sess, args.checkpoint)
    # reader = pywrap_tensorflow.NewCheckpointReader(args.checkpoint)
    # var_to_shape_map = reader.get_variable_to_shape_map()
    # for key in var_to_shape_map:
    #     print("tensor_name: ", key)

    input_graph_def = tf.get_default_graph().as_graph_def()
    # for n in tf.get_default_graph().as_graph_def().node:
    #     print(n.name)
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,  # The session
        input_graph_def,  # input_graph_def is useful for retrieving the nodes
        args.output_node_names.split(",")
    )

with tf.gfile.GFile(args.output_graph, "wb") as f:
    f.write(output_graph_def.SerializeToString())