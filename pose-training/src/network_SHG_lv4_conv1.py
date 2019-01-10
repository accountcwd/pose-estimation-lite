# -*- coding: utf-8 -*-
# @Time    : 18-4-12 5:12 PM
# @Author  : edvard_hua@live.com
# @FileName: network_mv2_cpm.py
# @Software: PyCharm

import tensorflow as tf
import tensorflow.contrib.slim as slim
from network_base import max_pool, upsample, inverted_bottleneck, separable_conv, convb, conv_fc, conv2d, is_trainable

N_KPOINTS = 14
STAGE_NUM = 3

out_channel_ratio = lambda d: int(d * 1.0)
up_channel_ratio = lambda d: int(d * 1.0)

l2s = []
INPUT_C = 24



def hourglass_module(inp, stage_nums, ST):
    if stage_nums > 0:
        down_sample = max_pool(inp, 2, 2, 2, 2, name="hourglass_downsample_%d_%d" % (ST, stage_nums))

        block_front = slim.stack(down_sample, inverted_bottleneck,
                                 [
                                     #(up_channel_ratio(6), out_channel_ratio(24), 0, 3),
                                     #(up_channel_ratio(6), out_channel_ratio(24), 0, 3),
                                    #  (up_channel_ratio(6), out_channel_ratio(24), 0, 3),
                                    # (up_channel_ratio(6), out_channel_ratio(24), 0, 3),
                                     (up_channel_ratio(6), out_channel_ratio(24), 0, 3),
                                 ], scope="hourglass_front_%d_%d" % (ST, stage_nums))
        stage_nums -= 1
        block_mid = hourglass_module(block_front, stage_nums, ST)
        block_back = inverted_bottleneck(
            block_mid, up_channel_ratio(6), INPUT_C,
            0, 3, scope="hourglass_back_%d_%d" % (ST, stage_nums))

        up_sample = upsample(block_back, 2, "hourglass_upsample_%d_%d" % (ST, stage_nums))

        # jump layer
        branch_jump = slim.stack(inp, inverted_bottleneck,
                                 [
                                     #(up_channel_ratio(6), out_channel_ratio(24), 0, 3),
                                     #(up_channel_ratio(6), out_channel_ratio(24), 0, 3),
                                    #  (up_channel_ratio(6), out_channel_ratio(24), 0, 3),
                                    # (up_channel_ratio(6), out_channel_ratio(24), 0, 3),
                                     (up_channel_ratio(6), INPUT_C, 0, 3),
                                 ], scope="hourglass_branch_jump_%d_%d" % (ST, stage_nums))

        curr_hg_out = tf.add(up_sample, branch_jump, name="hourglass_out_%d_%d" % (ST, stage_nums))
        # mid supervise
        #l2s.append(curr_hg_out) # H,W, N_KPOINTS

        return curr_hg_out

    _ = inverted_bottleneck(
        inp, up_channel_ratio(6), out_channel_ratio(24),
        0, 3, scope="hourglass_mid_%d_%d" % (ST, stage_nums)
    )
    return _


def build_network(input, trainable):
    is_trainable(trainable)

    net = convb(input, 3, 3, out_channel_ratio(16), 2, name="Conv2d_0")

    # 128, 112
    net = slim.stack(net, inverted_bottleneck,
                     [
                         (1, out_channel_ratio(16), 0, 3),
                         (1, out_channel_ratio(16), 0, 3)
                     ], scope="Conv2d_1")

    # 64, 56
    net = slim.stack(net, inverted_bottleneck,
                     [
                         (up_channel_ratio(6), out_channel_ratio(24), 1, 3),
                         (up_channel_ratio(6), out_channel_ratio(24), 0, 3),
                         (up_channel_ratio(6), out_channel_ratio(24), 0, 3),
                         (up_channel_ratio(6), out_channel_ratio(24), 0, 3),
                         (up_channel_ratio(6), out_channel_ratio(INPUT_C), 0, 3), #24
                     ], scope="Conv2d_2")

    net_h_w = int(net.shape[1])
    # build network recursively
    
    hg_out0 = hourglass_module(net, STAGE_NUM, 0)
    ll_0 = convb(hg_out0, 1, 1, INPUT_C, 1, name='Conv2d_l0')
    l0_out = conv2d(ll_0, 1, 1, N_KPOINTS, 1, name='l0_out')
    l2s.append(l0_out)
    l0_out_ = conv2d(l0_out, 1, 1, INPUT_C, 1, name='Conv2d_l0_')
    ll_0_ = conv2d(ll_0, 1, 1, INPUT_C, 1, name='l0_out_')
    in_1 = tf.add_n([net, l0_out_, ll_0_])

    hg_out1 = hourglass_module(in_1, STAGE_NUM, 1)
    ll_1 = convb(hg_out1, 1, 1, INPUT_C, 1, name='Conv2d_l1')
    l1_out = conv2d(ll_1, 1, 1, N_KPOINTS, 1, name='l1_out')
    l2s.append(l1_out)
    l1_out_ = conv2d(l1_out, 1, 1, INPUT_C, 1, name='Conv2d_l1_')
    ll_1_ = conv2d(ll_1, 1, 1, INPUT_C, 1, name='l1_out_')
    in_2 = tf.add_n([in_1, l1_out_, ll_1_])

    hg_out2 = hourglass_module(in_2, STAGE_NUM, 2)
    ll_2 = convb(hg_out2, 1, 1, INPUT_C, 1, name='Conv2d_l2')
    l2_out = conv2d(ll_2, 1, 1, N_KPOINTS, 1, name='l2_out')
    l2s.append(l2_out)
    l2_out_ = conv2d(l2_out, 1, 1, INPUT_C, 1, name='Conv2d_l2_')
    ll_2_ = conv2d(ll_2, 1, 1, INPUT_C, 1, name='l2_out_')
    in_3 = tf.add_n([in_2, l2_out_, ll_2_])
    
    hg_out3 = hourglass_module(in_3, STAGE_NUM, 3)
    ll_3 = convb(hg_out3, 1, 1, INPUT_C, 1, name='Conv2d_l3')
    l3_out = conv2d(ll_3, 1, 1, N_KPOINTS, 1, name='l3_out')
    l2s.append(l3_out)
    

    for index, l2 in enumerate(l2s):
        l2_w_h = int(l2.shape[1])
        if l2_w_h == net_h_w:
            continue
        assert 'wrong place' == ''
        scale = net_h_w // l2_w_h
        l2s[index] = upsample(l2, scale, name="upsample_for_loss_%d" % index)

    return l3_out, l2s
