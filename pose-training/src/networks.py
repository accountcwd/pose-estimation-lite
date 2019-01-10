# -*- coding: utf-8 -*-
# @Time    : 18-3-6 3:20 PM
# @Author  : zengzihua@huya.com
# @FileName: data_filter.py
# @Software: PyCharm

import network_mv2_cpm
import network_mv2_hourglass
import network_mv2_hourglass_slim
import network_mv2_stacked_hourglass_2x3
import network_mv2_stacked_hourglass_3x3
import network_SHG_lv2_conv2_3
import network_SHG_lv2_conv2
import network_SHG_lv2_conv3_backbone3
import network_SHG_lv3_conv1_2
import network_SHG_lv3_conv1
import network_SHG_lv4_conv1

def get_network(type, input, trainable=True):
    if type == 'mv2_cpm':
        net, loss = network_mv2_cpm.build_network(input, trainable)
    elif type == "mv2_hourglass":
        net, loss = network_mv2_hourglass.build_network(input, trainable)        
    elif type == "mv2_stacked_hourglass_2x3":
        net, loss = network_mv2_stacked_hourglass_2x3.build_network(input, trainable) 
    elif type == "mv2_stacked_hourglass_3x3":
        net, loss = network_mv2_stacked_hourglass_3x3.build_network(input, trainable) 
    elif type == "network_SHG_lv2_conv2_3":
        net, loss = network_SHG_lv2_conv2_3.build_network(input, trainable)                
    elif type == "network_SHG_lv2_conv2":
        net, loss = network_SHG_lv2_conv2.build_network(input, trainable)        
    elif type == "network_SHG_lv2_conv3_backbone3":
        net, loss = network_SHG_lv2_conv3_backbone3.build_network(input, trainable)        
    elif type == "network_SHG_lv3_conv1_2":
        net, loss = network_SHG_lv3_conv1_2.build_network(input, trainable)  
    elif type == "network_SHG_lv3_conv1":
        net, loss = network_SHG_lv3_conv1.build_network(input, trainable) 
    elif type == "network_SHG_lv4_conv1":
        net, loss = network_SHG_lv4_conv1.build_network(input, trainable) 
    return net, loss
