# -*- coding: utf-8 -*-
# @Time    : 18-7-10 上午9:41
# @Author  : zengzihua@huya.com
# @FileName: benchmark.py
# @Software: PyCharm

import tensorflow as tf
import numpy as np
import json
import argparse
import cv2
import os
import math
import time
import tqdm
from scipy.ndimage.filters import gaussian_filter
MAX = 30000

def mark_keypoint(_img, x, y):
    for i in range(-3,4):
        for j in range(-3,4):
            if not out_of_range(x+i, y+j, _img.shape[0], _img.shape[1]):
                _img[x+i,y+j,:] =[0, 255, 255]
    return _img

def out_of_range(x, y, w, h):
    return (x<0 or y<0 or x>=w or y>=h)

def cal_coord(pred_heatmaps, images_anno, img_path):
    coords = {}
    for img_id in pred_heatmaps.keys():
        heat_h, heat_w, n_kpoints = pred_heatmaps[img_id].shape
        scale_h, scale_w = heat_h / images_anno[img_id]['height'], heat_w / images_anno[img_id]['width']
        coord = []
        ori_img = cv2.imread(os.path.join(img_path, images_anno[img_id]['file_name']))
        for p_ind in range(n_kpoints):
            heat = pred_heatmaps[img_id][:, :, p_ind]
            heat = gaussian_filter(heat, sigma=5)
            ind = np.unravel_index(np.argmax(heat), heat.shape)
            coord_x = int((ind[1] + 1) / scale_w)
            coord_y = int((ind[0] + 1) / scale_h)
            coord.append((coord_x, coord_y, heat[ind[0],ind[1]]))
            # if img_id % 20==0:
            #     heat_r = cv2.resize(heat, (images_anno[img_id]['width'], images_anno[img_id]['height']))
            #     heat_re = heat_r /np.max(heat_r)*255
            #     ori_ = np.mean(ori_img, 2)
            #     ori_ = ori_.astype(np.uint8)
            #     ori_ +=heat_re.astype(np.uint8)
            #     cv2.namedWindow('img0',0)
            #     cv2.namedWindow('img1',0)
            #     cv2.namedWindow('img2',0)
            #     cv2.imshow('img0',heat_re.astype(np.uint8))
            #     cv2.imshow('img1',ori_img.astype(np.uint8))
            #     cv2.imshow('img2',ori_.astype(np.uint8))
            #     cv2.waitKey(0)
            #     ind = np.unravel_index(np.argmax(heat), heat.shape)
            #     coord_x = int((ind[1] + 1) / scale_w)
            #     coord_y = int((ind[0] + 1) / scale_h)
            #     print(heat_r.shape)
            #     print(coord_x, coord_y)
            #     print(heat_r[coord_y-1,coord_x-1])
            

        coords[img_id] = coord

    cv2.destroyAllWindows()

    return coords


def infer(frozen_pb_path, output_node_name, img_path, images_anno):
    with tf.gfile.GFile(frozen_pb_path, "rb") as f:
        restored_graph_def = tf.GraphDef()
        restored_graph_def.ParseFromString(f.read())

    tf.import_graph_def(
        restored_graph_def,
        input_map=None,
        return_elements=None,
        name=""
    )

    graph = tf.get_default_graph()
    input_image = graph.get_tensor_by_name("image:0")
    output_heat = graph.get_tensor_by_name("%s:0" % output_node_name)

    res = {}
    use_times = []
    count=0
    with tf.Session() as sess:
        for img_id in images_anno.keys():
            count+=1
            # if count >MAX+1:break
            ori_img = cv2.imread(os.path.join(img_path, images_anno[img_id]['file_name']))
            shape = input_image.get_shape().as_list()
            inp_img = cv2.resize(ori_img, (shape[2], shape[1]))
            st = time.time()
            heat = sess.run(output_heat, feed_dict={input_image: [inp_img]})
            # heatmaps=heat
            # for i in range(heatmaps.shape[3]):
            #     heatmap0 = inp_img
            #     heatmap0[:,:,0] += cv2.resize(heatmaps[0,:,:,i]/np.max(heatmaps[0,:,:,i]) *255, (shape[2], shape[1])).astype(np.uint8)
            #     heatmap0[:,:,2] += cv2.resize(heatmaps[0,:,:,i]/np.max(heatmaps[0,:,:,i]) *255, (shape[2], shape[1])).astype(np.uint8)
            #     heatmap0[:,:,1] += cv2.resize(heatmaps[0,:,:,i]/np.max(heatmaps[0,:,:,i]) *255, (shape[2], shape[1])).astype(np.uint8)
            #     cv2.namedWindow('img0',0)
            #     cv2.imshow('img0',heatmap0.astype(np.uint8))
            #     print(heatmap0.astype(np.uint8))
            #     cv2.waitKey(0)
            # cv2.destroyAllWindows()
            infer_time = 1000 * (time.time() - st)
            # print("img_id = %d, cost_time = %.2f ms" % (img_id, infer_time))
            use_times.append(infer_time)
            res[img_id] = np.squeeze(heat)  
    print("Average inference time = %.2f ms" % np.mean(use_times))
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PCKh benchmark")
    parser.add_argument("--prefix", type=str, default='lv2_conv2')
    parser.add_argument("--frozen_pb_path", type=str, default="/home/cwd/project/PoseEstimationForMobile/training/graph/multi_SHG_lv2_conv2.pb")
    # parser.add_argument("--anno_json_path", type=str, default="/home/cwd/DateSet/ai_challenger_2017/single_1.2box/single_keypoint_valid_annotations.json")
    # parser.add_argument("--img_path", type=str, default="/home/cwd/DateSet/ai_challenger_2017/single_1.2box")
    # parser.add_argument("--anno_json_path", type=str, default="/home/cwd/DateSet/ai_challenger_2017/single_only/single_keypoint_validation_annotations.json")
    # parser.add_argument("--img_path", type=str, default="/home/cwd/DateSet/ai_challenger_2017/single_only")
    parser.add_argument("--anno_json_path", type=str, default="/home/cwd/DateSet/ai_challenger_2017/multi_refine/single_keypoint_valid_annotations.json")
    parser.add_argument("--img_path", type=str, default="/home/cwd/DateSet/ai_challenger_2017/multi_refine")
    # parser.add_argument("--anno_json_path", type=str, default="/home/cwd/DateSet/ai_challenger/ai_challenger_valid.json")
    # parser.add_argument("--img_path", type=str, default="/home/cwd/DateSet")
    # parser.add_argument("--anno_json_path", type=str, default="/home/cwd/DateSet/ai_challenger/keypoint1.8/ai_challenger_valid.json")
    # parser.add_argument("--img_path", type=str, default="/home/cwd/DateSet//ai_challenger/keypoint1.8")
    # parser.add_argument("--anno_json_path", type=str, default="/home/cwd/DateSet/ai_challenger/keypoint1.1/ai_challenger_valid.json")
    # parser.add_argument("--img_path", type=str, default="/home/cwd/DateSet//ai_challenger/keypoint1.1")
    # parser.add_argument("--output_node_name", type=str, default="hourglass_out_3")
    parser.add_argument("--output_node_name", type=str, default="l1_out/BiasAdd")
    
    parser.add_argument("--gpus", type=str, default="0")
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    anno = json.load(open(args.anno_json_path))
    print("Total test example=%d" % len(anno['images']))

    images_anno = {}
    keypoint_annos = {}
    # transform = list(zip(
    #     [1, 2, 4, 6, 8, 3, 5, 7, 10, 12, 14, 9, 11, 13],
    #     [1, 2, 4, 6, 8, 3, 5, 7, 10, 12, 14, 9, 11, 13]
    # ))
    # transform = list(zip(
    #     [4, 6, 8, 3, 5, 7, 10, 12, 14, 9, 11, 13, 1, 2],
    #     [4, 6, 8, 3, 5, 7, 10, 12, 14, 9, 11, 13, 1, 2]
    # ))
    # transform = list(zip(
    #     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    #     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    # ))
    transform = list(zip(
        [13, 14, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        [13, 14, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    )) #oringin
    
    for img_info, anno_info in zip(anno['images'], anno['annotations']):
        images_anno[img_info['id']] = img_info

        prev_xs = anno_info['keypoints'][0::3]
        prev_ys = anno_info['keypoints'][1::3]
        prev_lab = anno_info['keypoints'][2::3]
        new_kp = []
        for idx, idy in transform:
            new_kp.append(
                (prev_xs[idx-1], prev_ys[idy-1], prev_lab[idy-1])
            )

        keypoint_annos[anno_info['image_id']] = new_kp

    pred_heatmap = infer(args.frozen_pb_path, args.output_node_name, args.img_path, images_anno)
    pred_coords = cal_coord(pred_heatmap, images_anno, args.img_path)

    scores = []
    count=0
    for img_id in keypoint_annos.keys():
        if img_id==MAX:break
        groundtruth_anno = keypoint_annos[img_id]

        head_gt = groundtruth_anno[0]
        neck_gt = groundtruth_anno[1]
        # head_gt = groundtruth_anno[12]
        # neck_gt = groundtruth_anno[13]

        threshold = math.sqrt((head_gt[0] - neck_gt[0]) ** 2 + (head_gt[1] - neck_gt[1]) ** 2)
        ori_img = cv2.imread(os.path.join(args.img_path, images_anno[img_id]['file_name']))
        curr_score = []
        point_ok =0
        point_fail= 0
        for index, coord in enumerate(pred_coords[img_id]):
            
            pred_x, pred_y ,score= coord
            gt_x, gt_y, label = groundtruth_anno[index]
                            
            d = math.sqrt((pred_x-gt_x)**2 + (pred_y-gt_y)**2)
            if d > threshold:
                curr_score.append(0)
            else:
                curr_score.append(1)
        curr_score_ = np.mean(curr_score)                
        scores.append(curr_score_)
        # if curr_score_ <= 0.5:
        if img_id <=50:
            for index, coord in enumerate(pred_coords[img_id]):
                
                pred_x, pred_y ,score= coord
                if score>0.01:
                    mark_keypoint(ori_img, pred_y, pred_x)           
                cv2.imwrite( '../infer/%d_%s.jpg' %(count,args.prefix) , ori_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        count+=1

    print("PCKh=%.2f" % (np.mean(scores) * 100))

    
    with open('result.txt', 'a') as f:
        f.write('%s : %f \n' % (args.prefix ,np.mean(scores) * 100))
        
