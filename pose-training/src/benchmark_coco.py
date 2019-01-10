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
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
MAX = 8000

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
            keypoint.extend(coord_x, coord_y, 2)
            coord.append((coord_x, coord_y, heat[ind[0],ind[1]]))

            

        coords[img_id] = coord
        keypoints.append()

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

    result = []
    use_times = []
    count=0
    with tf.Session() as sess:
        for img_id in images_anno.keys():
            count+=1
            # if count >MAX+1:break
            ori_h, ori_w = images_anno[img_id]['height'], images_anno[img_id]['width'] 
            ori_img = cv2.imread(os.path.join(img_path, images_anno[img_id]['file_name']))
            shape = input_image.get_shape().as_list()
            inp_img = cv2.resize(ori_img, (shape[2], shape[1]))
            st = time.time()
            heats_run = sess.run(output_heat, feed_dict={input_image: [inp_img]})
            heats = np.squeeze(heats_run)
            infer_time = 1000 * (time.time() - st)
            print("img_id = %d, cost_time = %.2f ms" % (img_id, infer_time))
            use_times.append(infer_time)
            heat_h, heat_w  = heats.shape[0], heats.shape[1]
            scale_h, scale_w = heat_h / ori_h, heat_w / ori_w
            keypoint=[]
            klist = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 1]
            for p_ind in klist:
                heat = heats[:, :, p_ind]
                heat = gaussian_filter(heat, sigma=5)
                ind = np.unravel_index(np.argmax(heat), heat.shape)
                coord_x = int((ind[1] + 1) / scale_w)
                coord_y = int((ind[0] + 1) / scale_h)
                keypoint.extend([coord_x, coord_y, 1])
                # mark_keypoint(ori_img, coord_y, coord_x)

            # cv2.namedWindow('img0',0)
            # cv2.imshow('img0',ori_img.astype(np.uint8))
            # cv2.waitKey(0)        
            # cv2.destroyAllWindows()
            keypoint.extend([0,0,0,0,0,0,0,0,0])
            item = {
                'image_id': images_anno[img_id]['id'],
                'category_id': 1,
                'keypoints': keypoint,
                'score': 1
            }
            result.append(item)
                
    print("Average inference time = %.2f ms" % np.mean(use_times))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PCKh benchmark")
    parser.add_argument("--prefix", type=str, default='multi_lv2_conv2')
    parser.add_argument("--frozen_pb_path", type=str, default="/home/cwd/project/PoseEstimationForMobile/training/graph/multi_SHG_lv2_conv2.pb")
    # parser.add_argument("--anno_json_path", type=str, default="/home/cwd/DateSet/ai_challenger_2017/single_1.2box/single_keypoint_valid_annotations.json")
    # parser.add_argument("--img_path", type=str, default="/home/cwd/DateSet/ai_challenger_2017/single_1.2box")
    parser.add_argument("--anno_json_path", type=str, default="/home/cwd/DateSet/ai_challenger_2017/single_only/single_keypoint_validation_annotations.json")
    parser.add_argument("--img_path", type=str, default="/home/cwd/DateSet/ai_challenger_2017/single_only")
    # parser.add_argument("--anno_json_path", type=str, default="/home/cwd/DateSet/ai_challenger_2017/multi_refine/single_keypoint_valid_annotations.json")
    # parser.add_argument("--img_path", type=str, default="/home/cwd/DateSet/ai_challenger_2017/multi_refine")
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

    results = infer(args.frozen_pb_path, args.output_node_name, args.img_path, images_anno)

    write_json = './%s.json' % (args.prefix)
    with open(write_json,'w') as fp:
        json.dump(results, fp)

    cocoGt = COCO(args.anno_json_path)
    catIds = cocoGt.getCatIds(catNms=['human'])
    keys = cocoGt.getImgIds(catIds=catIds)
    cocoDt = cocoGt.loadRes(write_json)
    cocoEval = COCOeval(cocoGt, cocoDt, 'keypoints')
    cocoEval.params.imgIds = keys
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
