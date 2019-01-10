import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms

import torch.nn as nn
import torch.utils.data
import numpy as np
from opt import opt

from dataloader_webcam import WebcamLoader, DetectionLoader, DetectionProcessor, DataWriter, crop_from_dets, Mscoco
from yolo.darknet import Darknet
from yolo.util import write_results, dynamic_write_results
from SPPE.src.main_fast_inference import *

from SPPE.src.utils.img import im_to_torch
import os
import sys
from tqdm import tqdm
import time
from fn import getTime
import cv2

from pPose_nms import write_json

import tensorflow as tf
from scipy.ndimage.filters import gaussian_filter

args = opt
args.dataset = 'coco'


def mark_keypoint(_img, x, y):
    for i in range(-3,4):
        for j in range(-3,4):
            if not out_of_range(x+i, y+j, _img.shape[0], _img.shape[1]):
                _img[x+i,y+j,:] =[0, 255, 255]
    return _img

def out_of_range(x, y, w, h):
    return (x<0 or y<0 or x>=w or y>=h)

def loop():
    n = 0
    while True:
        yield n
        n += 1

if __name__ == "__main__":
    webcam = args.webcam
    mode = args.mode
    if not os.path.exists(args.outputpath):
        os.mkdir(args.outputpath)

    # Load input video
    data_loader = WebcamLoader(webcam).start()
    (fourcc,fps,frameSize) = data_loader.videoinfo()

    # Load detection loader
    print('Loading YOLO model..')
    sys.stdout.flush()
    det_loader = DetectionLoader(data_loader, batchSize=args.detbatch).start()
    det_processor = DetectionProcessor(det_loader).start()
    
    # Load pose model
    # pose_dataset = Mscoco()
    # if args.fast_inference:
        # pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
    # else:
    #     pose_model = InferenNet(4 * 1 + 1, pose_dataset)
    # pose_model.cuda()
    # pose_model.eval()

    # Data writer
    save_path = os.path.join(args.outputpath, 'AlphaPose_webcam'+webcam+'.avi')
    writer = DataWriter(args.save_video, save_path, cv2.VideoWriter_fourcc(*'XVID'), fps, frameSize).start()

    runtime_profile = {
        'dt': [],
        'pt': [],
        'pn': []
    }

    print('Starting webcam demo, press Ctrl + C to terminate...')
    sys.stdout.flush()
    im_names_desc =  tqdm(loop())
    batchSize = args.posebatch

    #tensorflow stuff
    
    frozen_pb_path ='/home/cwd/project/PoseEstimationForMobile/training/graph/mv2_stacked_hourglass_3x3_128x256_nobn.pb'
    output_node_name ='l2_out/BiasAdd'
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
    with tf.Session() as sess:
        for i in im_names_desc:
            try:
                start_time = getTime()            
                # with torch.no_grad():
                (inps, orig_img, im_name, boxes, scores, pt1, pt2) = det_processor.read()
                if boxes is None or boxes.nelement() == 0:
                    writer.save(None, None, None, None, None, orig_img, im_name.split('/')[-1])
                    continue
                # print(inps.size(), orig_img.shape, im_name, boxes.size(), scores.size(), pt1.size(), pt2.size())
                ckpt_time, det_time = getTime(start_time)
                runtime_profile['dt'].append(det_time)
                # Pose Estimation
                
                datalen = inps.size(0)
                
                # leftover = 0
                # if (datalen) % batchSize:
                #     leftover = 1
                # num_batches = datalen // batchSize + leftover
                # hm = []
                # for j in range(num_batches):
                #     inps_j = inps[j*batchSize:min((j +  1)*batchSize, datalen)].cuda()
                    # hm_j = pose_model(inps_j)
                #     hm.append(hm_j)
                # hm = torch.cat(hm)
                # ckpt_time, pose_time = getTime(ckpt_time)
                # runtime_profile['pt'].append(pose_time)

                # hm = hm.cpu().data
                hm = []
                ckpt_time = getTime()
                _time = getTime()
                for i in range(datalen):
                    inp_img = np.array(inps[i,:,:,:]).transpose(1,2,0)
                    # cv2.imshow('img0',inp_img.astype(np.uint8))
                    # cv2.waitKey(0)
                    heat = sess.run(output_heat, feed_dict={input_image: [inp_img]}) #(1, 48, 48, 14) #20ms                    
                    #heatmaps=heat
                    # for i in range(heatmaps.shape[3]):
                    #     heatmap = heatmaps[0, :, :, i]
                    #     heatmap = gaussian_filter(heatmap, sigma=5)
                    #     ind = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                    #     coord_x = int((ind[1] + 1) *4)
                    #     coord_y = int((ind[0] + 1) *4)
                    #     inp_ = inp_img
                    #     inp_ = mark_keypoint(inp_, coord_y, coord_x)
                    #     cv2.namedWindow('img0',0)
                    #     cv2.imshow('img0',inp_.astype(np.uint8))
                    #     # print(max(heatmap0))
                    #     cv2.waitKey(0)
                    tf_out = heat.transpose(0,3,1,2)
                    tf_out = torch.from_numpy(tf_out)
                    # print(tf_out.size())
                    hm.append(tf_out)
                    
                # cv2.destroyAllWindows()
                ckpt_time, post_time = getTime(ckpt_time)
                print('det time:%4f pose time: %4f boxes:%d'% (det_time, post_time ,datalen))
                hm = torch.cat(hm)  #(N, 14, 48, 48)s
                writer.save(boxes, scores, hm, pt1, pt2, orig_img, im_name.split('/')[-1])
          
                runtime_profile['pn'].append(post_time)
                if args.profile:
                    # TQDM
                    im_names_desc.set_description(
                    'det time: {dt:.3f} | pose time: {pt:.2f} | post processing: {pn:.4f}'.format(
                        dt=np.mean(runtime_profile['dt']), pt=np.mean(runtime_profile['pt']), pn=np.mean(runtime_profile['pn']))
                    )
            except KeyboardInterrupt:
                break

    print(' ')
    print('===========================> Finish Model Running.')
    if (args.save_img or args.save_video) and not args.vis_fast:
        print('===========================> Rendering remaining images in the queue...')
        print('===========================> If this step takes too long, you can enable the --vis_fast flag to use fast rendering (real-time).')
    while(writer.running()):
        pass
    writer.stop()
    final_result = writer.results()
    write_json(final_result, args.outputpath)
