from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image
import pandas as pd
import random 
import argparse
import pickle as pkl

from utils.img import  cropBox, im_to_torch, transformBoxInvert_batch
import tensorflow as tf

poseResH = 256
poseResW = 128

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)

def vis_frame_fast(frame, res, format='coco'):
    '''
    frame: frame image
    im_res: im_res of predictions
    format: coco or mpii

    return rendered image
    '''
    if format == 'coco':
        l_pair = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Head
            (1, 5), (5, 6), (6, 7), (1, 8), (8, 9), 
            (9, 10), (1, 11), (11, 12),  # Body
            (12, 13) 
        ]
        p_color = [(0, 255, 255), (0, 191, 255),(0, 255, 102),(0, 77, 255), (0, 255, 0), #Nose, LEye, REye, LEar, REar
                    (77,255,255), (77, 255, 204), (77,204,255), (191, 255, 77), (77,191,255), (191, 255, 77), #LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                    (204,77,255), (77,255,204), (191,77,255), (77,255,191), (127,77,255), (77,255,127), (0, 255, 255)] #LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
        line_color = [PURPLE, RED, RED, RED, BLUE, BLUE, BLUE,
                    RED, RED, RED, BLUE, BLUE, BLUE,
                    (255,127,77), (0,77,255), (255,77,36)]
    else:
        NotImplementedError

    # im_name = im_res['imgname'].split('/')[-1]
    img = frame

    for human in res:
        part_line = {}
        kp_preds = human['keypoints']
        kp_scores = human['kp_score']
        bbox = human['bbox'] #cwd edit
        print(kp_preds)
        print(kp_scores)
        # kp_preds = torch.cat((kp_preds, torch.unsqueeze((kp_preds[5,:]+kp_preds[6,:])/2,0)))
        # kp_scores = torch.cat((kp_scores, torch.unsqueeze((kp_scores[5,:]+kp_scores[6,:])/2,0)))



        #Draw bbox cwd edit
        # flag = human['is_hm'].numpy() * 255
        bbox = human['bbox'] #cwd edit
        bg = img.copy()
        img = cv2.rectangle(bg, (bbox[0], bbox[1]), (bbox[2], bbox[3]),(0 ,255 , 255))
        # Draw keypoints
        for n in range(kp_scores.shape[0]):
            if kp_scores[n] <= 0.1:
                continue
            cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
            part_line[n] = (cor_x, cor_y)
            cv2.circle(img, (cor_x, cor_y), 4, p_color[n], -1)
        # Draw limbs
        for i, (start_p, end_p) in enumerate(l_pair):
            if start_p in part_line and end_p in part_line:
                start_xy = part_line[start_p]
                end_xy = part_line[end_p]
                cv2.line(img, start_xy, end_xy, line_color[i], 2*(kp_scores[start_p] + kp_scores[end_p]) + 1)
    return img

def getPrediction(hms, pt1, pt2, inpH, inpW, resH, resW):
    '''
    Get keypoint location from heatmaps
    '''

    assert hms.dim() == 4, 'Score maps should be 4-dim'
    maxval, idx = torch.max(hms.view(hms.size(0), hms.size(1), -1), 2)
    maxval = maxval.view(hms.size(0), hms.size(1), 1)
    idx = idx.view(hms.size(0), hms.size(1), 1) + 1 # (N, 14, 1)
    preds = idx.repeat(1, 1, 2).float() #(N, 14, 2) 记录idx
    preds[:, :, 0] = (preds[:, :, 0] - 1) % hms.size(3)
    preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / hms.size(3))

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()  
    preds *= pred_mask  #将预测最大值为0的坐标值置0屏蔽    
    # Very simple post-processing step to improve performance at tight PCK thresholds
    for i in range(preds.size(0)):   # N
        for j in range(preds.size(1)): # keypoint
            hm = hms[i][j]
            pX, pY = int(round(float(preds[i][j][0]))), int(round(float(preds[i][j][1])))
            if 0 < pX < resW - 1 and 0 < pY < resH - 1:
                diff = torch.Tensor(
                    (hm[pY][pX + 1] - hm[pY][pX - 1], hm[pY + 1][pX] - hm[pY - 1][pX]))
                preds[i][j] += diff.sign() * 0.25  #加上梯度
    preds += 0.2

    preds_tf = torch.zeros(preds.size())

    preds_tf = transformBoxInvert_batch(preds, pt1, pt2, inpH, inpW, resH, resW)
   
    return preds, preds_tf, maxval

def crop_from_dets(img, boxes):
    '''
    Crop human from origin image according to Dectecion Results
    '''

    imght = img.size(1)
    imgwidth = img.size(2)
    tmp_img = img
    inps = []
    pt1 = []
    pt2 = []
    for i, box in enumerate(boxes):
        upLeft = torch.Tensor(
            (float(box[0]), float(box[1])))
        bottomRight = torch.Tensor(
            (float(box[2]), float(box[3])))

        ht = bottomRight[1] - upLeft[1]
        width = bottomRight[0] - upLeft[0]
        if width > 100:
            scaleRate = 0.2
        else:
            scaleRate = 0.3

        upLeft[0] = max(0, upLeft[0] - width * scaleRate / 2)
        upLeft[1] = max(0, upLeft[1] - ht * scaleRate / 2)
        bottomRight[0] = max(
            min(imgwidth - 1, bottomRight[0] + width * scaleRate / 2), upLeft[0] + 5)
        bottomRight[1] = max(
            min(imght - 1, bottomRight[1] + ht * scaleRate / 2), upLeft[1] + 5)

        inps.append(cropBox(tmp_img, upLeft, bottomRight, poseResH, poseResW))
        pt1.append(upLeft)
        pt2.append(bottomRight)

    return inps, pt1, pt2
    
def get_test_input(input_dim, CUDA):
    img = cv2.imread("imgs/messi.jpg")
    img = cv2.resize(img, (input_dim, input_dim)) 
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    
    if CUDA:
        img_ = img_.cuda()
    
    return img_

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = cv2.resize(orig_im, (inp_dim, inp_dim))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

def write(x, img):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img

def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    
    parser = argparse.ArgumentParser(description='YOLO v3 Cam Demo')
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.80)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "160", type = str)
    return parser.parse_args()



if __name__ == '__main__':
    cfgfile = "cfg/yolov3.cfg"
    weightsfile = "/home/cwd/project/AlphaPose_old/models/yolo/yolov3.weights"
    num_classes = 80

    args = arg_parse()
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0
    CUDA = torch.cuda.is_available()
    

    
    
    num_classes = 80
    bbox_attrs = 5 + num_classes
    
    model = Darknet(cfgfile)
    model.load_weights(weightsfile)
    
    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    
    assert inp_dim % 32 == 0 
    assert inp_dim > 32

    if CUDA:
        model.cuda()
            
    model.eval()
    
    videofile = 'video.avi'
    
    cap = cv2.VideoCapture(0)
    
    assert cap.isOpened(), 'Cannot capture source'
    
    frames = 0
    # tensorflow stuff
    frozen_pb_path ='/home/cwd/project/PoseEstimationForMobile/training/graph/SHG_2v2_cls.pb'
    output_node_name ='l1_out/BiasAdd'
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
    output_cls =  graph.get_tensor_by_name("cls_net/cls_reshape:0")

    with tf.Session() as sess:        
        while cap.isOpened():
            try:
                start = time.time()    
                ret, frame = cap.read() #480x640
                if ret:
                    
                    img, orig_im, dim = prep_image(frame, inp_dim)
                    im_dim = torch.FloatTensor(dim).repeat(1,2)                        
                                    
                    if CUDA:
                        im_dim = im_dim.cuda()
                        img = img.cuda()
                    
                    output = model(Variable(img), CUDA)
                    #[1:5] = x1,y1,x2,y2
                    output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)
                    
                    if type(output) == int:
                        frames += 1
                        print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
                        cv2.imshow("frame", orig_im)
                        key = cv2.waitKey(1)
                        if key & 0xFF == ord('q'):
                            break
                        continue
                    

                
                    output[:,1:5] = torch.clamp(output[:,1:5], 0.0, float(inp_dim))/inp_dim
                    
        #            im_dim = im_dim.repeat(output.size(0), 1)
                    output[:,[1,3]] *= frame.shape[1]
                    output[:,[2,4]] *= frame.shape[0]
                    box_scores = output[:, 5:6]
                    inps, pt1, pt2 = crop_from_dets(im_to_torch(frame), output[:,1:5])
                    # for i in range(len(inps)):
                        # inp_img = np.array(inps[i]).transpose(1,2,0)
                        # cv2.imshow('img0',inp_img.astype(np.uint8))
                        # cv2.waitKey(0)
                    inp_img = np.array(inps[0]).transpose(1,2,0)
                    heat, pred_cls = sess.run([output_heat, output_cls], feed_dict={input_image: [inp_img]}) #(1, 48, 48, 14) #20ms   
                    tf_out = heat.transpose(0,3,1,2)
                    tf_out = torch.from_numpy(tf_out)

                    kp_box, kp_img, kp_scores = getPrediction(
                    tf_out, pt1[0].unsqueeze(0), pt2[0].unsqueeze(0), poseResH, poseResW, poseResH/4, poseResW/4)

                    result=[{
                        'keypoints': kp_img[0],
                        'kp_score': kp_scores[0],
                        'proposal_score': box_scores[0],
                        'bbox' : output[0,1:5],
                    }]
                    vis_img = vis_frame_fast(frame, result)
                    cv2.imshow("vis", vis_img)
                    key = cv2.waitKey(1)
                    # classes = load_classes('data/coco.names')
                    # colors = pkl.load(open("pallete", "rb"))
                    
                    # list(map(lambda x: write(x, orig_im), output))
                    
                    
                    # cv2.imshow("frame", orig_im)
                    # key = cv2.waitKey(1)
                    # if key & 0xFF == ord('q'):
                    #     break
                    # frames += 1
                    print("FPS of the video is {:5.2f}".format( 1 / (time.time() - start)))
                else:
                    break
            except KeyboardInterrupt:
                break
        

    
    

