# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pandas as pd

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
from tqdm import tqdm
from scipy.misc import imread
from matplotlib import pyplot as plt
import pdb
import socket
from datetime import datetime



xrange = range  # Python 3


def save_weight(weight, time, seen):
    time = np.where(time == 0, 1, time)
    weight = weight / time[:, np.newaxis]
    result_map = np.zeros((len(weight), len(weight)))
    for i in range(len(weight)):
        for j in range(len(weight)):
            v1 = weight[i]
            v2 = weight[j]
            # v1_ = np.linalg.norm(v1)
            # v2_ = np.linalg.norm(v2)
            # v12 = np.sum(v1*v2)
            # print(v12)
            # print(v1_)
            # print(v2_)
            distance = np.linalg.norm(v1 - v2)
            if np.sum(v1 * v2) == 0:
                result_map[i][j] = 0
            else:
                result_map[i][j] = distance

    df = pd.DataFrame(result_map)

    ## save to xlsx file

    filepath = 'similarity_%d.xlsx' % (seen)

    df.to_excel(filepath, index=False)

    weight = weight * 255

    cv2.imwrite('./weight_%d.png' % (seen), weight)

def prep_im_for_blob(im, target_size):
    """Mean subtract and scale an image for use in a blob."""

    im = im.astype(np.float32, copy=False)
    # changed to use pytorch models
    im /= 255.  # Convert range to [0,1]
    # normalization for pytroch pretrained models.
    # https://pytorch.org/docs/stable/torchvision/models.html
    pixel_means = [0.485, 0.456, 0.406]
    pixel_stdens = [0.229, 0.224, 0.225]

    # normalize manual
    im -= pixel_means  # Minus mean
    im /= pixel_stdens  # divide by stddev

    # im = im[:, :, ::-1]
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    # if np.round(im_scale * im_size_max) > max_size:
    #     im_scale = float(max_size) / float(im_size_max)
    # im = imresize(im, im_scale)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)

    return im, im_scale

def crop(image, purpose, size):


    cut_image = image[int(purpose[1]):int(purpose[3]),int(purpose[0]):int(purpose[2]),:]


    height, width = cut_image.shape[0:2]

    max_hw   = max(height, width)
    cty, ctx = [height // 2, width // 2]

    cropped_image  = np.zeros((max_hw, max_hw, 3), dtype=cut_image.dtype)

    x0, x1 = max(0, ctx - max_hw // 2), min(ctx + max_hw // 2, width)
    y0, y1 = max(0, cty - max_hw // 2), min(cty + max_hw // 2, height)

    left, right = ctx - x0, x1 - ctx
    top, bottom = cty - y0, y1 - cty

    cropped_cty, cropped_ctx = max_hw // 2, max_hw // 2
    y_slice = slice(cropped_cty - top, cropped_cty + bottom)
    x_slice = slice(cropped_ctx - left, cropped_ctx + right)
    cropped_image[y_slice, x_slice, :] = cut_image[y0:y1, x0:x1, :]


    return cv2.resize(cropped_image, (size,size), interpolation=cv2.INTER_LINEAR)

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='coco', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfgs/res101.yml', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res50', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models', default="models",
                        type=str)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        default=True)
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        default=True)
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        default=True)
    parser.add_argument('--parallel_type', dest='parallel_type',
                        help='which part of model to parallel, 0: all, 1: model before roi pooling',
                        default=0, type=int)
    parser.add_argument('--s', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=10, type=int)
    parser.add_argument('--p', dest='checkpoint',
                        help='checkpoint to load network',
                        default=2995, type=int)
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        action='store_true')
    parser.add_argument('--seen', dest='seen', default=2, type=int)
    parser.add_argument('--a', dest='average', default=1, type=int)
    parser.add_argument('--g', dest='group',
                        help='which group',
                        default=0)
    args = parser.parse_args()
    return args


lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY
thresh = 0.05
max_per_image = 100

def get_ready(query_img_path):
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    np.random.seed(cfg.RNG_SEED)
    if args.dataset == "coco":
        args.imdb_name = "coco_2017_train"
        args.imdbval_name = "coco_2017_val"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']


    # args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)
    args.cfg_file = "cfgs/{}_{}.yml".format(args.net, args.group) if args.group != 0 else "cfgs/{}.yml".format(args.net)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    cfg.TRAIN.USE_FLIPPED = False
    # imdb_vs, roidb_vs, ratio_list_vs, ratio_index_vs, query_vs = combined_roidb('coco_2014_valminusminival', False)
    imdb_vu, roidb_vu, ratio_list_vu, ratio_index_vu, query_vu = combined_roidb(args.imdbval_name, False,
                                                                                seen=args.seen)
    # imdb_vs.competition_mode(on=True)
    imdb_vu.competition_mode(on=True)

    input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)
    load_name = os.path.join(input_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

    # initilize the network here.
    if args.net == 'res50':
        fasterRCNN = resnet(imdb_vu.classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']

    print('load model successfully!')
    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    query = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    catgory = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)





    if args.cuda:
        cfg.CUDA = True
        fasterRCNN.cuda()



    output_dir_vu = get_output_dir(imdb_vu, 'faster_rcnn_unseen')

    dataset_vu = roibatchLoader(roidb_vu, ratio_list_vu, ratio_index_vu, query_vu, 1, imdb_vu.num_classes,
                                training=False, seen=args.seen)
    fasterRCNN.eval()


    avg = 0
    dataset_vu.query_position = avg



    num_images_vu = len(imdb_vu.image_index)


    all_boxes = [[[] for _ in xrange(num_images_vu)]
                 for _ in xrange(imdb_vu.num_classes)]

    _t = {'im_detect': time.time(), 'misc': time.time()}
    det_file = os.path.join(output_dir_vu, 'detections_%d_%d.pkl' % (args.seen, avg))
    print(det_file)

    # make query data
    query_im = imread(query_img_path)
    query_im = cv2.resize(query_im, dsize=(640, 480), interpolation=cv2.INTER_LINEAR)
    _query_im = np.copy(query_im)
    query_im, query_im_scale = prep_im_for_blob(query_im, target_size=128)
    query_im = torch.tensor(query_im)
    query_im = torch.unsqueeze(query_im, 0)
    query_im = query_im.transpose(1, 3)
    query = query_im.transpose(2, 3)
    query  = query.cuda()

    return fasterRCNN, all_boxes, query, _query_im

def main(cv2_img, fasterRCNN, all_boxes, query, _query_im):

    index = 0
    data = [0, 0, 0, 0, 0]
    im = cv2_img
    im = cv2.resize(im, dsize=(640, 480), interpolation=cv2.INTER_LINEAR)

    _im = np.copy(im)

    # make im_data
    im, im_scale = prep_im_for_blob(im, target_size=600)
    im = torch.tensor(im)
    im = torch.unsqueeze(im, 0)
    im = im.transpose(1, 3)
    im_data = im.transpose(2, 3)

    im_data = data[0] = im_data.cuda()
    im_info = data[2] = torch.tensor([[600, 899, 1.4052]])
    gt_boxes = data[3] = torch.rand(1, 4, 5) # don't care
    catgory = data[4] = torch.tensor([1])



    rois, cls_prob, bbox_pred, \
    rpn_loss_cls, rpn_loss_box, \
    RCNN_loss_cls, _, RCNN_loss_bbox, \
    rois_label, weight = fasterRCNN(im_data, query, im_info, gt_boxes, catgory)


    scores = cls_prob.data
    boxes = rois.data[:, :, 1:5]

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.data
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                         + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
            box_deltas = box_deltas.view(1, -1, 4)

        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)

    pred_boxes /= data[2][0][2].item()

    scores = scores.squeeze()
    pred_boxes = pred_boxes.squeeze()

    im2show = np.copy(_im)

    inds = torch.nonzero(scores > thresh).view(-1)
    # if there is det


    if inds.numel() > 0:
        cls_scores = scores[inds]
        _, order = torch.sort(cls_scores, 0, True)
        #order = order[0]
        cls_boxes = pred_boxes[inds, :]

        cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
        # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
        cls_dets = cls_dets[order]
        keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
        cls_dets = cls_dets[keep.view(-1).long()]
        all_boxes[data[4]][index] = cls_dets.cpu().numpy()

        im2show = vis_detections(im2show, 'shot', cls_dets.cpu().numpy(), 0.8)
        im2show = cv2.cvtColor(im2show, cv2.COLOR_BGR2RGB)
        _query_im = cv2.cvtColor(_query_im, cv2.COLOR_BGR2RGB)
        _im2show = np.concatenate((im2show, _query_im), axis=1)

        cv2.imwrite(img_dir + "/" + str(img_count) + ".png", _im2show)

        plt.imshow(_im2show)
        plt.show()

        img_scores.loc[img_scores.shape[0]] = [img_count, cls_scores[order][0]]

    # # Limit to max_per_image detections *over all classes*
    # if max_per_image > 0:
    #     try:
    #         image_scores = all_boxes[data[4]][index][:, -1]
    #         if len(image_scores) > max_per_image:
    #             image_thresh = np.sort(image_scores)[-max_per_image]
    #
    #             keep = np.where(all_boxes[data[4]][index][:, -1] >= image_thresh)[0]
    #             all_boxes[data[4]][index] = all_boxes[data[4]][index][keep, :]
    #     except:
    #         pass
    #
    #
    # o_query = query[0].permute(1, 2, 0).contiguous().cpu().numpy()
    # o_query *= [0.229, 0.224, 0.225]
    # o_query += [0.485, 0.456, 0.406]
    # o_query *= 255
    # o_query = o_query[:, :, ::-1]
    #
    # (h, w, c) = im2show.shape
    # o_query = cv2.resize(o_query, (h, h), interpolation=cv2.INTER_LINEAR)
    # o_query = cv2.cvtColor(o_query, cv2.COLOR_BGR2RGB)
    #
    # im2show = np.concatenate((im2show, o_query), axis=1)
    # im2show = cv2.cvtColor(im2show, cv2.COLOR_BGR2RGB)
    # plt.imshow(im2show)
    # plt.show()

    # cv2.imwrite('./test_img/%d.png' % (i), im2show)

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

if __name__ == '__main__':
    args = parse_args()
    TCP_IP = '192.168.50.201'
    TCP_PORT = 5001
    print('listen', TCP_IP, TCP_PORT)

    # TCP소켓 열고 수신 대기
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((TCP_IP, TCP_PORT))
    s.listen(True)
    conn, addr = s.accept()

    # todo need to change query object before run code
    query_object = "doll"
    query_img_path = 'test/query/' + query_object + '.png'

    now = datetime.now()
    date_time = now.strftime("%m%d")
    img_dir = "/home/aupair/Desktop/One-Shot-Object-Detection/test/output/" + query_object + str(date_time)
    img_dir_scene = "/home/aupair/Desktop/One-Shot-Object-Detection/test/output/" + query_object + str(date_time) + "/scene"
    os.mkdir(img_dir)
    os.mkdir(img_dir_scene)

    img_count = 0

    fasterRCNN, all_boxes, query, _query_im = get_ready(query_img_path)

    img_scores = pd.DataFrame({"img_num": [0], "cls_score": [0]})

    while True:
        try:
            img_count += 1
            # String형의 이미지를 수신받아서 이미지로 변환 하고 화면에 출력
            length = recvall(conn, 16)  # 길이 16의 데이터를 먼저 수신하는 것은 여기에 이미지의 길이를 먼저 받아서 이미지를 받을 때 편리하려고 하는 것이다.
            stringData = recvall(conn, int(length))
            data = np.fromstring(stringData, dtype='uint8')
            if len(data) == 333:
                img_scores_data = pd.DataFrame(img_scores)
                img_scores_data.to_csv(img_dir + "/" + "score_info" + ".csv")
                continue
            decimg = cv2.imdecode(data, 1)
            decimg_scene = cv2.cvtColor(decimg, cv2.COLOR_BGR2RGB)
            cv2.imwrite(img_dir_scene + "/" + str(img_count) + ".png", decimg_scene)
            main(decimg, fasterRCNN, all_boxes, query, _query_im)
        except:
            s.close()

            exit()



