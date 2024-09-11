from yolact import Yolact
from utils.augmentations import BaseTransform, FastBaseTransform, Resize

from utils import timer
from utils.functions import SavePath
from layers.output_utils import postprocess, undo_image_transformation


from data import cfg, set_cfg, set_dataset

from timeit import default_timer as timer

import torch.backends.cudnn as cudnn
import argparse
import random

import os
import cv2


from collections import defaultdict
import numpy as np

import networkx as nx
import matplotlib.pyplot as plt
from itertools import product
import math
import csv
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import os.path as osp

import numpy as np
# Import model configuration
from gin import GIN
from gcn import GCN
from ginlaf import LAFNet
from COCOParser import COCOParser


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='YOLACT COCO Evaluation')
    parser.add_argument('--trained_model',
                        default='weights/ssd300_mAP_77.43_v2.pth', type=str,
                        help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
    parser.add_argument('--top_k', default=5, type=int,
                        help='Further restrict the number of predictions to parse')
    parser.add_argument('--cuda', default=True, type=str2bool,
                        help='Use cuda to evaulate model')
    parser.add_argument('--fast_nms', default=True, type=str2bool,
                        help='Whether to use a faster, but not entirely correct version of NMS.')
    parser.add_argument('--cross_class_nms', default=False, type=str2bool,
                        help='Whether compute NMS cross-class or per-class.')
    parser.add_argument('--display_masks', default=True, type=str2bool,
                        help='Whether or not to display masks over bounding boxes')
    parser.add_argument('--display_bboxes', default=True, type=str2bool,
                        help='Whether or not to display bboxes around masks')
    parser.add_argument('--display_text', default=True, type=str2bool,
                        help='Whether or not to display text (class [score])')
    parser.add_argument('--display_scores', default=True, type=str2bool,
                        help='Whether or not to display scores in addition to classes')
    parser.add_argument('--display', dest='display', action='store_true',
                        help='Display qualitative results instead of quantitative ones.')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true',
                        help='Shuffles the images when displaying them. Doesn\'t have much of an effect when display is off though.')
    parser.add_argument('--ap_data_file', default='results/ap_data.pkl', type=str,
                        help='In quantitative mode, the file to save detections before calculating mAP.')
    parser.add_argument('--resume', dest='resume', action='store_true',
                        help='If display not set, this resumes mAP calculations from the ap_data_file.')
    parser.add_argument('--max_images', default=-1, type=int,
                        help='The maximum number of images from the dataset to consider. Use -1 for all.')
    parser.add_argument('--output_coco_json', dest='output_coco_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this just dumps detections into the coco json file.')
    parser.add_argument('--bbox_det_file', default='results/bbox_detections.json', type=str,
                        help='The output file for coco bbox results if --coco_results is set.')
    parser.add_argument('--mask_det_file', default='results/mask_detections.json', type=str,
                        help='The output file for coco mask results if --coco_results is set.')
    parser.add_argument('--config', default=None,
                        help='The config object to use.')
    parser.add_argument('--output_web_json', dest='output_web_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this dumps detections for usage with the detections viewer web thingy.')
    parser.add_argument('--web_det_path', default='web/dets/', type=str,
                        help='If output_web_json is set, this is the path to dump detections into.')
    parser.add_argument('--no_bar', dest='no_bar', action='store_true',
                        help='Do not output the status bar. This is useful for when piping to a file.')
    parser.add_argument('--display_lincomb', default=False, type=str2bool,
                        help='If the config uses lincomb masks, output a visualization of how those masks are created.')
    parser.add_argument('--benchmark', default=False, dest='benchmark', action='store_true',
                        help='Equivalent to running display mode but without displaying an image.')
    parser.add_argument('--no_sort', default=False, dest='no_sort', action='store_true',
                        help='Do not sort images by hashed image ID.')
    parser.add_argument('--seed', default=None, type=int,
                        help='The seed to pass into random.seed. Note: this is only really for the shuffle and does not (I think) affect cuda stuff.')
    parser.add_argument('--mask_proto_debug', default=False, dest='mask_proto_debug', action='store_true',
                        help='Outputs stuff for scripts/compute_mask.py.')
    parser.add_argument('--no_crop', default=False, dest='crop', action='store_false',
                        help='Do not crop output masks with the predicted bounding box.')
    parser.add_argument('--image', default=None, type=str,
                        help='A path to an image to use for display.')
    parser.add_argument('--image_dir', default=None, type=str,
                        help='A path to an image to use for display.')
    parser.add_argument('--name', default=None, type=str,
                        help='A path to an image to use for display.')
    parser.add_argument('--images', default=None, type=str,
                        help='An input folder of images and output folder to save detected images. Should be in the format input->output.')
    parser.add_argument('--video', default=None, type=str,
                        help='A path to a video to evaluate on. Passing in a number will use that index webcam.')
    parser.add_argument('--video_multiframe', default=1, type=int,
                        help='The number of frames to evaluate in parallel to make videos play at higher fps.')
    parser.add_argument('--score_threshold', default=0, type=float,
                        help='Detections with a score under this threshold will not be considered. This currently only works in display mode.')
    parser.add_argument('--dataset', default="/content/test.json", type=str,
                        help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
    parser.add_argument('--dir', default="./content", type=str,
                        help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
    parser.add_argument('--coco_images_dir', default="/content/images", type=str,
                        help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
    parser.add_argument('--gcn_model', default=None, type=str,
                        help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
    parser.add_argument('--model', default="GCN", type=str,
                        help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
    parser.add_argument('--file_res', default="/content/", type=str,
                        help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
    parser.add_argument('--GCN', default=True, type=str2bool,
                        help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
    parser.add_argument('--detect', default=False, dest='detect', action='store_true',
                        help='Don\'t evauluate the mask branch at all and only do object detection. This only works for --display and --benchmark.')
    parser.add_argument('--display_fps', default=False, dest='display_fps', action='store_true',
                        help='When displaying / saving video, draw the FPS on the frame')
    parser.add_argument('--nb_label', default=1, type=int,
                        help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
    parser.add_argument('--beta', default=0.1, type=float,
                        help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
    parser.add_argument('--emulate_playback', default=False, dest='emulate_playback', action='store_true',
                        help='When saving a video, emulate the framerate that you\'d get running in real-time mode.')
    parser.add_argument('--display_scene', default=True, type=str2bool,
                        help='Whether or not to display score in addition to classes')
    parser.add_argument('--hidden', default=1024, type=int,
                        help='Define the hidden layer size')

    parser.set_defaults(no_bar=False, display=False, resume=False, output_coco_json=False, output_web_json=False,
                        shuffle=False,
                        benchmark=False, no_sort=False, no_hash=False, mask_proto_debug=False, crop=True, detect=False,
                        display_fps=False,
                        emulate_playback=False)

    global args
    args = parser.parse_args(argv)

    if args.output_web_json:
        args.output_coco_json = True

    if args.seed is not None:
        random.seed(args.seed)


iou_thresholds = [x / 100 for x in range(50, 100, 5)]
coco_cats = {}  # Call prep_coco_cats to fill this
coco_cats_inv = {}
color_cache = defaultdict(lambda: {})



def sort_res_scene(dets_out, gcn, img, h, w ,nb_label,beta, undo_transform=True):
    """
    Note: If undo_transform=False then im_h and im_w are allowed to be None.
    """
    if undo_transform:
        img_numpy = undo_image_transformation(img, w, h)
        img_gpu = torch.Tensor(img_numpy).cuda()
    else:
        img_gpu = img / 255.0
        h, w, _ = img.shape

    # with timer.env('Postprocess'):
    save = cfg.rescore_bbox
    cfg.rescore_bbox = True
    t = postprocess(dets_out, w, h, visualize_lincomb=args.display_lincomb,
                    crop_masks=args.crop,
                    score_threshold=args.score_threshold)
    cfg.rescore_bbox = save

    # with timer.env('Copy'):
    idx = t[1].argsort(0, descending=True)[:args.top_k]

    if cfg.eval_mask_branch:
        # Masks are drawn on the GPU, so don't copy
        masks = t[3][idx]
    classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]

    num_dets_to_consider = min(args.top_k, classes.shape[0])
    for j in range(num_dets_to_consider):
        if scores[j] < args.score_threshold:
            num_dets_to_consider = j
            break

    if num_dets_to_consider == 0:
        # print("ERROR")
        res = [-1, 0]
        return res
    else:
        # print("DETECTED OBJECT: {}".format(num_dets_to_consider))
        position_ = []
        class_ = []
        information = []
        # Create nodes and edges list
        nodes = []
        edges = []
        edges_ = []
        sets = []

        n = 0
        for j in reversed(range(num_dets_to_consider)):
            x1, y1, x2, y2 = boxes[j, :]
            score = scores[j]
            class_id = classes[j]
            # Create Graph
            pos = np.array([[x1, y1]])
            posi = [x1, y1]
            if n == 0:
                position = pos
            else:
                position = np.concatenate((position, pos), axis=0)
            # Compute the object diagonal
            diagonal = float(math.sqrt(x2 * x2 + y2 * y2))
            info = [x1, y1, x2, y2, class_id, diagonal]
            position_.append(posi)
            information.append(info)
            n = n + 1

        length = len(position_)
        min_value = min(position_)
        min_index = position_.index(min_value)
        information = sorted(information)
        if(length > nb_label):
            for i in range(0, length - 1):
                for n in range((i + 1), length):
                    # Computhe distance between objects to define corresponding weight for specific edge
                    distance = (math.sqrt((position[i][0] - position[n][0]) ** 2 + (position[i][1] - position[n][1]) ** 2))
                    edges.append({"source": i, "target": n, "attributes": {"weight": distance}})

                for i in range(0, length):
                    nodes.append({"id": i, "attributes": {"class": float(information[i][4]), "size": information[i][5]}})
                    distances = []
                    if i < length:
                        for n in range((i + 1), length):
                            # Computhe distance between objects to define corresponding weight for specific edge
                            dist = (math.sqrt((information[i][0] - information[n][0]) ** 2 + (
                                        information[i][1] - information[n][1]) ** 2))
                            distances.append(dist)
                    else:
                        dist = 0
                        distance.append(dist)
                    if (len(distances) > 0):
                        min_value = min(distances)
                        min_index = distances.index(min_value)
                    else:
                        min_value = 1000
                        min_index = 0
                    if i < (length - 1):
                        idx = min_index + 1 + i
                        edges_.append({"source": i, "target": idx, "attributes": {"weight": min_value}})
                        for j in range(len(distances)):
                            if ((distances[j] <= (min_value + beta * min_value)) and j != min_index):
                                if j < i:
                                    idx = j
                                else:
                                    idx = j + 1
                                edges_.append({"source": i, "target": idx, "attributes": {"weight": distances[j]}})

            node_attributes = []
            for node in nodes:
                node_attributes.append([node["attributes"][key] for key in node["attributes"]])

            s = []
            t = []
            edge_index = []
            edge_attributes = []

            for edge in edges_:
                source = int(edge["source"])
                target = int(edge["target"])
                s.append(source)
                t.append(target)
                edge_attributes.append([edge["attributes"][key] for key in edge["attributes"]])

            edge_index = [s, t]
            # Tensorised
            node_attributes = torch.Tensor(node_attributes)
            position_ = torch.Tensor(position_)

            edge_index = torch.Tensor(edge_index)
            edge_index = edge_index.to(torch.long)
            batch = torch.zeros([len(node_attributes)], dtype=torch.long)
            num_node_features = 2
            num_edge_features = 1
            num_classes = 2
            start = timer()
            outg = gcn(node_attributes, edge_index, batch)
            res_gcn = outg.argmax(dim=1)
            end = timer()
            time = end-start

            if(res_gcn==0):
                print("INDOOR scene")
                res_="INDOOR"
            else:
                print("OUTDOOR scene")
                res_ = "OUTDOOR"
            # print("Content : {}".format(content))
            res = [res_gcn, time]
            return res
        else:
            # print("DEFAUT")
            res = [-1, 0]
            return res
    return res


def inference(net: Yolact, gcn, image_dir,nb_label, beta):
    net.detect.use_fast_nms = args.fast_nms
    net.detect.use_cross_class_nms = args.cross_class_nms
    cfg.mask_proto_debug = args.mask_proto_debug

    # TODO Currently we do not support Fast Mask Re-scroing in evalimage, evalimages, and evalvideo
    if args.GCN is True:

        # Lister tous les fichiers dans le dossier
        files = os.listdir(image_dir)
        # Filtrer les fichiers pour garder uniquement les images (extensions communes)
        extensions_validees = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        images = [f for f in files if f.lower().endswith(extensions_validees)]
        nb_pred = 1
        cpt_time = 0.0
        results = []
        for im in images:
            # Select images
            image_name = os.path.join(image_dir, im)
            frame = torch.from_numpy(cv2.imread(image_name)).cuda().float()
            batch = FastBaseTransform()(frame.unsqueeze(0))
            preds = net(batch)
            # img, h, w
            h, w, _ = frame.shape
            result = sort_res_scene(preds, gcn, frame,h, w,nb_label, beta, undo_transform=False)
            results.append(result)
            if result[0] != -1:
                nb_pred = nb_pred+1
                cpt_time = cpt_time + result[1]

        av_time = cpt_time/nb_pred
        print('Average time: %5.7f ms' % (av_time))
        return results




if __name__ == '__main__':
    parse_args()

    if args.config is not None:
        set_cfg(args.config)

    if args.config is None:
        model_path = SavePath.from_str(args.trained_model)
        # TODO: Bad practice? Probably want to do a name lookup instead.
        args.config = model_path.model_name + '_config'
        print('Config not specified. Parsed %s from the file name.\n' % args.config)
        set_cfg(args.config)

    if args.display_scene:
        # model_gcn = model.load_state_dict(torch.load('model2_maxpool.pth'))
        state_dict = torch.load(args.gcn_model)
        num_node_features = 2
        num_edge_features = 1
        hidden_channels = 32
        hidden_channels = args.hidden
        num_classes = 2
        # batch_= 64
        if(args.model=='GIN'):
            print("GIN PROCESS")
            hidden_channels = 1024
            model_gcn = GIN(num_node_features,hidden_channels,num_classes)

        if (args.model == 'GCN'):
            print("GCN PROCESS")
            hidden_channels = 1024
            model_gcn = GCN(num_node_features, hidden_channels, num_classes)

        if (args.model == 'GINLAF'):
            print("GINLAF PROCESS")
            hidden_channels = 32
            model_gcn = LAFNet(num_node_features, hidden_channels, num_classes)
        # model_gcn=model_gcn.load_state_dict(state_dict)
        # model_gcn = torch.load('model2_maxpool_.pth')
        model_gcn.load_weights(args.gcn_model)
        print("LOADED GCN MODEL")

    if args.detect:
        cfg.eval_mask_branch = False

    with torch.no_grad():
        if not os.path.exists('results'):
            os.makedirs('results')

        if args.cuda:
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        print('Loading model...', end='')
        net = Yolact()
        net.load_weights(args.trained_model)
        net.eval()
        print(' Done.')

        if args.cuda:
            net = net.cuda()

        if args.display_scene:
            print("EXECUTE Graph Neural Network:")
            criterion = torch.nn.CrossEntropyLoss()
            model_gcn.eval()
            model_gcn = model_gcn.cuda()
            # model, data = model.to(device), data.to(device)
            print("Minimum number of label classes: {} and beta value: {}".format(args.nb_label, args.beta))
            results_ = inference(net, model_gcn, args.image_dir, args.nb_label, args.beta)