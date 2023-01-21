# ------------------------------------------------------------------------------
# Copyright (c) Southeast University. Licensed under the MIT License.
# Written by Sen Yang (yangsenius@seu.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os.path as osp
import torch

import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
import math
import numpy as np
import cv2

keypoint_name = {
    0: "nose",
    1: "eye(l)",
    2: "eye(r)",
    3: "ear(l)",
    4: "ear(r)",
    5: "sho.(l)",
    6: "sho.(r)",
    7: "elb.(l)",
    8: "elb.(r)",
    9: "wri.(l)",
    10: "wri.(r)",
    11: "hip(l)",
    12: "hip(r)",
    13: "kne.(l)",
    14: "kne.(r)",
    15: "ank.(l)",
    16: "ank.(r)",
    17: "random",
    18: "random",
}


class plt_config:
    def __init__(self, dataset_name):
        assert dataset_name == "coco", "{} dataset is not supported".format(
            dataset_name
        )
        self.n_kpt = 17
        # edge , color
        self.EDGES = [
            ([15, 13], [255, 0, 0]),  # l_ankle -> l_knee
            ([13, 11], [155, 85, 0]),  # l_knee -> l_hip
            ([11, 5], [155, 85, 0]),  # l_hip -> l_shoulder
            ([12, 14], [0, 0, 255]),  # r_hip -> r_knee
            ([14, 16], [17, 25, 10]),  # r_knee -> r_ankle
            ([12, 6], [0, 0, 255]),  # r_hip  -> r_shoulder
            ([3, 1], [0, 255, 0]),  # l_ear -> l_eye
            ([1, 2], [0, 255, 5]),  # l_eye -> r_eye
            ([1, 0], [0, 255, 170]),  # l_eye -> nose
            ([0, 2], [0, 255, 25]),  # nose -> r_eye
            ([2, 4], [0, 17, 255]),  # r_eye -> r_ear
            ([9, 7], [0, 220, 0]),  # l_wrist -> l_elbow
            ([7, 5], [0, 220, 0]),  # l_elbow -> l_shoulder
            ([5, 6], [125, 125, 155]),  # l_shoulder -> r_shoulder
            ([6, 8], [25, 0, 55]),  # r_shoulder -> r_elbow
            ([8, 10], [25, 0, 255]),
        ]  # r_elbow -> r_wrist


def plot_poses(
    img, skeletons, config=plt_config("coco"), save_path=None, dataset_name="coco"
):

    cmap = matplotlib.cm.get_cmap("hsv")
    canvas = img.copy()
    n_kpt = config.n_kpt
    for i in range(n_kpt):
        rgba = np.array(cmap(1 - i / n_kpt - 1.0 / n_kpt * 2))
        rgba[0:3] *= 255
        for j in range(len(skeletons)):
            if len(skeletons[j][i]) > 2 and skeletons[j][i, 2] > 0:
                cv2.circle(
                    canvas,
                    tuple(skeletons[j][i, 0:2].astype("int32")),
                    3,
                    (255, 255, 255),
                    thickness=-1,
                )

    stickwidth = 2
    for i in range(len(config.EDGES)):
        for j in range(len(skeletons)):
            edge = config.EDGES[i][0]
            color = config.EDGES[i][1]
            if len(skeletons[j][edge[0]]) > 2:
                if skeletons[j][edge[0], 2] == 0 or skeletons[j][edge[1], 2] == 0:
                    continue

            cur_canvas = canvas.copy()
            X = [skeletons[j][edge[0], 1], skeletons[j][edge[1], 1]]
            Y = [skeletons[j][edge[0], 0], skeletons[j][edge[1], 0]]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly(
                (int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1
            )
            cv2.fillConvexPoly(cur_canvas, polygon, color)
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

    return canvas


def update_config(cfg, yamlfilename):
    cfg.defrost()
    cfg.merge_from_file(yamlfilename)
    cfg.TEST.MODEL_FILE = osp.join(cfg.DATA_DIR, cfg.TEST.MODEL_FILE)
    cfg.freeze()


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


def pose(
    image,
    model,
    query_locations,
    model_name="transposer",
    mode="dependency",
    threshold=None,
    device=torch.device("cuda"),
    kpt_color="white",
    img_name="image",
    save_img=False,
):

    assert mode in ["dependency", "affect"]
    inputs = torch.cat([image.to(device)]).unsqueeze(0)
    features = []
    global_enc_atten_maps = []
    std = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(-1).unsqueeze(-1)
    mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(-1).unsqueeze(-1)
    img_vis = image * std + mean
    img_vis = img_vis.permute(1, 2, 0).detach().cpu().numpy()
    img_vis_kpts = img_vis.copy()
    img_vis_kpts = plot_poses(img_vis_kpts, [query_locations])
    plt.axis('off')
    plt.imshow(img_vis_kpts)
     

    if save_img:
        plt.savefig("attention_map_{}_{}_{}.jpg".format(img_name, mode, model_name))
    plt.show()
