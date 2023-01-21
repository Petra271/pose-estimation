# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import numpy as np
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as T
import sys 
import skimage.io as io

sys.path.insert(1, '/content/drive/MyDrive/pose-estimation-research/transpose')
sys.path.insert(1, '/content/drive/MyDrive/pose-estimation-research/transpose/lib')

from utils import transforms, vis
from core.inference import get_final_preds
from visualize import pose
import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import validate
from utils.utils import create_logger

import dataset
import models

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid')


    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    model = model.to("cuda")
    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        ckpt_state_dict = torch.load(cfg.TEST.MODEL_FILE)
        model.load_state_dict(ckpt_state_dict, strict=True)   
    else:
        model_state_file = os.path.join(
            final_output_dir, 'final_state.pth'
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))
    w, h = cfg.MODEL.IMAGE_SIZE

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    # Data loading code
    normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        T.Compose([
            T.ToTensor(),
            normalize,
        ])
    )

    #print(valid_dataset[0][0])
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True
    )
    with torch.no_grad():
      img = next((i for i, item in enumerate(valid_dataset.db) if item["imgnum"] == 50638), None)
      img = valid_dataset[img][0]
      print(img)
      device = "cuda"
      inputs = torch.cat([img.to(device)]).unsqueeze(0)
      outputs = model(inputs)
      if isinstance(outputs, list):
          output = outputs[-1]
      else:
          output = outputs

      if cfg.TEST.FLIP_TEST: 
          input_flipped = np.flip(inputs.cpu().numpy(), 3).copy()
          input_flipped = torch.from_numpy(input_flipped).cuda()
          outputs_flipped = model(input_flipped)

          if isinstance(outputs_flipped, list):
              output_flipped = outputs_flipped[-1]
          else:
              output_flipped = outputs_flipped

          output_flipped = transforms.flip_back(output_flipped.cpu().numpy(),
                                    valid_dataset.flip_pairs)
          output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

          output = (output + output_flipped) * 0.5
          
      preds, maxvals = get_final_preds(
              cfg, output.clone().cpu().numpy(), None, None, transform_back=False)

      query_locations = np.array([p*4+0.5 for p in preds[0]])

  
    pose(img, model, query_locations, model_name="transposer", mode='dependency', save_img=True, threshold=0.0)


if __name__ == '__main__':
    main()