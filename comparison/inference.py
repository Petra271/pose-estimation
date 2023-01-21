from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import sys
import argparse
import numpy as np

def evaluate_mAP(res_file, ann_type='keypoints', ann_file='./data/coco/annotations/person_keypoints_val2017.json'):
    coco_gt = COCO(ann_file)
    coco_det = coco_gt.loadRes(res_file)

    cocoEval = COCOeval(coco_gt, coco_det, ann_type)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    

    if isinstance(cocoEval.stats[0], dict):
        stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)',
                       'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']
        parts = ['body', 'foot', 'face', 'hand', 'fullbody']

        info = {}
        for i, part in enumerate(parts):
            info[part] = cocoEval.stats[i][part][0]
        return info
    else:
        stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)',
                       'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']
        info_str = {}
        for ind, name in enumerate(stats_names):
            info_str[name] = cocoEval.stats[ind]
        return info_str['AP']


def calc_dists(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1
    return dists


def main():
  args = parse_args()
  evaluate_mAP(res_file=args.res_file, ann_file=args.ann_file)

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("res_file", type=str)
  parser.add_argument("--ann_file", type=str, 
  default="/content/drive/MyDrive/data/coco/annotations/person_keypoints_val2017.json")
  parser.add_argument("--ann_type", choices=("bbox", "segm", "keypoints"), default="keypoints")
  return parser.parse_args()

  
if __name__ == "__main__":
  main()

