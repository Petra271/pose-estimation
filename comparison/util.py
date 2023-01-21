import json
import pylab
import skimage.io as io
import matplotlib.pyplot as plt
import sys
import os

def plot_img_kpts_coco(coco, anns, img, cat_ids, figsize=(8.0, 10.0), 
                      iscrowd=None, title="", fontsize=26, segm=True):
  #pylab.rcParams['figure.figsize'] = figsize
  I = io.imread(img['coco_url'])
  plt.imshow(I); 
  plt.axis('off')
  plt.title(title, fontsize=fontsize)
  if not segm and "segmentation" in anns[0]:
    [det.pop("segmentation") for det in anns]
  coco.showAnns(anns)


def get_kpts(res_file, image_id, num):
  with open(res_file, "r") as f:
    res = json.load(f)
    data = [ det for det in res if det.get("image_id") == image_id]
  return data


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        