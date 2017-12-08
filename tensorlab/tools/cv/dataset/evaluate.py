import numpy as np
from PIL import Image
import os
import argparse


def evaluation(args):
    verify = False
    p_filenames = os.listdir(args.pred_path)
    gt_filenames = os.listdir(args.gt_path)

    if len(p_filenames) != len(gt_filenames):
        lens = len(gt_filenames) - len(p_filenames)
        print('about %d segmentation images not in groundtruth' % lens)
    else:
        print('all output segmentation picture in groundtrth')

    p_files, gt_files = [], []
    for i in p_filenames:
        if os.path.splitext(os.path.split(i)[-1])[-1] == '.png':
            p = os.path.splitext(os.path.split(i)[-1])[0]
            p_files.append(p)
    for j in gt_filenames:
        if os.path.splitext(os.path.split(j)[-1])[-1] == '.png':
            gt = os.path.splitext(os.path.split(j)[-1])[0]
            gt_files.append(gt)

    for i in range(len(p_files)):
        if p_files[i] in gt_files:
            pred = Image.open(os.path.join(args.pred_path, '%s.png' % p_files[i]))
            gt = Image.open(os.path.join(args.gt_path, '%s.png' % p_files[i]))

            #covert pred picture
            pred_piexl = np.array(pred, dtype=np.uint8)
            # pred_piexl[pred_piexl[:] == 255] = 0
            pred_piexl[pred_piexl[:] != 0] = 1

            #covert gt picture
            label_piexl = np.array(gt, dtype=np.uint8)
            label_piexl[label_piexl[:] == 255] = 0
            label_piexl[label_piexl[:] != 0] = 1

            #compare
            compare = pred_piexl - label_piexl
            if np.min(compare) == 0 and np.max(compare) == 0:
                verify = True
            else:
                print("pred image was not match groundtruth, pred abspath is %s" % os.path.join(args.pred_path, '%s.png' % p_files[i]))
        else:
            print('wrong data path')

    return verify


if __name__ == "__main__":

    def str2bool(v):return v.lower() in ("yes", "true", "t", "1", True)
    parser = argparse.ArgumentParser(prog="python", description="verify cv dataset")
    parser.add_argument('--pred-path', type=str, help='previous level directory of pred abs path')
    parser.add_argument('--gt-path', type=str, default=None, help='previous level directory of groundtruth abs path')

    args = parser.parse_args()

    evaluation(args)

