import numpy as np
import os
from PIL import Image


def check(filename):
    cord_info, min_mask=[], []
    seg = Image.open(os.path.join('/datasets/Labels/VOC/VOCdevkit/VOC2007/', '%s.png' % filename))
    seg_copy = np.array(seg, dtype=np.uint8)
    seg_copy[seg_copy[:] == 0] = 255
    for i in range(2):
        _min = np.min(seg_copy)
        cord_x, cord_y = np.where(seg_copy[:] == _min)
        min_cord_y, max_cord_y, min_cord_x, max_cord_x = np.min(cord_y), np.max(cord_y), np.min(cord_x), np.max(cord_x)
        cord_info.append([min_cord_y, max_cord_y, min_cord_x, max_cord_x])
        seg_copy[seg_copy[:] == _min] = 255
        min_mask.append(_min)
    print(cord_info)
    return



if __name__ == "__main__":
    check('000032')






[[316, 499, 114, 155], [373, 404, 184, 220], [282, 389, 35, 166]]
'pred image was not match groundtruth, pred abspath is /datasets/Labels/VOC/VOCdevkit/VOC2012/JPEGImages/2008_005245.png'
'pred image was not match groundtruth, pred abspath is /datasets/Labels/VOC/VOCdevkit/VOC2012/JPEGImages/2009_000455.png'
'pred image was not match groundtruth, pred abspath is /datasets/Labels/VOC/VOCdevkit/VOC2012/JPEGImages/2009_004969.png'
'pred image was not match groundtruth, pred abspath is /datasets/Labels/VOC/VOCdevkit/VOC2012/JPEGImages/2009_005069.png'
'pred image was not match groundtruth, pred abspath is /datasets/Labels/VOC/VOCdevkit/VOC2012/JPEGImages/2011_002644.png'