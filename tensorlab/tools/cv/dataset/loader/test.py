import numpy as np
import os
from PIL import Image


def check(filename):
    cord_info, min_mask=[], []
    seg = Image.open(os.path.join('/datasets/VOC/VOCdevkit/VOC2007/SegmentationObject', '%s.png' % filename))
    seg_copy = np.array(seg, dtype=np.uint8)
    seg_copy[seg_copy[:] == 0] = 255
    for i in range(3):
        _min = np.min(seg_copy)
        cord_x, cord_y = np.where(seg_copy[:] == _min)
        min_cord_y, max_cord_y, min_cord_x, max_cord_x = np.min(cord_y), np.max(cord_y), np.min(cord_x), np.max(cord_x)
        cord_info.append([min_cord_y, max_cord_y, min_cord_x, max_cord_x])
        seg_copy[seg_copy[:] == _min] = 255
        min_mask.append(_min)
    print(cord_info)
    return



if __name__ == "__main__":
    check('002403')






[[316, 499, 114, 155], [373, 404, 184, 220], [282, 389, 35, 166]]
