import os
import argparse
import yaml
import cv2
import numpy as np


def gen_analyzed_image(data_path, file_path):
    # load doc
    doc = yaml.load(file_path)

    # load image
    image_path = os.path.join(data_path, doc.dataset, doc.path)
    image = cv2.imread(image_path)

    # get all objects
    for o in doc.objects:
        # segmentation
        if True:
            seg2d = o.segmentation2d
            color = tuple(np.random.randint(50, 255, size=3).tolist())
            indexes = seg2d == True
            image[indexes] = color

        # box
        if True:
            box = o.box
            color = tuple(np.random.randint(50, 255, size=3).tolist())
            lt = (box[0], box[1])
            rb = (box[0] + box[2], box[1] + box[3])
            cv2.rectangle(image, lt, rb, color)

    return image







if __name__ == "__main__":

    def str2bool(v):return v.lower() in ("yes", "true", "t", "1", True)
    parser = argparse.ArgumentParser(prog="python", description="transfer cv dataset")
    parser.add_argument('--data-path', type=str, help='dataset path')
    parser.add_argument('--file-path', type=str, help='label file path')
    parser.add_argument('--output-path', type=str, default=os.curdir, help='root path for all dataset')

    args = parser.parse_args()

    image = gen_analyzed_image(args.data_path, args.file_path)
    cv2.imwrite()


