import os
import select
import sys
import argparse
import yaml
import cv2
import numpy as np


def press_key_stop(message = None):
    if message is not None: print(message)
    while True:
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            c = sys.stdin.read(1)
            break
        else:
            cv2.waitKey(10)


def gen_analyzed_image(data_path, doc):
    # load image
    image_path = os.path.join(data_path, doc.dataset, doc.path)
    image = cv2.imread(image_path)

    # get all objects
    for o in doc.objects:
        # segmentation
        if o.segmentation2d is not None:
            seg2d = o.segmentation2d
            color = tuple(np.random.randint(50, 255, size=3).tolist())
            indexes = seg2d == True
            image[indexes] = color

        # box
        if o.box is not None:
            box = o.box
            color = tuple(np.random.randint(50, 255, size=3).tolist())
            lt = (box[0], box[1])
            rb = (box[0] + box[2], box[1] + box[3])
            cv2.rectangle(image, lt, rb, color)

    return image



def show_analyzed_image(data_path, file_path):
    # load doc
    doc = yaml.load(file_path)

    # gen image
    image = gen_analyzed_image(data_path, doc)

    # show
    cv2.imshow(doc.path, image)
    press_key_stop('Press key to stop')






if __name__ == "__main__":

    def str2bool(v):return v.lower() in ("yes", "true", "t", "1", True)
    parser = argparse.ArgumentParser(prog="python", description="transfer cv dataset")
    parser.add_argument('--data-path', type=str, help='dataset path')
    parser.add_argument('--file-path', type=str, help='label file path')
    parser.add_argument('--output-path', type=str, default=os.curdir, help='root path for all dataset')

    args = parser.parse_args()

    # analyze
    show_analyzed_image(args.data_path, args.file_path)
