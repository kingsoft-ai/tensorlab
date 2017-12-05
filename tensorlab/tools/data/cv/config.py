from easydict import EasyDict as edict
from tensorlab.tools.data.cv.loader import VOCLoder, COCOLoder ,ADE20KLoader

DATASETS = edict(

    VOC = edict(
        name = 'VOC',
        loader = VOCLoder,
        year = ['07','12'],
        split = ['train','val','test']

    ),

    ADE=edict(
        name='ADE',
        loader=ADE20KLoader,
        split=['train', 'validation']

    ),

    COCO = edict(
        name = 'COCO',
        loader = COCOLoder,
        split = [['train2014', 'val2014'],['test2014', 'test2015']]  #separate training and testing datasets, using 2-d matrix , 0:training & valadation  1:testing
    )
)




IMAGE_FILE_EXTENSIONS = ['.jpg', '.jpeg', '.png']

DEFAULT_LABEL_PATH = 'Labels'


CLASSIFICATION = [
    '__background__',
    'airplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'dining table',
    'dog',
    'horse',
    'motorcycle',
    'person',
    'potted plant',
    'sheep',
    'couch',
    'train',
    'tv',
    'apple',
    'backpack',
    'banana',
    'baseball bat',
    'baseball glove',
    'bear',
    'bed',
    'bench',
    'book',
    'bowl',
    'broccoli',
    'cake',
    'carrot',
    'cell phone',
    'clock',
    'cup',
    'donut',
    'elephant',
    'fire hydrant',
    'fork',
    'frisbee',
    'giraffe',
    'hair drier',
    'handbag',
    'hot dog',
    'keyboard',
    'kite',
    'knife',
    'laptop',
    'microwave',
    'mouse',
    'orange',
    'oven',
    'parking meter',
    'pizza',
    'refrigerator',
    'remote',
    'sandwich',
    'scissors',
    'sink',
    'skateboard',
    'skis',
    'snowboard',
    'spoon',
    'sports ball',
    'stop sign',
    'suitcase',
    'surfboard',
    'teddy bear',
    'tennis racket',
    'tie',
    'toaster',
    'toilet',
    'toothbrush',
    'traffic light',
    'truck',
    'umbrella',
    'vase',
    'wine glass',
    'zebra'
]