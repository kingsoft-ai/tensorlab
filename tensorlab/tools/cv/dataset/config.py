from easydict import EasyDict as edict


VOC = edict(
    name = 'VOC',

    year = '07',
    split = ['train','val','test']

)

COC = edict(
    name = 'COC',
    split = ['train2014', 'val2014', 'test2014', 'test2015', 'minival2014', 'valminusminival2014', 'test-dev2015']
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