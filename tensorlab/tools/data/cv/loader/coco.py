from .loader import Loader
from pycocotools.coco import COCO
from pycocotools import mask
import os
from ..document import Document

coco_ids = {'airplane': 5, 'apple': 53, 'backpack': 27, 'banana': 52,
            'baseball bat': 39, 'baseball glove': 40, 'bear': 23, 'bed': 65,
            'bench': 15, 'bicycle': 2, 'bird': 16, 'boat': 9, 'book': 84,
            'bottle': 44, 'bowl': 51, 'broccoli': 56, 'bus': 6, 'cake': 61,
            'car': 3, 'carrot': 57, 'cat': 17, 'cell phone': 77, 'chair': 62,
            'clock': 85, 'couch': 63, 'cow': 21, 'cup': 47, 'dining table':
            67, 'dog': 18, 'donut': 60, 'elephant': 22, 'fire hydrant': 11,
            'fork': 48, 'frisbee': 34, 'giraffe': 25, 'hair drier': 89,
            'handbag': 31, 'horse': 19, 'hot dog': 58, 'keyboard': 76, 'kite':
            38, 'knife': 49, 'laptop': 73, 'microwave': 78, 'motorcycle': 4,
            'mouse': 74, 'orange': 55, 'oven': 79, 'parking meter': 14,
            'person': 1, 'pizza': 59, 'potted plant': 64, 'refrigerator': 82,
            'remote': 75, 'sandwich': 54, 'scissors': 87, 'sheep': 20, 'sink':
            81, 'skateboard': 41, 'skis': 35, 'snowboard': 36, 'spoon': 50,
            'sports ball': 37, 'stop sign': 13, 'suitcase': 33, 'surfboard':
            42, 'teddy bear': 88, 'tennis racket': 43, 'tie': 32, 'toaster':
            80, 'toilet': 70, 'toothbrush': 90, 'traffic light': 10, 'train':
            7, 'truck': 8, 'tv': 72, 'umbrella': 28, 'vase': 86, 'wine glass':
            46, 'zebra': 24}
coco_ids_to_cats = dict(map(reversed, list(coco_ids.items())))

class COCOLoder(Loader):
    def __init__(self, root_path, config):
        super(COCOLoder, self).__init__(root_path, config)
        cfg = config
        self.split = cfg.split
        assert len(self.split) > 1, 'Please separate training and testing datasets'
        self.root = root_path
        self.test_json = '%s/annotations/image_info_%s.json'
        self.train_json = '%s/annotations/instances_%s.json'

        # self.train_coco = COCO(self.train_json % (self.root, self.split[0]))
        # self.test_coco = COCO(self.train_json % (self.root, self.split[1]))
        self.train_cocos = []
        self.test_cocos = []

        for i in range(len(self.split[0])):
            coco_a = COCO(self.train_json % (self.root, self.split[0][i]))
            self.train_cocos.append(coco_a)
        for j in range(len(self.split[1])):
            coco_b = COCO(self.test_json % (self.root, self.split[1][j]))
            self.test_cocos.append(coco_b)
    #get annotation info
    def _get_coco_annotations(self, img_id, coco, only_instances=True):
        iscrowd = False if only_instances else None
        return coco.loadAnns(coco.getAnnIds(
            imgIds=img_id, catIds=[], iscrowd=iscrowd))

    def read_annotations(self, img_id, coco):
        bboxes, cats, masks = [], [], []
        anns = self._get_coco_annotations(img_id, coco)
        img = coco.loadImgs(img_id)[0]
        w, h = img['width'], img['height']
        for i in range(len(anns)):
            cat = coco_ids_to_cats[anns[i]['category_id']]
            cats.append(cat)
            bbox = anns[i]['bbox']
            bboxes.append(bbox)
            mask = self._read_segmentation(anns[i], h, w)
            mask[mask[:]> 0] = i+1
            masks.append(mask)
        return bboxes, cats, w, h, masks

    #read segmentation info
    def _read_segmentation(self, ann, H, W):
        s = ann['segmentation']
        s = s if type(s) == list else [s]
        return mask.decode(mask.frPyObjects(s, H, W)).max(axis=2)

    # def get_semantic_segmentation(self, img_id, coco):
    #     img = coco.loadImgs(img_id)[0]
    #     h, w = img['height'], img['width']
    #     segmentation = np.zeros((h, w), dtype=np.uint8)
    #     coco_anns = self._get_coco_annotations(img_id, coco, only_instances=True)
    #     for ann in coco_anns:
    #         mask = self._read_segmentation(ann, h, w)
    #         cid = ann['category_id']
    #         assert mask.shape == segmentation.shape
    #         segmentation[mask > 0] = cid
    #     return segmentation


    def collect_train_list(self):
        file_paths = []
        path = '%s/%s'
        for i in range(len(self.train_cocos)):
            self.train_filenames = self.train_cocos[i].getImgIds()
            self.img = self.train_cocos[i].loadImgs(self.train_filenames)
            for j in range(len(self.img)):
                file_path = path % (self.split[0][i], self.img[j]['file_name'])
                file_paths.append(file_path)
        return file_paths

    def collect_test_list(self):
        file_paths = []
        path = '%s/%s'
        for i in range(len(self.test_cocos)):
            test_filenames = self.test_cocos[i].getImgIds()
            name = self.test_cocos[i].loadImgs(test_filenames)
            for j in range(len(name)):
                file_path = path % (self.split[1][i], name[j]['file_name'])
                file_paths.append(file_path)

        return file_paths

    def process(self, file_path, doc):
        objects = []
        name = os.path.splitext(os.path.split(file_path)[-1])[0]
        id = str(name).split('_')[-1]
        id = int(id)
        year = str(name).split('_')[1]
        for i in range(len(self.train_cocos)):
            if year == self.split[0][i]:
                bboxs, obj_names, w, h, segmentations = self.read_annotations(id, self.train_cocos[i])

        for j in range(len(self.test_cocos)):
            if year == self.split[1][j]:
                bboxs, obj_names, w, h, segmentations = self.read_annotations(id, self.test_cocos[j])

        doc.wight = w
        doc.height = h
        assert len(bboxs) == len(obj_names), 'Wrong lable descriptions'
        for i in range(len(obj_names)):
            obj = Document()
            obj.box = bboxs[i]
            obj.name = obj_names[i]
            obj.segmentation = segmentations[i]
            objects.append(obj)
        doc.objects = objects

        return doc
