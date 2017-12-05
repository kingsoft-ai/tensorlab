from .loader import Loader
from pycocotools.coco import COCO
from pycocotools import mask
import numpy as np
import os
from ..document import Document

class COCLoder(Loader):
    def __init__(self, root_path, config):
        super(COCLoder, self).__init__(root_path, config)
        cfg = config
        self.split_train = cfg.split_train
        self.split_test = cfg.split_test
        self.root = root_path
        self.test_json = '%s/annotations/image_info_%s.json'
        self.train_json = '%s/annotations/instances_%s.json'

    #get annotation info
    def _get_coco_annotations(self, img_id, only_instances=True):
        iscrowd = False if only_instances else None
        return self.coco.loadAnns(self.coco.getAnnIds(
            imgIds=img_id, catIds=self.included_coco_ids, iscrowd=iscrowd))

    def read_annotations(self, img_id):
        anns = self._get_coco_annotations(img_id)
        bboxes = [ann['bbox'] for ann in anns]
        cats = [ann['category_id'] for ann in anns]
        labels = [[cat_name] for cat_name in cats]
        img = self.coco.loadImgs(img_id)[0]

        return np.array(bboxes).reshape((-1, 4)), np.array(labels), \
               img['width'], img['height'], np.zeros_like(labels, dtype=np.bool)


    #read segmentation info
    def _read_segmentation(self, ann, H, W):
        s = ann['segmentation']
        s = s if type(s) == list else [s]
        return mask.decode(mask.frPyObjects(s, H, W)).max(axis=2)

    def get_semantic_segmentation(self, img_id):
        img = self.coco.loadImgs(img_id)[0]
        h, w = img['height'], img['width']
        segmentation = np.zeros((h, w), dtype=np.uint8)
        coco_anns = self._get_coco_annotations(img_id, only_instances=False)
        for ann in coco_anns:
            mask = self._read_segmentation(ann, h, w)
            cid = self.coco_ids_to_internal[ann['category_id']]
            assert mask.shape == segmentation.shape
            segmentation[mask > 0] = cid
        return segmentation


    def collect_train_list(self):
        file_paths = []
        path = '%s/%s'
        for i in range(len(self.split_train)):
            # print(self.train_json % self.root, self.split_train[i])
            coco = COCO(self.train_json % (self.root, self.split_train[i]))
            self.train_filenames = coco.getImgIds()
            self.img = coco.loadImgs(self.train_filenames)
            for j in range(len(self.img)):
                file_path = path % (self.split_train[i], self.img[j]['file_name'])
                file_paths.append(file_path)
        return file_paths

    def collect_test_list(self):
        file_paths = []
        # path = '%s/%s'
        # for i in range(len(self.split_test)):
        #     coco = COCO(self.test_json % (self.root, self.split_test[i]))
        #     test_filenames = coco.getImgIds()
        #     name = coco.loadImgs(test_filenames)
        #     for j in range(len(name)):
        #         file_path = path % (self.split_test[i], name[j]['file_name'])
        #         file_paths.append(file_path)
        return file_paths

    def process(self, file_path, doc):
        objects = []
        # for i in range(len(self.img)):



        bboxs, obj_name, w, h= self.read_annotations(f)
        doc.width = w
        doc.height = h
        assert len(bboxs) == len(obj_name),  "Wrong lable descriptions"

        for i in range(len(obj_name)):
            obj = Document()
            obj.box = bboxs[i]
            obj.name = obj_name[i]

            if filename in self.train_seg_name + self.test_seg_lable:
                obj.segmentation = self.read_segmentations(seg_dirpath, bboxs, i)

            objects.append(obj)
        doc.objects = objects
        return doc
