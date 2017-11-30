from .loader import Loader
import os
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image

from .. import config, document


class VOCLoder(Loader):
    def __init__(self, root_path, config):
        super(VOCLoder, self).__init__(root_path, config)
        cfg = config
        self.year = cfg.year
        self.split = cfg.split
        assert cfg.year in ['07','12'], "wrong year"
        self.root = root_path
        self.filelist = 'VOCdevkit/VOC20%s/ImageSets/Main/%s.txt'
        self.seg_filelist = 'VOCdevkit/VOC20%s/ImageSets/Segmentation/%s.txt'
        with open(os.path.join(self.root, self.seg_filelist % (self.year, self.split[0])), 'r') as f:
            self.seg_filenames1 = f.read().split('\n')[:-1]
        with open(os.path.join(self.root, self.seg_filelist % (self.year, self.split[1])), 'r') as f:
            self.seg_filenames2 = f.read().split('\n')[:-1]
        with open(os.path.join(self.root, self.seg_filelist % (self.year, self.split[2])), 'r') as f:
            self.test_seg_lable = f.read().split('\n')[:-1]
        self.train_seg_name = self.seg_filenames1+self.seg_filenames2
        # self.train_seg_lable = self.read_segmentations(self.train_seg_name)
        # self.test_seg_lable = self.read_segmentations(self.seg_filenames3)

    def read_annotations(self, name):
        bboxes = []
        cats = []
        # print('%s/VOCdevkit/VOC20%s/Annotations/%s.xml' % (self.root, self.year, name))
        for i in range(len(self.collect_train_list())):
            tree = ET.parse('%s/VOCdevkit/VOC20%s/Annotations/%s.xml' % (self.root, self.year, self.collect_train_list()[i]))
            root = tree.getroot()
            width = int(root.find('size/width').text)
            height = int(root.find('size/height').text)
            for obj in root.findall('object'):
                cat = obj.find('name').text
                cats.append(cat)
                bbox_tag = obj.find('bndbox')
                x = int(bbox_tag.find('xmin').text)
                y = int(bbox_tag.find('ymin').text)
                w = int(bbox_tag.find('xmax').text)-x
                h = int(bbox_tag.find('ymax').text)-y
                bboxes.append([x, y, w, h])
                # pose = obj.find('post').text



        output = bboxes, cats, width, height
        return output

    def read_segmentations(self, name):
        mask_index = []
        seg_folder = self.root + 'VOCdevkit/VOC20%s/SegmentationClass/' % self.year
        seg_file = seg_folder + name + '.png'
        seg_map = Image.open(seg_file)
        segmentation = np.sum(np.array(seg_map, dtype=np.uint8),axis=2)
        x,y = np.where(segmentation[:]>0)
        mask_index.append((x,y))
        return mask_index

    def collect_train_list(self):
        with open(os.path.join(self.root, self.filelist % (self.year, self.split[0])), 'r') as f:
            tra_filenames = f.read().split('\n')[:-1]
            # tra_filenames = 'VOCdevkit/VOC20%s/JPEGImages/%s.jpg' % (self.year, self.tra_filenames)
        with open(os.path.join(self.root, self.filelist % (self.year, self.split[0])), 'r') as f:
            val_filenames = f.read().split('\n')[:-1]
            # val_filenames = 'VOCdevkit/VOC20%s/JPEGImages/%s.jpg' % (self.year, self.val_filenames)
        # train_files = [tra_filenames, val_filenames]
        train_file = tra_filenames + val_filenames
        return train_file[:10]

    def collect_test_list(self):
        with open(os.path.join(self.root, self.filelist % (self.year, self.split[0])), 'r') as f:
            test_filenames = f.read().split('\n')[:-1]
            # test_filenames = 'VOCdevkit/VOC20%s/JPEGImages/%s.jpg' % (self.year, test_filenames)
        return test_filenames[:10]

    def process(self, file_path):
        doc = document.Document()
        objects = []
        obj = document.Document()

        with open(os.path.join(self.root, 'VOCdevkit/VOC20%s/Annotations/%s.xml' % (self.year, file_path)), 'r') as f:
            bboxs, obj_name, w, h= self.read_annotations(f)
        doc.width = w
        doc.height = h
        assert len(bboxs) == len(obj_name),  "Wrong lable descriptions"
        # num = len(obj_name)
        # for i in range(num):
        obj.box = bboxs
        obj.name = obj_name

        if obj_name == f:
            obj.segmentation = self._to_base64(self.read_segmentations(obj_name))
        objects.append(obj)
        doc.objects = objects
        return doc
