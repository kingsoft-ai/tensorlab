
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


    def read_annotations(self, name):
        bboxes = []
        cats = []
        tree = ET.parse(name)
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

        output = bboxes, cats, width, height
        return output

    def read_segmentations(self, name):
        mask_index = []
        seg_folder = os.path.join(self.root, 'VOCdevkit/VOC20%s/SegmentationClass/' % self.year)
        seg_file = os.path.join(seg_folder, name + '.png')
        seg_map = Image.open(seg_file)
        segmentation = np.array(seg_map, dtype=np.uint8)
        x,y = np.where(segmentation[:]>0)
        for cord in zip(x, y):
            cord = tuple(cord)
            mask_index.append(cord)
        return mask_index

    def collect_train_list(self):
        tra_filenames, val_filenames = [], []
        with open(os.path.join(self.root, self.filelist % (self.year, self.split[0])), 'r') as f:
            tra_filename = f.read().split('\n')[:-1]
            for i in range(len(tra_filename)):
                tra_file = 'VOCdevkit/VOC20%s/JPEGImages/%s.jpg' % (self.year, tra_filename[i])
                tra_filenames.append(tra_file)
        with open(os.path.join(self.root, self.filelist % (self.year, self.split[1])), 'r') as f:
            val_filename = f.read().split('\n')[:-1]
            for i in range(len(tra_filename)):
                val_file = 'VOCdevkit/VOC20%s/JPEGImages/%s.jpg' % (self.year, val_filename[i])
                val_filenames.append(val_file)
        train_file = tra_filenames + val_filenames
        return train_file[:10]

    def collect_test_list(self):
        test_filenames = []
        with open(os.path.join(self.root, self.filelist % (self.year, self.split[2])), 'r') as f:
            test_filename = f.read().split('\n')[:-1]
            for i in range(len(test_filename)):
                test_file = 'VOCdevkit/VOC20%s/JPEGImages/%s.jpg' % (self.year, test_filename[i])
                test_filenames.append(test_file)
        return test_filenames[:10]

    def process(self, file_path):
        doc = document.Document()
        objects = []


        filename = os.path.splitext(os.path.basename(file_path))[0]
        file_path = os.path.dirname(os.path.dirname(file_path))

        file_path = os.path.join(self.root, file_path, 'Annotations/%s.xml' % filename)

        with open(file_path, 'r') as f:
            bboxs, obj_name, w, h= self.read_annotations(f)
        doc.width = w
        doc.height = h
        assert len(bboxs) == len(obj_name),  "Wrong lable descriptions"
        for i in range(len(obj_name)):
            obj = doc.child()
            obj.box = bboxs[i]
            obj.name = obj_name[i]
        if filename in self.train_seg_name:
            obj.segmentation = self.read_segmentations(filename)
        objects.append(obj)
        doc.objects = objects
        return doc
