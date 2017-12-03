
from .loader import Loader
import os
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image

from ..document import Document


class VOCLoder(Loader):
    def __init__(self, root_path, config):
        super(VOCLoder, self).__init__(root_path, config)
        cfg = config
        self.year = cfg.year
        self.split = cfg.split
        assert cfg.year in ['07','12'], "Wrong year for this datasets"
        self.root = root_path
        self.filelist = 'VOCdevkit/VOC20%s/ImageSets/Main/%s.txt'
        self.seg_filelist = 'VOCdevkit/VOC20%s/ImageSets/Segmentation/%s.txt'

        # self.filelist = 'F:/datasets/VOC/VOCdevkit/VOC20%s/ImageSets/Main/%s.txt'
        # self.seg_filelist = 'F:/datasets/VOC/VOCdevkit/VOC20%s/ImageSets/Segmentation/%s.txt'

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

    def read_segmentations(self, name, bboxs, obj_index):
        seg_folder = os.path.join(self.root, 'VOCdevkit/VOC20%s/SegmentationObject/' % self.year)
        seg_file = os.path.join(seg_folder, name + '.png')
        seg_map = Image.open(seg_file)
        segmentation = np.array(seg_map, dtype=np.uint8)
        seg_copy = np.array(seg_map, dtype=np.uint8)
        min_mask = []
        cord_info = []
        for i in range(len(bboxs)):
            _min = np.min(seg_copy)
            if _min == 0:
                cord_x, cord_y = np.where(seg_copy[:] == _min)
                for j in zip(cord_x, cord_y):
                    seg_copy[j] = 255
            else:
                min_mask.append(_min)
                cord_x, cord_y = np.where(seg_copy[:] == _min)
                min_cord_y,max_cord_y,min_cord_x, max_cord_x = np.min(cord_y), np.max(cord_y), np.min(cord_x), np.max(cord_x)
                cord_info.append([min_cord_y,max_cord_y,min_cord_x, max_cord_x])
                for j in zip(cord_x, cord_y):
                    seg_copy[j] = 255

        dist = []
        y_min, y_max, x_min, x_max = bboxs[obj_index][0], bboxs[obj_index][0] + bboxs[obj_index][2], bboxs[obj_index][1], bboxs[obj_index][1] + bboxs[obj_index][3]
        for j in range(len(cord_info)):
            z1 = np.sqrt(np.square(cord_info[j][1] - y_max) + np.square(cord_info[j][3] - x_max))
            z2 = np.sqrt(np.square(cord_info[j][0] - y_min) - np.square(cord_info[j][2] - x_min))
            d = z1 + z2
            dist.append(d)
        right_index_bbox = np.where(dist == np.min(dist))
        assert len(min_mask)==len(cord_info)
        mask_seg = np.zeros((segmentation.shape),dtype=np.uint16)
        # y_min, y_max, x_min, x_max = bboxs[0], bboxs[0]+bboxs[2], bboxs[1], bboxs[1]+bboxs[3]
        # min_cord_y, max_cord_y, min_cord_x, max_cord_x = cord_info[cord_]
        x,y = np.where((segmentation[x_min:x_max, y_min:y_max] == min_mask[right_index_bbox] ))
        x = x + [x_min]*len(x)
        y = y + [y_min]*len(y)
        for cord in zip(x,y):
            mask_seg[cord] = obj_index

        return mask_seg

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
        return train_file

    def collect_test_list(self):
        test_filenames = []
        with open(os.path.join(self.root, self.filelist % (self.year, self.split[2])), 'r') as f:
            test_filename = f.read().split('\n')[:-1]
            for i in range(len(test_filename)):
                test_file = 'VOCdevkit/VOC20%s/JPEGImages/%s.jpg' % (self.year, test_filename[i])
                test_filenames.append(test_file)
        return test_filenames

    def process(self, file_path, doc):
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
            obj = Document()
            obj.box = bboxs[i]
            obj.name = obj_name[i]
            if filename in self.train_seg_name:
                #obj.segmentation = self.read_segmentations(filename,bboxs[i])
                obj.segmentation = self.read_segmentations(filename,bboxs, i)

            objects.append(obj)
        doc.objects = objects
        return doc
