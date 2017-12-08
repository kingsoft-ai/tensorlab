
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
        self.root = root_path
        self.filelist = 'VOCdevkit/VOC20%s/ImageSets/Main/%s.txt'
        self.seg_filelist = 'VOCdevkit/VOC20%s/ImageSets/Segmentation/%s.txt'
        self.seg1, self.seg2 = [], []
        for i in range(len(self.year)):
            with open(os.path.join(self.root, self.seg_filelist % (self.year[i], self.split[0])), 'r') as f:
                seg_filenames1 = f.read().split('\n')[:-1]
                self.seg1 += seg_filenames1
            with open(os.path.join(self.root, self.seg_filelist % (self.year[i], self.split[1])), 'r') as f:
                seg_filenames2 = f.read().split('\n')[:-1]
                self.seg2 += seg_filenames2
            self.train_seg_name = self.seg1+self.seg2
        with open(os.path.join(self.root, self.seg_filelist % (self.year[0], self.split[2])), 'r') as f:
            self.test_seg_lable = f.read().split('\n')[:-1]

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

    def read_segmentations(self, seg_file, bboxs):
        seg_map = Image.open(seg_file)
        seg_copy = np.array(seg_map, dtype=np.uint8)
        min_mask = []
        cord_info = []

        bg = np.min(seg_copy)
        if bg == 0:
            for i in range(len(bboxs)+1):
                _min = np.min(seg_copy)
                if _min == 0:
                    seg_copy[seg_copy[:]==_min]=255
                else:
                    min_mask.append(_min)
                    cord_x, cord_y = np.where(seg_copy[:] == _min)
                    min_cord_y,max_cord_y,min_cord_x, max_cord_x = np.min(cord_y), np.max(cord_y), np.min(cord_x), np.max(cord_x)
                    cord_info.append([min_cord_y,max_cord_y,min_cord_x, max_cord_x])
                    seg_copy[seg_copy[:] == _min] = 255
        else:
            for i in range(len(bboxs)):
                _min = np.min(seg_copy)
                min_mask.append(_min)
                cord_x, cord_y = np.where(seg_copy[:] == _min)
                min_cord_y, max_cord_y, min_cord_x, max_cord_x = np.min(cord_y), np.max(cord_y), np.min(cord_x), np.max(
                    cord_x)
                cord_info.append([min_cord_y, max_cord_y, min_cord_x, max_cord_x])
                seg_copy[seg_copy[:] == _min] = 255


        if not len(bboxs)==len(min_mask) == len(cord_info):
            print('maybe {} object has no bbox'.format(seg_file))

        dists = []
        for j in range(len(bboxs)):
            dist = []
            for k in range(len(cord_info)):
                y_min, y_max, x_min, x_max = bboxs[j][0], bboxs[j][0] + bboxs[j][2], \
                                             bboxs[j][1], bboxs[j][1] + bboxs[j][3]
                z1 = np.square(cord_info[k][1] - y_max) + np.square(cord_info[k][3] - x_max)
                z2 = np.square(cord_info[k][0] - y_min) + np.square(cord_info[k][2] - x_min)
                d = z1 + z2
                dist.append(d)
            dists.append(dist)

        cords = []
        for row in range(len(dists)):
            low = np.min(dists)
            b_ind, p_ind = np.where(dists[:] == low)
            b_ind, p_ind = b_ind[0], p_ind[0]
            cords.append([b_ind, p_ind])
            for clow in range(len(dists[0])):
                dists[b_ind][clow], dists[clow][p_ind] = np.inf, np.inf


        cords = sorted(cords, key=lambda c:c[0])

        segmentations = []
        for co in cords:
            segmentation = np.array(seg_map, dtype=np.uint8)
            segmentation[segmentation[:] != min_mask[co[1]]] = 0
            segmentations.append(segmentation)
        # print('image name: {}, min_mask:{}, dist:{}, cord_info{}'.format(seg_file, min_mask, dist, cord_info))
        # x,y = np.where(segmentation[x_min:x_max, y_min:y_max] == min_mask[right_index_bbox])
        # x = x + [x_min]*len(x)
        # y = y + [y_min]*len(y)
        # for cord in zip(x,y):
        #     mask_seg[cord] = obj_index + 1
        # segmentation[segmentation[:]!= min_mask[right_index_bbox]] = 0

        return segmentations

    def collect_train_list(self):
        tra_filenames, val_filenames = [], []
        for i in range(len(self.year)):
            with open(os.path.join(self.root, self.filelist % (self.year[i], self.split[0])), 'r') as f:
                tra_filename = f.read().split('\n')[:-1]
                for j in range(len(tra_filename)):
                    tra_file = 'VOCdevkit/VOC20%s/JPEGImages/%s.jpg' % (self.year[i], tra_filename[j])
                    tra_filenames.append(tra_file)
            with open(os.path.join(self.root, self.filelist % (self.year[i], self.split[1])), 'r') as f:
                val_filename = f.read().split('\n')[:-1]
                for j in range(len(tra_filename)):
                    val_file = 'VOCdevkit/VOC20%s/JPEGImages/%s.jpg' % (self.year[i], val_filename[j])
                    val_filenames.append(val_file)
        train_file = tra_filenames + val_filenames
        return train_file

    def collect_test_list(self):
        test_filenames = []
        with open(os.path.join(self.root, self.filelist % (self.year[0], self.split[2])), 'r') as f:
            test_filename = f.read().split('\n')[:-1]
            for j in range(len(test_filename)):
                test_file = 'VOCdevkit/VOC20%s/JPEGImages/%s.jpg' % (self.year[0], test_filename[j])
                test_filenames.append(test_file)
        return test_filenames

    def process(self, file_path, doc):
        objects = []
        filename = os.path.splitext(os.path.basename(file_path))[0]
        file_path = os.path.dirname(os.path.dirname(file_path))
        seg_dirpath = os.path.join(self.root, 'VOCdevkit/VOC2012/', 'SegmentationObject/%s.png' % '2009_004969')
        ann_dir_path = os.path.join(self.root, 'VOCdevkit/VOC2012/', 'Annotations/%s.xml' % '2009_004969')

        with open(ann_dir_path, 'r') as f:
            bboxs, obj_name, w, h= self.read_annotations(f)
        doc.width = w
        doc.height = h

        assert len(bboxs) == len(obj_name),  "Wrong lable descriptions"


        if filename in self.train_seg_name + self.test_seg_lable:
            segmentations = self.read_segmentations(seg_dirpath, bboxs)
            for i in range(len(obj_name)):
                obj = Document()
                obj.segmentation = segmentations[i]
                obj.box = bboxs[i]
                obj.name = obj_name[i]
                objects.append(obj)
        else:
            for i in range(len(obj_name)):
                obj = Document()
                obj.box = bboxs[i]
                obj.name = obj_name[i]
                objects.append(obj)
        doc.objects = objects
        return doc
