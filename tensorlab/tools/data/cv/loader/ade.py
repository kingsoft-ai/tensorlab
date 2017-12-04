import os
import collections
import numpy as np
import scipy.misc as m
import scipy.io
from PIL import Image
from io import StringIO

from .loader import Loader
from ..document import Document

def recursive(root_path, suffix=''):
    return [os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(root_path)
            for filename in filenames if filename.endswith(suffix)]

class ADE20KLoader(Loader):
    def __init__(self, root_path, config):
        super(ADE20KLoader, self).__init__(root_path, config)
        self.root_path = root_path
        self.config = config
        self.n_classes = 150
        self.files = collections.defaultdict(list)
        for split in ["training", "validation"]:
            #file_list_orignal = recursive(root_path + 'images' + split + '/', '.jpg')
            #file_list_seg = recursive(root_path + 'images' + split + '/', '.png')
            path = os.path.join(root_path, 'ADE20K', 'images', split)
            file_list_orignal = self._search_files(path, ['.jpg'])
            self.files[split] = file_list_orignal

    def collect_train_list(self):

        return self.files["training"] + self.files["validation"]

    def collect_test_list(self):

        return []

    def get_label_path(self):
        labels_path=[]
        attr_path = []
        for split in ["training", "validation"]:
            for index in self.files[split]:
                label = self._get_fileseg(index)
                labels_path.append(label)

        return labels_path

    def get_object_class_mask(self, seg_image):
        mask_image = seg_image.astype(int)
        label_mask = np.zeros(mask_image.shape[0],mask_image.shape[1])
        label_mask = (mask_image[:,:,0] / 10.0) * 256 + mask_image[:,:,1]

        return np.array(label_mask, dtype=np.uint8)

    def process(self,file_path,doc):
        objects = []
        file_path = os.path.join(self.root_path, file_path)
        seg_path = file_path[:-4] + '_seg.png'
        attr_path = file_path[:-4] + '_atr.txt'
        class_names = []
        with open(attr_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                splits = line.split('#')
                instance_number = str(splits[0])
                part_level = int(splits[1])
                occlude = int(splits[2])
                if (part_level == 0):
                    class_names.append(str(splits[4]))
                else:
                    continue
        seg_map = Image.open(seg_path)

        mask_image = np.array(seg_map,dtype=np.uint8)
        doc.width = mask_image.shape[0]
        doc.height = mask_image.shape[1]

        mask_image_blue = mask_image[:, :, 2]
        ins_mask = np.unique(mask_image_blue) #Blue value
        index_mask = ins_mask[1:ins_mask.size]

        # assert len(index_mask) == len(class_names), "class size and mask index are not matched!"
        # print(len(index_mask),len(class_names))
        # print(attr_path)

        if(len(index_mask) == len(class_names)):

            for k in range(len(class_names)):
                obj = Document()
                obj.name = class_names[k]
                mask_image_blue[mask_image_blue[:] != index_mask[k]] = 0
                # mask_seg = np.array(mask_image_blue, dtype=np.uint8)
                obj.segmentation = mask_image_blue
                if obj.segmentation is None: continue
                objects.append(obj)

            doc.objects = objects
        else:
            print('data cast！！！')




