import os
import collections
import numpy as np
import scipy.misc as m
import scipy.io
import matplotlib.pyplot as plt
from PIL import Image

from .loader import Loader
from .. import config
from ..document import Document

def recursive(root_path, suffix=''):
    return [os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(root_path)
            for filename in filenames if filename.endswith(suffix)]

class ADE20KLoader(Loader):
    def __init__(self, root_path, config):
        super(ADE20KLoader, self).__init__(root_path, config)
        self.root_path = root_path
        self.root_path = root_path
        self.config = config
        self.n_classes = 150
        self.files = collections.defaultdict(list)
        for split in ["training", "validation"]:
            file_list = recursive(root_path + 'images' + split + '/', '.jpg')
            self.files[split] = file_list

    def collect_train_list(self):

        return self.files["training"] + self.files["validation"]

    def collect_test_list(self):

        return []

    def _get_fileseg(self, index):
        img_path = self.files[self.split][index].rstrip()
        lbl_path = img_path[:-4] + '_seg.png'
        attr_path = img_path[:-4] + '_atr.txt'
        output = lbl_path,attr_path

        return output

    def get_labelpath(self):
        labels_path=[]
        for split in ["training", "validation"]:
            for index in self.files[split]:
                label = self._get_fileseg(index)
                labels_path.append(label)

        return labels_path

    def get_objectClassMask(self, mask_image):
        mask_image = mask_image.astype(int)
        label_mask = np.zeros(mask_image.shape[0],mask_image.shape[1])
        label_mask = (mask_image[:,:,0] / 10.0) * 256 + mask_image[:,:,1]

        return np.array(label_mask, dtype=np.uint8)

    def get_instanceMask(self, mask_image):
        mask_image = mask_image.astype(int)
        _, _, instance_index = np.unique(mask_image[:,:,2])
        instance_mask = np.reshape(instance_index - 1,np.array(mask_image[:,:,2]).size)

        return instance_mask

    def parse_attr(self,filename):
        f =open(filename,'r')
        d = np.loadtxt(f,delimiter='#',
                       dtype={'names': ('instance_number', 'part_level', 'occlude', 'class_name', 'orignal_name','list'),
                              'formats': ('s4', 'i4', 'i4', 's4', 's4','s4')}
                       )
        f.close()
        instance_n = d['instance_number']
        parts = d['part_level']
        isocclude = ['occlude']
        classes = ['class_name']
        orignals = ['orignal_name']
        ls = ['list']
        object = []
        for ins,part,isocc,class_name,ori,l in zip(instance_n,parts,isocclude,classes,orignals,ls)
            inumber = int(ins)
            ispart = part > 0
            iscrop = isocc
            doc = Document()
            doc.name = class_name

    def decode_segmap(self, temp, plot=False):
        # Verify that the color mapping is 1-to-1
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = 10 * (l % 10)
            g[temp == l] = l
            b[temp == l] = 0

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = (r / 255.0)
        rgb[:, :, 1] = (g / 255.0)
        rgb[:, :, 2] = (b / 255.0)
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb

    # def process(self,file_path,doc):





if __name__ == '__main__':
    # imgpath = "/Users/qianliu/work/tensorlab/tensorlab/tools/cv/dataset/ADE_train_00015646_seg.png"
    # seg_map = Image.open(imgpath)
    # seg_map.show()
    #
    # print(seg_map)
    data = scipy.io.loadmat('/Users/qianliu/work/tensorlab/tensorlab/tools/cv/dataset/loader/index_ade20k.mat')
    print(data.keys())
    print(1)


