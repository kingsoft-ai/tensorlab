import os
import cv2
import json
import numpy as np
from PIL import Image
from PIL import ImageDraw
from collections import namedtuple

from .loader import Loader
from ..document import Document

# create A point in a polygon ,like as tuple
Point = namedtuple('Point', ['x', 'y'])

# Class that contains the information of a single annotated object in json 'objects'
class JsObject:
    # Constructor
    def __init__(self):
        # the label
        self.label = ""
        # the polygon as list of points
        self.polygon = []

    def __str__(self):
        polyText = ""
        if self.polygon:
            if len(self.polygon) <= 4:
                for p in self.polygon:
                    polyText += '({},{}) '.format( p.x , p.y )
            else:
                polyText += '({},{}) ({},{}) ... ({},{}) ({},{})'.format(
                    self.polygon[0].x, self.polygon[0].y,
                    self.polygon[1].x, self.polygon[1].y,
                    self.polygon[-2].x, self.polygon[-2].y,
                    self.polygon[-1].x, self.polygon[-1].y)
        else:
            polyText = "none"
        text = "Object: {} - {}".format( self.label , polyText )
        return text

    def parse_object(self, json_text):
        self.label = str(json_text['label'])
        self.polygon = [Point(p[0],p[1]) for p in json_text['polygon']]

# The annotation of a whole image
class Annotation:
    # Constructor
    def __init__(self,json_file_name):
        self.file_path = json_file_name
        # the width of that image and thus of the label image
        self.imgWidth  = 0
        # the height of that image and thus of the label image
        self.imgHeight = 0
        # the list of objects ,including labels and polygon points
        self.objects = []

    def from_json_text(self):
        with open(self.file_path, 'r') as f:
            json_str = f.read()
            json_dict = json.loads(json_str)
            self.imgWidth = int(json_dict['imgWidth'])
            self.imgHeight = int(json_dict['imgHeight'])
            self.objects = []
            for index in json_dict['objects']:
                obj = JsObject()
                obj.parse_object(index)
                self.objects.append(obj)

# create a label image from json's polygon ,color index is segmentation id

class CityScapesLoader(Loader):
    def __init__(self, root_path, config):
        super(CityScapesLoader, self).__init__(root_path, config)
        self.root_path = root_path
        self.config = config

        self.imgWidth = 0
        # the height of that image and thus of the label image
        self.imgHeight = 0
        # the list of objects ,including labels and polygon points
        self.objects = []

        for split in ["train", "val"]:
            #file_list_orignal = recursive(root_path + 'images' + split + '/', '.jpg')
            #file_list_seg = recursive(root_path + 'images' + split + '/', '.png')
            path = os.path.join(root_path, 'CityScapes', split)
            file_list_orignal = self._search_files(path, ['.png'])
            self.files[split] = file_list_orignal

    def from_json_text(self):
        with open(self.file_path, 'r') as f:
            json_str = f.read()
            json_dict = json.loads(json_str)
            self.imgWidth = int(json_dict['imgWidth'])
            self.imgHeight = int(json_dict['imgHeight'])
            self.objects = []
            for index in json_dict['objects']:
                obj = JsObject()
                obj.parse_object(index)
                self.objects.append(obj)

    def create_label_image(self, outline=None):
        # the size of the image
        lable_image_size = (self.imgWidth, self.imgHeight)
        background_color = 0
        # append objects
        self.from_json_text()
        # this is the image that we want to create
        labelImg = Image.new("L", lable_image_size, background_color)
        # a drawer to draw into the image --polygon
        drawer = ImageDraw.Draw(labelImg)
        color_index = 1

        print(len(self.objects))
        print()
        for obj in self.objects:
            polygon = obj.polygon
            label = obj.label

            if len(polygon) == 0:
                continue
            color_index += 1
            # debug
            print(obj.label, color_index)

            try:
                if outline:
                    drawer.polygon(polygon, fill=color_index, outline=outline)
                else:
                    drawer.polygon(polygon, fill=color_index)
            except:
                print("Failed to draw polygon with label {}".format(label))
                raise

        return labelImg


    def collect_train_list(self):

        return self.files["training"] + self.files["val"]

    def collect_test_list(self):

        return self.files["test"]

    def process(self, file_path, doc):
        label_image = self.create_label_image()

        mask_image = np.array(label_image, dtype=np.uint8)
        doc.width = mask_image.shape[0]
        doc.height = mask_image.shape[1]

        ins_mask = np.unique(mask_image)  # Blue value
        index_mask = ins_mask[1:ins_mask.size]


        if (len(index_mask) == len(self.objects)):

            id=1
            for k in range(len(self.objects)):
                obj = Document()
                obj.name = self.objects[k].label
                id += 1
                # mask_seg = np.array(mask_image_blue, dtype=np.uint8)
                obj.segmentation = id
                if obj.segmentation is None: continue
                self.objects.append(obj)

            doc.objects = self.objects
            print(doc)
        else:
            print('data cast！！！')


