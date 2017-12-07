import os
import cv2
import json
import collections
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

        self.files = collections.defaultdict(list)

        for split in ["train", "val", "test"]:
            path = os.path.join(root_path, 'leftImg8bit',split)

            file_list_orignal = self._search_files(path, ['.png'])
            self.files[split] = file_list_orignal

    def from_json_text(self,file_path):

        with open(file_path, 'r') as f:
            json_str = f.read()
            json_dict = json.loads(json_str)
            self.imgWidth = int(json_dict['imgWidth'])
            self.imgHeight = int(json_dict['imgHeight'])
            self.objects = []
            label_id = 0
            for index in json_dict['objects']:
                obj = JsObject()
                label_id += 1
                obj.parse_object(index)
                self.objects.append(obj)

    def create_label_image(self, outline=None):
        # the size of the image
        lable_image_size = (self.imgWidth, self.imgHeight)
        background_color = 0
        # append objects
        # this is the image that we want to create
        labelImg = Image.new("L", lable_image_size, background_color)
        # from ..document import _DEFAULT_PALETTE
        # labelImg.putpalette(_DEFAULT_PALETTE)
        # a drawer to draw into the image --polygon
        drawer = ImageDraw.Draw(labelImg)
        color_index = 0
        outline = 0

        # print(len(self.objects))
        # print()

        def get_image_colors(img):
            uimage = np.unique(np.array(img))
            return uimage

        type_dict = {}

        colors = get_image_colors(labelImg)

        for obj in self.objects:

            polygon = obj.polygon
            label = obj.label

            if len(polygon) == 0:
                print('polygon size is too small')
                continue
            color_index += 1

            type_dict[color_index] = True
            # print('fuck')
            # print(label, color_index)

            try:
                if outline:
                    drawer.polygon(polygon, fill=color_index, outline=outline)
                else:
                    drawer.polygon(polygon, fill=color_index)
                    # labelImg.show()
            except:
                print("Failed to draw polygon with label {}".format(label))
                raise
            #
            # colors = get_image_colors(labelImg)
            #
            # if(len(colors) - len(pre_color) != 1):
            #     for c in pre_color:
            #         if c != color_index and c not in colors:
            #             print('index {} overwrite {}'.format(color_index, c))

#              print(label, color_index)
        # labelImg.show()
        return labelImg

    def collect_train_list(self):

        return self.files["train"] + self.files["val"]

    def collect_test_list(self):

        return self.files["test"]

    def process(self, file_path, doc):
        # dst = f.replace("_polygons.json", "_instanceTrainIds.png"
        file_path = os.path.join(self.root_path, file_path)
        json_path = file_path[:-15] +'gtFine_polygons.json'
        label_image_name = file_path[:-15] + 'mask.jpg'
        # print(json_path)
        self.from_json_text(json_path)
        label_image = self.create_label_image()
        label_image.save(label_image_name)

        mask_image = np.array(label_image,dtype=np.uint8)
        doc.width = mask_image.shape[0]
        doc.height = mask_image.shape[1]

        ins_mask = np.unique(mask_image)  # Blue value

        index_mask = ins_mask[1:ins_mask.size]
        # print('ins mask :')
        # print(index_mask)
        # print(len(index_mask), len(self.objects))
        objs = np.arange(1, len(self.objects)+1, 1)
        seg_objects = []
        for k in range(len(index_mask)):
            index = index_mask[k]
            segmentation = np.array(label_image, dtype=np.uint8)

            if index in objs:

                id = np.argwhere(objs == index)
                obj = Document()
                class_name = self.objects[id[0][0]].label
                obj.name = class_name
                cnt = self.objects[id[0][0]].polygon
                x, y, w, h = cv2.boundingRect(np.array(cnt))
                obj.box = [x, y, w, h]
                # mask_seg = np.array(mask_image_blue, dtype=np.uint8)
                segmentation[segmentation[:] != index] = 0
                obj.segmentation = segmentation
                if obj.segmentation is None: continue
                seg_objects.append(obj)

        doc.objects = seg_objects
        # print(doc)


