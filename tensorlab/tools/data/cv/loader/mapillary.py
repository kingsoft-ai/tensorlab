import numpy as np
from PIL import Image
import os
import json
from .loader import Loader
from ..document import Document


class MapillaryLoader(Loader):
    def __init__(self, root_path, config):
        super().__init__(root_path, config)
        self.root_path = root_path
        self.config = config

        # Extract catagory information
        dataset_config_path = os.path.join(root_path, "mapillary-vistas-dataset_public_v1.0", 'config.json')
        with open(dataset_config_path) as config_file:
            conf = json.load(config_file)
        self.labels = conf['labels']


    def collect_train_list(self):
        l = []

        for split in self.config.split:
            path = os.path.join(self.root_path, "mapillary-vistas-dataset_public_v1.0", split, "images")
            files_found = self._search_files(path, ['.jpg'])
            l = l + files_found
        return l

    # The testing set in Mapillary data does not come with labels
    def collect_test_list(self):
        return []

    def read_segmantation(self, instance_path, labels):

        image_array = np.array(Image.open(instance_path), dtype=np.uint16)

        w, h = image_array.shape

        segmentations = []
        obj_names = []
        bbox = []

        #Extract catagory information
        #config_path = os.path.dirname(os.path.dirname(os.path.dirname(instance_path))) # os.path.realpath('{}../../../'.format(instance_path))

        #config_path = os.path.join(config_path, 'config.json')
        #with open(config_path) as config_file:
        #    conf = json.load(config_file)
        #labels = conf['labels']

        #when catagory id equals to 0, change it to another value
        image_array[image_array == 0] = len(labels)

        instance_id = np.unique(image_array)

        #used as segmentation id because the original ones are of np.uint16 type
        segmentation_count = 1

        for id in instance_id:
            #cat_id = int(id / 256)
            #instance_id = int(id % 256)
            cat_id = id >> 8
            instance_id = id - ((id >> 8) << 8)

            #skip if the catagory is not instance-specific
            #if not labels[cat_id if id != 0 else len(labels)]["instances"]:
                #continue
            if labels[cat_id]['instances'] == False: continue

            instance_array = np.copy(image_array)
            instance_array[image_array != id] = 0
            instance_array[image_array == id] = segmentation_count
            segmentation_count += 1
            segmentations.append(instance_array)

            instance_coodinates = np.where(instance_array != 0)
            min_x = int(np.min(instance_coodinates[0]))
            min_y = int(np.min(instance_coodinates[1]))
            max_x = int(np.max(instance_coodinates[0]))
            max_y = int(np.max(instance_coodinates[1]))
            box = [min_y, min_x, max_y - min_y, max_x - min_x]
            bbox.append(box)

            name = labels[cat_id if id != 0 else len(labels)]["readable"] + " - " + str(instance_id)
            obj_names.append(name)

        return bbox, obj_names, w, h, segmentations

    def process(self, file_path, doc):
        objects = []
        filename = os.path.splitext(os.path.basename(file_path))[0]
        filepath = os.path.dirname(os.path.dirname(file_path))

        instance_path = os.path.join(self.root_path, filepath, "instances/%s.png" % filename)

        bbox, obj_names, w, h, segmentations = self.read_segmantation(instance_path, self.labels)

        doc.width = w
        doc.height = h

        for i in range(len(segmentations)):
            obj = Document()
            obj.segmentation = segmentations[i]
            obj.name = obj_names[i]
            obj.box = bbox[i]
            objects.append(obj)
        doc.objects = objects

        return doc