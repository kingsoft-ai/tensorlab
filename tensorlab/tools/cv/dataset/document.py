####################################################################################################
# ROOT:
#   dataset: a name of dataset (framework set)
#   path: picture relative path (framework set)
#   width: picture width
#   height: picture width
#   seg_tag: a segmentation tag (framework set)
#   box_tag: a box tag (framework set)
#   objects: a array full with document
#
# Object:
#   name: a unique tag name of this object
#   box: a bounding box [x, y, w, h]
#   segmentation: a base64 encoder of segmentation described by [(x1,y1), (x2, y2) ...]
#   pose: person pose name

####################################################################################################

import base64
import pickle
import numpy as np
from tensorlab.utils.yaml_object import YamlObject

class Document(YamlObject):

    @property
    def segmentation(self):
        # check cache
        if '__segmentation__' in self._cache:
            return self._cache['__segmentation__']

        # parse segmentation
        s = self.__get_attr__('segmentation')
        b = base64.decode(s)
        seg = pickle.loads(b)

        # save cache
        self._cache['__segmentation__'] = seg
        return seg

    @segmentation.setter
    def segmentation(self, v):
        b = pickle.dumps(v)
        s = base64.b64encode(b)
        self.__set_attr__('segmentation', s)


    # generate segementation 2d image with (width, height), dtype is bool
    @property
    def segmentation2d(self):
        # check cache
        if '__segmentation2d__' in self._cache:
            return self._cache['__segmentation2d__']

        # convert
        root = self.root
        width = root.width
        height = root.height
        seg = self.segmentation
        seg = np.array(seg)
        indexes = np.split(seg, 2, axis=1)
        image = np.zeros((width, height), dtype=np.bool)
        image[indexes] = 1

        # save
        self._cache['__segmentation2d__'] = image
        return image


    def find(self, key):
        def _find(o):
            if o.has(key): return getattr(o, key)
            v = None
            for c in self.childs:
                v = _find(c)
                if v is not None: break
            return v
        return _find(self)







