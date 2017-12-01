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

import os
import numpy as np
from tensorlab.utils.yaml_object import YamlObject
from PIL import Image

class Document(YamlObject):
    def __init__(self):
        super(YamlObject, self).__init__()
        self.__dict__['__cache__'] = {}

    @property
    def _cache(self): return self.__dict__['__cache__']

    @property
    def segmentation_id(self): return self.__get_attr__('segmentation')

    def segmentation_getter(self): return self._cache.get('__segmentation__')

    def segmentation_setter(self, seg_image):
        seg_image = seg_image.astype(np.uint8)
        self._cache['__segmentation__'] = seg_image
        self.__set_attr__('segmentation', seg_image.max())


    def save(self, path):
        # save yml
        super(Document, self).save(path)

        # collect datas
        segs = self.search_all('__segmentation__', in_cache=True)

        # save segmentation
        if len(segs) > 0:
            w, h = segs[0].shape
            im_array = np.array(segs).reshape(w, h, -1).sum(axis=2)
            seg_path = os.path.splitext(path)[0] + '.png'
            image = Image.fromarray(im_array, mode='L')
            image.save(seg_path)




    def search(self, key, in_cache=False):
        result = None
        def _do(o):
            exist = False
            if in_cache:
                exist = key in o._cache.keys()
                if exist: result = o._cache.get(key)
            else:
                exist = o.has(key)
                if exist: result = o.get(key)
            return exist

        self._search_docs(_do)
        return result


    def search_all(self, key, in_cache=False):
        results = []

        def _do(o):
            if in_cache:
                exist = key in o._cache.keys()
                if exist: results.append(o._cache.get(key))
            else:
                exist = o.has(key)
                if exist: results.append(o.get(key))

        self._search_docs(_do)
        return results


    def _search_docs(self, callback):
        def _finds(o):
            if isinstance(o, Document):
                stop = callback(o)
                if stop: return

            if isinstance(o, dict):
                for k, v in o.items():
                    _finds(v)

            elif isinstance(o, list):
                for v in o:
                    _finds(v)

        _finds(self)


    property(segmentation_getter, segmentation_setter)




