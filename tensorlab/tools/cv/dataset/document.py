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
from PIL import Image
from PIL.ImagePalette import ImagePalette
from tensorlab.utils.yaml_object import YamlObject

# set output palette
_DEFAULT_PALETTE_COLOR_256 = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128, 128, 64, 128, 0, 192, 128, 128, 192, 128, 64, 64, 0, 192, 64, 0, 64, 192, 0, 192, 192, 0, 64, 64, 128, 192, 64, 128, 64, 192, 128, 192, 192, 128, 0, 0, 64, 128, 0, 64, 0, 128, 64, 128, 128, 64, 0, 0, 192, 128, 0, 192, 0, 128, 192, 128, 128, 192, 64, 0, 64, 192, 0, 64, 64, 128, 64, 192, 128, 64, 64, 0, 192, 192, 0, 192, 64, 128, 192, 192, 128, 192, 0, 64, 64, 128, 64, 64, 0, 192, 64, 128, 192, 64, 0, 64, 192, 128, 64, 192, 0, 192, 192, 128, 192, 192, 64, 64, 64, 192, 64, 64, 64, 192, 64, 192, 192, 64, 64, 64, 192, 192, 64, 192, 64, 192, 192, 192, 192, 192, 32, 0, 0, 160, 0, 0, 32, 128, 0, 160, 128, 0, 32, 0, 128, 160, 0, 128, 32, 128, 128, 160, 128, 128, 96, 0, 0, 224, 0, 0, 96, 128, 0, 224, 128, 0, 96, 0, 128, 224, 0, 128, 96, 128, 128, 224, 128, 128, 32, 64, 0, 160, 64, 0, 32, 192, 0, 160, 192, 0, 32, 64, 128, 160, 64, 128, 32, 192, 128, 160, 192, 128, 96, 64, 0, 224, 64, 0, 96, 192, 0, 224, 192, 0, 96, 64, 128, 224, 64, 128, 96, 192, 128, 224, 192, 128, 32, 0, 64, 160, 0, 64, 32, 128, 64, 160, 128, 64, 32, 0, 192, 160, 0, 192, 32, 128, 192, 160, 128, 192, 96, 0, 64, 224, 0, 64, 96, 128, 64, 224, 128, 64, 96, 0, 192, 224, 0, 192, 96, 128, 192, 224, 128, 192, 32, 64, 64, 160, 64, 64, 32, 192, 64, 160, 192, 64, 32, 64, 192, 160, 64, 192, 32, 192, 192, 160, 192, 192, 96, 64, 64, 224, 64, 64, 96, 192, 64, 224, 192, 64, 96, 64, 192, 224, 64, 192, 96, 192, 192, 224, 192, 192, 0, 32, 0, 128, 32, 0, 0, 160, 0, 128, 160, 0, 0, 32, 128, 128, 32, 128, 0, 160, 128, 128, 160, 128, 64, 32, 0, 192, 32, 0, 64, 160, 0, 192, 160, 0, 64, 32, 128, 192, 32, 128, 64, 160, 128, 192, 160, 128, 0, 96, 0, 128, 96, 0, 0, 224, 0, 128, 224, 0, 0, 96, 128, 128, 96, 128, 0, 224, 128, 128, 224, 128, 64, 96, 0, 192, 96, 0, 64, 224, 0, 192, 224, 0, 64, 96, 128, 192, 96, 128, 64, 224, 128, 192, 224, 128, 0, 32, 64, 128, 32, 64, 0, 160, 64, 128, 160, 64, 0, 32, 192, 128, 32, 192, 0, 160, 192, 128, 160, 192, 64, 32, 64, 192, 32, 64, 64, 160, 64, 192, 160, 64, 64, 32, 192, 192, 32, 192, 64, 160, 192, 192, 160, 192, 0, 96, 64, 128, 96, 64, 0, 224, 64, 128, 224, 64, 0, 96, 192, 128, 96, 192, 0, 224, 192, 128, 224, 192, 64, 96, 64, 192, 96, 64, 64, 224, 64, 192, 224, 64, 64, 96, 192, 192, 96, 192, 64, 224, 192, 192, 224, 192, 32, 32, 0, 160, 32, 0, 32, 160, 0, 160, 160, 0, 32, 32, 128, 160, 32, 128, 32, 160, 128, 160, 160, 128, 96, 32, 0, 224, 32, 0, 96, 160, 0, 224, 160, 0, 96, 32, 128, 224, 32, 128, 96, 160, 128, 224, 160, 128, 32, 96, 0, 160, 96, 0, 32, 224, 0, 160, 224, 0, 32, 96, 128, 160, 96, 128, 32, 224, 128, 160, 224, 128, 96, 96, 0, 224, 96, 0, 96, 224, 0, 224, 224, 0, 96, 96, 128, 224, 96, 128, 96, 224, 128, 224, 224, 128, 32, 32, 64, 160, 32, 64, 32, 160, 64, 160, 160, 64, 32, 32, 192, 160, 32, 192, 32, 160, 192, 160, 160, 192, 96, 32, 64, 224, 32, 64, 96, 160, 64, 224, 160, 64, 96, 32, 192, 224, 32, 192, 96, 160, 192, 224, 160, 192, 32, 96, 64, 160, 96, 64, 32, 224, 64, 160, 224, 64, 32, 96, 192, 160, 96, 192, 32, 224, 192, 160, 224, 192, 96, 96, 64, 224, 96, 64, 96, 224, 64, 224, 224, 64, 96, 96, 192, 224, 96, 192, 96, 224, 192, 224, 224, 192]
_DEFAULT_PALETTE = ImagePalette('RGB', np.array(_DEFAULT_PALETTE_COLOR_256, dtype=np.uint8).tobytes())
_DEFAULT_PALETTE.dirty = 1
_DEFAULT_PALETTE.rawmode = 'RGB'

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


    @staticmethod
    def load(path):
        # load doc
        doc = YamlObject.load(path)
        if doc is not Document: return doc

        # load segmentation
        seg_path = os.path.splitext(path)[0] + '.png'
        if os.path.isfile(seg_path):
            image = Image.open(seg_path)
            seg_docs = []
            def _do(o):
                if not o.has('segmentation'): return
                seg_image = image.copy()
                indexes = (seg_image != o.segmentation_id) & (seg_image != 0)
                seg_image[indexes] = 0
                o._cache['__segmentation__'] = seg_image

            doc._search_docs(_do)

        return doc


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
            image = Image.fromarray(im_array, mode='P')
            image.palette = _DEFAULT_PALETTE
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




