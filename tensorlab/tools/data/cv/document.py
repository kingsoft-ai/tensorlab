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
#   segmentation_id: a id described a value in each pixel of segmentation picture
#   pose: person pose name

####################################################################################################

import os
import numpy as np
from PIL import Image
from PIL.ImagePalette import ImagePalette
from tensorlab.utils.yaml_object import YamlObject
from tensorlab.image import DEFAULT_PALETTE_COLOR_256

# set output palette
_DEFAULT_PALETTE = ImagePalette('RGB', np.array(DEFAULT_PALETTE_COLOR_256, dtype=np.uint8).tobytes())
#_DEFAULT_PALETTE.dirty = 1
_DEFAULT_PALETTE.rawmode = 'RGB'

class Document(YamlObject):
    def __init__(self):
        super(YamlObject, self).__init__()
        self.__dict__['__cache__'] = {}

    @property
    def _cache(self): return self.__dict__['__cache__']

    @property
    def segmentation_id(self): return self.__get_attr__('segmentation_id')

    @property
    def segmentation(self): return self.segmentation_getter()

    def segmentation_getter(self):
        return self._cache.get('__segmentation__')

    def segmentation_setter(self, seg_image):
        seg_image = seg_image.astype(np.uint8)
        self._cache['__segmentation__'] = seg_image
        self.__set_attr__('segmentation_id', int(seg_image.max()))


    @staticmethod
    def load(path):
        # load doc
        doc = YamlObject.load(path, __class__)

        # load segmentation
        seg_path = os.path.splitext(path)[0] + '.png'
        if os.path.isfile(seg_path):
            image = Image.open(seg_path)
            seg_image = np.array(image)
            seg_docs = []
            def _do(o):
                if not o.has('segmentation_id'): return
                seg = seg_image.copy()
                indexes = (seg != o.segmentation_id) & (seg != 0)
                seg[indexes] = 0
                o._cache['__segmentation__'] = seg

            doc._search_docs(_do)

        return doc


    def save(self, path):
        # save yml
        super(Document, self).save(path)

        # collect datas
        segs = self.search_all('__segmentation__', in_cache=True)

        # save segmentation
        if len(segs) > 0:
            seg_path = os.path.splitext(path)[0] + '.png'
            im_array = np.stack(segs, axis=2).sum(axis=2)
            image = Image.fromarray(im_array.astype(np.uint8), mode='P')
            image.putpalette(_DEFAULT_PALETTE)
            image.save(seg_path)



    def search(self, key, in_cache=False):
        result = None
        def _do(o):
            nonlocal result
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




