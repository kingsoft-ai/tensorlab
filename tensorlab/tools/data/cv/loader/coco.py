from .loader import Loader
#from pycocotools.coco import COCO
#from pycocotools import mask

class COCOLoder(Loader):
    def __init__(self, root_path, config):
        super(COCOLoder, self).__init__(root_path, config)
        cfg = config
        self.year = cfg.year
        self.split = cfg.split
        self.root = root_path
        if 'test' in self.split:
            json = '%s/annotations/image_info_%s.json'
        else:
            json = '%s/annotations/instances_%s.json'


    def read_annotations(self, name):pass

    def read_segmentations(self, name):pass

    def collect_train_list(self):

        self.coco = COCO(json % self.root, self.year)

    def collect_test_list(self):pass

    def process(self, file_path):pass

