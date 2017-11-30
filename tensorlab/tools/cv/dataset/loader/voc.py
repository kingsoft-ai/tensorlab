from .loader import Loader
import os


class VOCLoder(Loader):
    def __init__(self, root_path, config):
        super(VOCLoder, self).__init__(root_path, config)
        cfg = config
        assert cfg.year in ['07','12'] and cfg.split in ['train','val','test']
        self.root = os.path.join(root_path, 'VOCdevkit/VOC20%s/' % cfg.year)
        self.segmentation = False
        if self.segmentation:
            filelist = 'ImageSets/Segmentation/%s.txt'

        else:
            filelist = 'ImageSets/Main/%s.txt'

        with open(os.path.join(self.root, filelist % self.split), 'r') as f:
            self.filenames = f.read().split('\n')[:-1]


    def collect_train_list(self):
        filelist = 'ImageSets/Main/%s.txt'
        with open(os.path.join(self.root, filelist % self.split), 'r') as f:
            self.filenames = f.read().split('\n')[:-1]

    def collect_test_list(self):pass

    def process(self, file_path, doc):pass
