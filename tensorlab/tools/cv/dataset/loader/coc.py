from .loader import Loader


class COCLoder(Loader):
    def __init__(self, root_path, config):
        super(COCLoder, self).__init__(root_path, config)
        cfg = config
        self.year = cfg.year
        self.split = cfg.split
        assert cfg.year in ['07', '12'], "wrong year"
        self.root = root_path


    def read_annotations(self, name):pass

    def read_segmentations(self, name):pass

    def collect_train_list(self):pass

    def collect_test_list(self):pass

    def process(self, file_path):pass

