from .loader import Loader


class COCLoder(Loader):
    def __init__(self, root_path, config):
        super(COCLoder, self).__init__(root_path, config)



    def read_segmentations(self, name):pass

    def collect_train_list(self):pass

    def collect_test_list(self):pass

    def process(self, file_path):pass

