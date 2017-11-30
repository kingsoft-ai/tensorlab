from .loader import Loader
from .. import config

class VOCLoder(Loader):
    def __init__(self, root_path, config):
        super(VOCLoder, self).__init__(root_path, config)


    def collect_train_list(self):
        return self._search_files(self.config.images, config.IMAGE_FILE_EXTENSIONS)[0:10]

    def collect_test_list(self):
        return []

    def process(self, file_path): return None