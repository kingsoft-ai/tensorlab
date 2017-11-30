from .loader import Loader


class COCLoder(Loader):
    def __init__(self, root_path, config):
        super(COCLoder, self).__init__(root_path, config)