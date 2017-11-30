from .loader import Loader


class VOCLoder(Loader):
    def __init__(self, root_path, config):
        super(VOCLoder, self).__init__(root_path, config)