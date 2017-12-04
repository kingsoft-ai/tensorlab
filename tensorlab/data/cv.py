import multiprocessing

class DatasetCV(object):
    def __init__(self, data_path, num_workers=multiprocessing.cpu_count()):
        self._data_path = data_path
        self._num_workers = num_workers
        self.pool = multiprocessing.Pool(num_workers)


    def load(self):
        pass








