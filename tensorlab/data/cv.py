import os
import math
import multiprocessing
import numpy as np
from tensorlab.data.dataset import Dataset
from tensorlab.tools.data.cv.config import DEFAULT_LABEL_PATH, DATASETS
from tensorlab.tools.data.cv.document import Document


class DatasetCV(Dataset):
    def __init__(self,
                 data_path,
                 label_path=None,
                 num_workers=multiprocessing.cpu_count(),
                 datasets = None,
                 max_cache = np.infty,
                 shuffle = True,
                 filter = None,
                 ):

        super(DatasetCV).__init__()

        # base attributes
        self._data_path = data_path
        self._label_path = label_path if label_path is not None else os.path.join(data_path, DEFAULT_LABEL_PATH)
        self._num_workers = num_workers
        self._datasets = datasets if datasets is not None else self._search_datasets()
        self._max_cache = max_cache
        self._shuffle = shuffle
        self._filter = filter
        self._pool = multiprocessing.Pool(num_workers)

        # caches
        self._cache_images = []
        self._cache_labels = []
        self._all_labels = []

        # load labels
        self._load_labels()


    def load(self):
        pass


    def _search_datasets(self):
        datasets = []
        for ds in DATASETS.keys():
            dataset_path = os.path.join(self._data_path, ds)
            index_path = os.path.join(self._label_path, ds, ds+'.yml')
            if not os.path.isdir(dataset_path) or not os.path.isfile(index_path): continue
            datasets.append(ds)
        return datasets


    def _load_labels(self):
        import time
        st = time.time()
        # load index files
        train_doc_paths, test_doc_paths = [], []
        results = self._parallelize(DatasetCV._job_load_docs,
                                    [os.path.join(self._label_path, name, name+'.yml') for name in self._datasets])

        docs = [doc for docs in results for doc in docs]
        for i in range(len(self._datasets)):
            ds_name = self._datasets[i]
            doc = docs[i]
            train_doc_paths += [os.path.join(self._label_path, ds_name, p) for p in doc.trains]
            test_doc_paths += [os.path.join(self._label_path, ds_name, p) for p in doc.tests]

        # load docs
        train_results = self._parallelize(DatasetCV._job_load_docs, train_doc_paths)
        test_results = self._parallelize(DatasetCV._job_load_docs, test_doc_paths)
        train_docs = [doc for docs in train_results for doc in docs]
        test_docs = [doc for docs in test_results for doc in docs]

        print(len(train_docs))
        print(len(test_docs))

        print(time.time() - st)



    @staticmethod
    def _job_load_docs(doc_paths):
        docs = []
        for doc_path in doc_paths:
            doc = Document.load(doc_path)
            docs.append(doc)
        return docs


    def _parallelize(self, func, datas, *args):
        loads_per_worker = math.ceil(len(datas) / self._num_workers)
        results = []
        for i in range(self._num_workers):
            st = i * loads_per_worker
            ed = min(st + loads_per_worker, len(datas))
            if st > len(datas): break
            data = datas[st:ed]
            results.append(self._pool.apply_async(func, (data,) + args))
        return [r.get() for r in results]




if __name__ == '__main__':
    # parse args
    import argparse
    def str2bool(v):return v.lower() in ("yes", "true", "t", "1", True)
    parser = argparse.ArgumentParser(prog="python", description="transfer cv dataset")
    parser.add_argument('--data-path', type=str, required=True, help='dataset path')
    args = parser.parse_args()

    # create dataset
    dataset = DatasetCV(args.data_path)


