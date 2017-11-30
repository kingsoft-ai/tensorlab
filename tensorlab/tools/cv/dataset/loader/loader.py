import os


class Loader(object):
    def __init__(self, root_path, config):
        self.root_path = root_path
        self.config = config


    def collect_train_list(self): raise NotImplementedError("collect_train_list need to implement")

    def collect_test_list(self): raise NotImplementedError("collect_test_list need to implement")


    def _search_files(self, relative_dir, exts=[]):
        path = os.path.join(self.root_path, relative_dir)

        files = []
        for rt, dirs, files in os.walk(path):
            for f in files:
                ext = os.path.splitext(f)[-1]
                if ext not in exts: continue
                files.append(f.split(self.root_path)[1:])

        return files

