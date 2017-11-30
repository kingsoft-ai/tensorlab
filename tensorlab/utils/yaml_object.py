import yaml
from collections import OrderedDict

def represent_ordereddict(self, data):
    return self.represent_mapping(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, data.items())

def create_dumper(cls):
    class Dumper(yaml.Dumper):
        def __init__(self, *args, **kwargs):
            yaml.Dumper.__init__(self, *args, **kwargs)
            self.add_representer(cls, type(self).represent_ordereddict)

        represent_ordereddict = represent_ordereddict

    return Dumper


class YamlObject(OrderedDict):
    def __setattr__(self, key, value): self.__set_attr__(key, value)

    def __getattr__(self, item): self.__get_attr__(item)

    def __set_attr__(self, key, value):
        if key in dict.keys(self):
            raise Exception("repeated setting for key {}".format(key))
        self[key] = value

    def __get_attr__(self, key): return self[key]

    def save(self, path):
        dumper = create_dumper(type(self))
        data = yaml.dump(self, Dumper=dumper)
        with open(path, 'w') as f:
            f.write(data)
        del data

