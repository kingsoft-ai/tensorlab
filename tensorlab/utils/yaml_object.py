import yaml
from collections import OrderedDict
from types import MethodType

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
    def __init__(self, parent=None):
        super(YamlObject, self).__init__()
        self.__dict__['__cache__'] = {}
        self.__dict__['__parent__'] = parent
        self.__dict__['__childs__'] = []
        self.__dict__['__root__'] = None if parent is None else parent.__dict__['__root__']

    @property
    def parent(self): return self.__dict__['__parent__']

    @property
    def childs(self): return self.__dict__['__childs__']

    @property
    def root(self): return self.__dict__['__root__']

    @property
    def _cache(self): return self.__dict__['__cache__']

    def __setattr__(self, key, value):
        setter = '{}_setter'.format(key)
        f = getattr(self, setter)
        if type(f) is MethodType:
            f(value)
        else:
            self.__set_attr__(key, value)

    def __getattr__(self, item):
        return self.__get_attr__(item)


    def __set_attr__(self, key, value):
        if key in dict.keys(self):
            raise Exception("repeated setting for key {}".format(key))
        self[key] = value

    def __get_attr__(self, key): return self.get(key, None)

    def child(self):
        child = type(self)(parent=self)
        self.childs.append(child)
        return child

    def has(self, k): return k in dict.keys(self)

    def save(self, path):
        data = self.dump()
        with open(path, 'w') as f:
            f.write(data)
        del data

    def dump(self):
        dumper = create_dumper(type(self))
        return yaml.dump(self, Dumper=dumper)




