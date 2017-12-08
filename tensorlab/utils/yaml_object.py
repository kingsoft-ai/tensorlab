import yaml
from collections import OrderedDict
from types import MethodType


def create_loader(cls):
    def _construct_yaml_map(self, node):
        data = cls()
        yield data
        value = self.construct_mapping(node)
        data.update(value)

    def _construct_mapping(self, node, deep=False):
        if isinstance(node, yaml.MappingNode):
            self.flatten_mapping(node)
        else:
            msg = 'expected a mapping node, but found %s' % node.id
            raise yaml.constructor.ConstructError(None,
                                                  None,
                                                  msg,
                                                  node.start_mark)

        mapping = cls()
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                hash(key)
            except TypeError as err:
                raise yaml.constructor.ConstructError(
                    'while constructing a mapping', node.start_mark,
                    'found unacceptable key (%s)' % err, key_node.start_mark)
            value = self.construct_object(value_node, deep=deep)
            mapping[key] = value
        return mapping

    class Loader(yaml.Loader):
        def __init__(self, *args, **kwargs):
            yaml.Loader.__init__(self, *args, **kwargs)

            self.add_constructor(
                'tag:yaml.org,2002:map', type(self).construct_yaml_map)
            self.add_constructor(
                'tag:yaml.org,2002:omap', type(self).construct_yaml_map)

        construct_yaml_map = _construct_yaml_map
        construct_mapping = _construct_mapping

    return Loader


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

    def __setattr__(self, key, value):
        setter = '{}_setter'.format(key)
        f = getattr(self, setter)
        if type(f) is MethodType:
            f(value)
        else:
            self.__set_attr__(key, value)

    def __getattr__(self, item):
        return self.__get_attr__(item)


    def __setstate__(self, state): self.__dict__.update(state)


    def __set_attr__(self, key, value):
        self[key] = value

    def __get_attr__(self, key): return self.get(key, None)

    @staticmethod
    def load(path, cls=None):
        if cls is None: cls = __class__
        with open(path) as f:
            loader = create_loader(cls)
            return yaml.load(f, Loader=loader)

    def has(self, k): return k in dict.keys(self)

    def save(self, path):
        data = self.dump()
        with open(path, 'w') as f:
            f.write(data)
        del data

    def dump(self):
        dumper = create_dumper(type(self))
        return yaml.dump(self, Dumper=dumper)




