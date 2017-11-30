import os
import sys
import argparse
from . import config
from .loader import VOCLoder, COCLoder



def process_loader(loader, output_path):
    # create output path
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    # get trains and tests
    trains = loader.collect_train_list()
    tests = loader.collect_test_list()





def transfer(args):
    # process args
    output_path = args.output_path
    if output_path is None:
        output_path = os.path.join(args.data_path, 'Labels')


    # collect loaders
    configs = []
    loaders = []

    if args.all or args.voc:
        configs.append(config.VOC)
        loaders.append(VOCLoder)

    if args.all or args.coc:
        configs.append(config.COC)
        loaders.append(COCLoder)


    for i in range(len(configs)):
        cfg = configs[i]
        loader_cls = loaders[i]
        root_path = os.path.join(args.data_path, cfg.name)
        label_path = os.path.join(output_path, cfg.name)
        loader = loader_cls(root_path, cfg)
        process_loader(loader, label_path)






if __name__ == "__main__":

    def str2bool(v):return v.lower() in ("yes", "true", "t", "1", True)
    parser = argparse.ArgumentParser(prog="python", description="transfer cv dataset")
    parser.add_argument('--data-path', type=str, help='root path for all dataset')
    parser.add_argument('--output-path', type=str, default=None, help='root path for all dataset')
    parser.add_argument('--all', type=str2bool, default=True, help="transfer all dataset")
    parser.add_argument('--voc', type=str2bool, default=None, help="transfer voc dataset")
    parser.add_argument('--coc', type=str2bool, default=None, help="transfer coc dataset")

    args = parser.parse_args()

    transfer(args)