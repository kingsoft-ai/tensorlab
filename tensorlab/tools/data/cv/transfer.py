import argparse
import os
import shutil

import progressbar
from tensorlab.tools.data.cv import config
from tensorlab.tools.data.cv.document import Document


def process_loader(name, loader, output_path):

    # recreate output path
    if os.path.isdir(output_path):
        shutil.rmtree(output_path)

    os.makedirs(output_path)

    # get trains and tests
    trains = loader.collect_train_list()
    tests = loader.collect_test_list()

    # process trains
    def process(f):
        # get label path
        label_path = os.path.join(output_path, f)
        label_path = os.path.splitext(label_path)[0] + '.yml'
        label_dir = os.path.dirname(label_path)
        if not os.path.isdir(label_dir): os.makedirs(label_dir)

        # init doc
        doc = Document()
        doc.dataset = name
        doc.path = f
        doc.seg_tag = False
        doc.box_tag = False

        # process
        loader.process(f, doc)

        # set tag
        doc.seg_tag = doc.search('segmentation_id') is not None
        doc.box_tag = doc.search('box') is not None

        # save
        doc.save(label_path)
        return doc

    train_docs = []
    test_docs = []
    widgets = [progressbar.FormatLabel(name), ' ',
                progressbar.Percentage(), ' ',
                progressbar.Bar('#'), ' ',
                progressbar.RotatingMarker()]

    with progressbar.ProgressBar(max_value=len(trains)+len(tests), widgets=widgets) as bar:
        index = 0
        for i in range(len(trains)):
            train_docs.append(process(trains[i]))
            bar.update(index)
            index += 1

        for i in range(len(tests)):
            test_docs.append(process(tests[i]))
            bar.update(index)
            index+= 1


    # output file list
    doc = Document()
    doc.trains = trains
    doc.tests = tests
    doc.save(os.path.join(output_path, name) + ".yml")



def transfer(data_path, output_path, export_tag):
    # process args
    if output_path is None:
        output_path = os.path.join(data_path, config.DEFAULT_LABEL_PATH)


    # collect loaders
    dataset_configs = []
    for (ds_name, ds_cfg) in config.DATASETS.items():
        if export_tag[ds_name] != True: continue
        dataset_configs.append(ds_cfg)


    # process loader
    for i in range(len(dataset_configs)):
        cfg = dataset_configs[i]
        loader_cls = cfg.loader
        root_path = os.path.join(data_path, cfg.name)
        label_path = os.path.join(output_path, cfg.name)
        loader = loader_cls(root_path, cfg)
        process_loader(cfg.name, loader, label_path)


if __name__ == "__main__":

    # parse args
    def str2bool(v):return v.lower() in ("yes", "true", "t", "1", True)
    parser = argparse.ArgumentParser(prog="python", description="transfer cv dataset")
    parser.add_argument('--data-path', type=str, help='root path for all dataset')
    parser.add_argument('--output-path', type=str, default=None, help='root path for all dataset')
    parser.add_argument('--voc', type=str2bool, default=True, help="transfer voc dataset")
    parser.add_argument('--coco', type=str2bool, default=True, help="transfer coco dataset")
    parser.add_argument('--ade', type=str2bool, default=True, help="transfer ade dataset")
    args = parser.parse_args()

    # data set export tag
    ds_names = config.DATASETS.keys()
    export_tag = {}
    for name in ds_names:
        export_tag[name] = getattr(args, name.lower())


    # transfer
    transfer(args.data_path, args.output_path, export_tag)
