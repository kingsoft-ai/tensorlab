import os
import shutil
import argparse
import progressbar
from tensorlab.tools.cv.dataset import config
from tensorlab.tools.cv.dataset.loader import VOCLoder, COCLoder
from tensorlab.tools.cv.dataset.document import Document


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

        for i in range(len(test_docs)):
            train_docs.append(process(test_docs[i]))
            bar.update(index)
            index+= 1


    # output file list
    doc = Document()
    doc.trains = trains
    doc.tests = tests
    doc.save(os.path.join(output_path, name) + ".yml")



def transfer(args):
    # process args
    output_path = args.output_path
    if output_path is None:
        output_path = os.path.join(args.data_path, config.DEFAULT_LABEL_PATH)


    # check export dataset
    export_all = False
    if  args.voc == None and \
        args.coc == None:
        export_all = True

    # collect loaders
    configs = []
    loaders = []

    if export_all or args.voc:
        configs.append(config.VOC)
        loaders.append(VOCLoder)

    if export_all or args.coc:
        configs.append(config.COC)
        loaders.append(COCLoder)


    # process loader
    for i in range(len(configs)):
        cfg = configs[i]
        loader_cls = loaders[i]
        root_path = os.path.join(args.data_path, cfg.name)
        label_path = os.path.join(output_path, cfg.name)
        loader = loader_cls(root_path, cfg)
        process_loader(cfg.name, loader, label_path)






if __name__ == "__main__":

    def str2bool(v):return v.lower() in ("yes", "true", "t", "1", True)
    parser = argparse.ArgumentParser(prog="python", description="transfer cv dataset")
    parser.add_argument('--data-path', type=str, help='root path for all dataset')
    parser.add_argument('--output-path', type=str, default=None, help='root path for all dataset')
    parser.add_argument('--voc', type=str2bool, default=None, help="transfer voc dataset")
    parser.add_argument('--coc', type=str2bool, default=None, help="transfer coc dataset")

    args = parser.parse_args()

    transfer(args)
