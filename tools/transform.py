import os
import sys
import argparse
from tensorflow.core.framework import graph_pb2
from tensorflow.python.platform import gfile
import tensorflow.python.tools

FIRST_FILE = "./freeze.pb"
# MIDDLE_FILE = "./optimizer.pb"

def search_node_name(graph):
    input_node_name = []
    tf_graph = graph_pb2.GraphDef()
    with gfile.Open(graph, "rb") as f:
        data = f.read()
        tf_graph.ParseFromString(data)
    for node in tf_graph.node:
        # nodes_map[node.name] = node
        if node.op == "Placeholder" and node.name != "Placeholder":
            input_node_name.append(node.name)
    return input_node_name

def search_tfscript():
    a = sys.modules["tensorflow.python.tools"].__file__
    freeze_path = os.path.abspath(os.path.join(os.path.dirname(a), 'freeze_graph.py'))
    optimizer_path = os.path.abspath(os.path.join(os.path.dirname(a), 'optimize_for_inference.py'))
    return freeze_path, optimizer_path

def freeze_graph(input_graph, input_checkpoint, output_graph, output_node_names, input_binary):
    cmd = "python {} ".format(search_tfscript()[0])
    cmd += "--input_binary {} " .format(input_binary)
    cmd += "--input_graph {} ".format(input_graph)
    cmd += "--input_checkpoint {} ".format(input_checkpoint)
    cmd += "--output_graph {} ".format(output_graph)
    cmd += "--output_node_names {} ".format(output_node_names)
    if os.path.exists(output_graph): os.remove(output_graph)
    print(cmd)
    os.system(cmd)

def optimizer_graph(input_graph, output_graph, input_names, output_node_names):
    cmd = "python {} ".format(search_tfscript()[1])
    cmd += "--input {} ".format(input_graph)
    cmd += "--output {} ".format(output_graph)
    cmd += "--input_names {} ".format(input_names)
    cmd += "--output_names {} ".format(output_node_names)
    if os.path.exists(output_graph): os.remove(output_graph)
    print(cmd)
    os.system(cmd)

# def quantize_graph(input_graph, output_graph, output_node_names, mode):
#     cmd = "python quantize_graph.py "
#     cmd += "--input {} ".format(input_graph)
#     cmd += "--output_node_names {} ".format(output_node_names)
#     cmd += "--output {} ".format(output_graph)
#     cmd += "--mode {} ".format(mode)
#     if os.path.exists(output_graph): os.remove(output_graph)
#     print(cmd)
#     os.system(cmd)

def do(args):
    input_graph = args.input_graph
    input_checkpoint = args.input_checkpoint
    output_graph = args.output_graph
    input_names = args.input_names
    output_node_names = args.output_node_names
    # mode = args.mode
    input_binary = args.input_binary

    if input_names is None:
        input_names = search_node_name(input_graph)
        input_names = '\"{}\"'.format(",".join(input_names))
    print("your input_node_name is: ", input_names)
    print("your output_node_name is: ", output_node_names)
    freeze_graph(input_graph,input_checkpoint,FIRST_FILE,output_node_names,input_binary)
    if not os.path.isfile(FIRST_FILE): raise Exception('freeze graph failed')
    optimizer_graph(FIRST_FILE, output_graph, input_names, output_node_names)
    # if not os.path.isfile(MIDDLE_FILE): raise Exception('optimizer for inference failed')
    # quantize_graph(MIDDLE_FILE, output_graph, output_node_names, mode)







if __name__ == "__main__":

    def str2bool(v):return v.lower() in ("yes", "true", "t", "1", True)
    parser = argparse.ArgumentParser(prog="python", description="auto convert pb to ios model")
    parser.add_argument('--input_graph', type=str, help='tensorflow graph path for excute')
    parser.add_argument('--input_checkpoint', type=str, help='tensorflow variables model path')
    parser.add_argument('--output_graph', type=str, help='output file name and path, comma separated')
    parser.add_argument('--output_node_names', type=str, help='The name of the output nodes, comma separated')
    parser.add_argument('--input_names', type=str, default=None, help='Input node names, comma separated')
    # parser.add_argument('--mode', type=str, default="weights_rounded", help='What transformation to apply (round, quantize,eightbit, weights, or weights_rounded)')
    parser.add_argument('--input_binary',nargs="?",const=False,type=str2bool,default=True,help="Whether the input files are in binary format")

    args = parser.parse_args()

    # do
    try:
        do(args)

    except Exception as e:
        parser.print_help()
        print(e)
