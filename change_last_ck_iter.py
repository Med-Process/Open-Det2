import os

import argparse


def save_last_chackpoint(path):
    files_name = "last_checkpoint"
    dir_name = os.path.dirname(path)
    base_dir = os.path.basename(path)    # 迭代
    save_file = os.path.join(dir_name, files_name)
    try:
        with open(save_file, "w") as file:
            file.write(base_dir)
    except IOError:
        # if file doesn't exist, maybe because it has just been
        # deleted by a separate process
        print("Wrong with last_checkpoint")


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--iter_model', type=str, default="", help="the iteration of the model")
    parser.add_argument(
        "--local-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )
    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    iter_model = args.iter_model

    save_last_chackpoint(iter_model)
    print("save file name of", os.path.basename(iter_model))