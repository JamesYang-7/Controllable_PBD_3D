from argparse import ArgumentParser
from yacs.config import CfgNode as CN


def get_args(filepath=None):
    parser = ArgumentParser(description='Training parameters')
    parser.add_argument('--cfg',
                        help='configure file name',
                        default="config/example.yaml",
                        type=str)

    if filepath is None:
        cli_args = parser.parse_args()
        filepath = cli_args.cfg
    print(f"using config {filepath}")
    cfg = CN(new_allowed=True)
    cfg.merge_from_file(filepath)
    cfg.filepath = filepath  # Add the cfg file path to the configuration
    return cfg