import argparse
import collections
from pathlib import Path
from utils import read_json, write_json
CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]

args = argparse.ArgumentParser(description='trail-7.23')
args.add_argument('-c', '--config', default=None, type=str,
                  help='config.json file path (default: None)')
for opt in options:
    args.add_argument(*opt.flags, default=None, type=opt.type)


args = args.parse_args()
cfg_name = Path(args.config)
config = read_json(cfg_name)
print(config)



