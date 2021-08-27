import json
from utils.util import read_json, write_json
from pathlib import Path
import glob
from data_loader.g2dataset import G2Dataset, G2Transforms


def path_json():
    """
    :param
    """
    path = "prediction_json/806"
    data = glob.glob(path + "/*.json")
    for k in range(2):
        name = ['A', 'B']
        path = data[k]
        file = read_json(path)
        output_result = {}
        types = ['single', 'double', 'triple', 'quadruple', 'T-shape']
        for i, ima_key in enumerate(file):
            for j, ty in enumerate(types):
                if file[ima_key][ty] > 0.96:
                    output_result[ima_key] = ty

        write_json(output_result, 'pseudo_json/pseudo806{}.json'.format(name[k]))


if __name__ == "__main__":
    path_json()
