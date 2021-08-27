import json
from collections import OrderedDict
from pathlib import Path

import torch
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold


# 写入json文件
# dict1 = {
#     'test': 1,
#     'test2': {
#         'o1': 10,
#         '02': 20
#     }
# }
# p = Path('zmx.json')
# file = p.open('wt')
# json.dump(dict1, file, indent=4)
#
# file = p.open('rt')
# j = json.load(file)
# print(type(j))
# print(j)

class student():
    def __init__(self, name, age):
        self.name = name
        self._age = age

    def __getitem__(self, item):
        if isinstance(item, str):
            return getattr(self, item)


x = np.array([
    [1, 2, 3, 4],
    [11, 12, 13, 14],
    [21, 22, 23, 24],
    [31, 32, 33, 34],
    [41, 42, 43, 44],
    [51, 52, 53, 54],
    [61, 62, 63, 64],
    [71, 72, 73, 74],
    [81, 82, 83, 84],
    [91, 92, 93, 94],
])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1])

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=123)

skf = StratifiedKFold(n_splits=5, random_state=123, shuffle=True)
for i, (train_index, test_index) in enumerate(skf.split(x, y)):
    x_train = [x[i] for i in train_index]
    x_test = [x[i] for i in test_index]
    y_train = [y[i] for i in train_index]
    y_test = [y[i] for i in test_index]

print(len(x_train))
for i in skf.split(x, y):
    print(i)
