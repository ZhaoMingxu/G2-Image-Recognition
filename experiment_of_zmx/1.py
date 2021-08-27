import glob
from pathlib import Path
from functools import partial
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from torchvision import transforms
from torchvision.transforms import functional
from PIL import Image
import torch
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from utils import inf_loop
from model import metric as module_metric
from manuscript2 import mixup_data

import tow



p = glob.glob('E:\Pycharm_docu\pictures\*.jpg')
# # phase = "train"
transform = transforms.Compose([
    # transforms.CenterCrop(1000),
    # transforms.Resize(256),
    # transforms.RandomCrop(224),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    transforms.ToPILImage()
])
# # phase=tta
# # transform = transforms.Compose([
# #     transforms.CenterCrop(1000),
# #     transforms.Resize(256),
# #     transforms.RandomCrop(224),
# #     transforms.RandomHorizontalFlip(),
# #     transforms.ToTensor(),
# #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
# #     transforms.ToPILImage()
# # ])
# # phase=test
# transform = transforms.Compose([
#     # transforms.CenterCrop(1000),
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#     transforms.ToPILImage()]
# )
image1 = Image.open(p[0])
after1 = transform(image1)
image2 = Image.open(p[1])
after2 = transform(image2)
image3 = Image.open(p[2])
after3 = transform(image3)
image4 = Image.open(p[3])
after4 = transform(image4)
image5 = Image.open(p[4])
after5 = transform(image5)
after1.show()
after2.show()
after3.show()
after4.show()
after5.show()
