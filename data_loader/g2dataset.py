from torch.utils.data import Dataset
from torchvision import transforms
# from skimage import io
# import cv2
import glob
import os
import torch
import numpy as np
import random
from random import shuffle
from torch.utils.data import DataLoader
import torchvision
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from PIL import Image
from utils.util import read_json

random.seed(123)


class G2Transforms(object):
    def __init__(self, output_size, phase):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.phase = phase
        if self.phase == "train" or self.phase == "all":
            self.transform = transforms.Compose([
                # transforms.CenterCrop(1000),
                # transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomResizedCrop(224, scale=(0.3, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        elif self.phase == "tta":
            self.transform = transforms.Compose([
                #                transforms.CenterCrop(1000),
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        else:
            self.transform = transforms.Compose([
                #                transforms.CenterCrop(1000),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __call__(self, sample):
        return self.transform(sample)


class G2Dataset(Dataset):
    def __init__(self, root_dir, phase, ratio=0.2, transform=None, kfold=None, nfold=None, fold=None, pseudo=False):
        self.root_dir = root_dir
        # types = os.listdir(os.path.join(self.root_dir,'images_train_5s'))
        types = ['单柱顺转1-1', '双柱顺转1-1', '三柱顺转1-1', '四柱顺转1-1', 'T型顺转1-1']
        x = []
        y = []
        for i, t in enumerate(types):
            pictures = glob.glob(self.root_dir + '/images_train_5s/' + t + '/*.jpg')
            x.extend(pictures)
            y.extend([i] * len(pictures))
        if pseudo:
            for k in range(2):
                name = ['A', 'B']
                json_file = read_json(self.root_dir + 'pseudo{}.json'.format(name[k]))
                labels = {'single': 0, 'double': 1, 'triple': 2, 'quadruple': 3, 'T-shape': 4}
                for i, img_key in enumerate(json_file):
                    data_path = self.root_dir + '/images_test_{}/'.format(name[i])
                    label_key = json_file[img_key]
                    label = labels[label_key]
                    data_path = data_path + (img_key + '.jpg')
                    x.extend([data_path])
                    y.extend([label])

        if not kfold:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=ratio, stratify=y, random_state=123)
        else:
            skf = StratifiedKFold(n_splits=nfold, random_state=123, shuffle=True)
            for i, (train_index, test_index) in enumerate(skf.split(x, y)):
                if i != fold:
                    continue
                # print(i,"TRAIN:", train_index, "TEST:", test_index)
                x_train = [x[i] for i in train_index]
                x_test = [x[i] for i in test_index]
                y_train = [y[i] for i in train_index]
                y_test = [y[i] for i in test_index]

        if phase == "train":
            self.data = x_train
            self.labels = y_train

        elif phase == "val":
            self.data = x_test
            self.labels = y_test
        elif phase == "all":
            self.data = x
            self.labels = y
        else:
            self.data = glob.glob(self.root_dir + 'images_test_B/*.jpg')
            self.labels = [0] * len(self.data)  # not true label

        self.transform = transform(224, phase)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        elif type(idx) is list:
            print("error")
            return
        images = Image.open(self.data[idx])
        if self.transform:
            images = self.transform(images)
        else:
            images = torch.from_numpy(np.asarray(images))

        label = self.labels[idx]
        # sample = {"image": images, "label": torch.tensor(label)}
        sample = (images, torch.tensor(label))
        return sample

    def types(self):
        types = os.listdir(os.path.join(self.root_dir, 'images_train_5s'))
        return types

    def datum(self):
        return self.data


if __name__ == "__main__":
    g2train = G2Dataset(root_dir="../", phase="train", transform=G2Transforms, pseudo=True)
