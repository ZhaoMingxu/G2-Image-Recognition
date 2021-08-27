from PIL import Image
import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


image = Image.open('E:\G2\data\images_test_A\image000000.jpg')
image = np.array(image)

transform1 = transforms.ToTensor()
image = transform1(image)

y = torch.tensor([0])
mixed_data, y_a, y_b, lam = mixup_data(image, y=y, use_cuda=False)
transform2 = transforms.ToPILImage()
