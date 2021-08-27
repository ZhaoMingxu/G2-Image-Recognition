import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import model.scheduler as scheduler
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device
import random
from data_loader.g2dataset import G2Dataset, G2Transforms
from torch.utils.data import DataLoader

# from model.resnet import resnet50,resnet18,resnet34

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)


def main(config, fold=None):
    logger = config.get_logger('train')
    if fold is not None:
        logger.info("fold {} begin ......".format(fold))
    # setup dataset instances
    train_set = G2Dataset(root_dir=config["data_loader"]["args"]["data_dir"],
                          phase=config["data_loader"]["args"]["stage"],
                          ratio=config["data_loader"]["args"]["validation_split"],
                          transform=G2Transforms, kfold=config["kfold"],
                          nfold=config["data_loader"]["args"]["fold"],
                          fold=fold)
    val_set = G2Dataset(root_dir=config["data_loader"]["args"]["data_dir"],
                        phase="val",
                        ratio=config["data_loader"]["args"]["validation_split"],
                        transform=G2Transforms, kfold=config["kfold"],
                        nfold=config["data_loader"]["args"]["fold"],
                        fold=fold)

    # setup data_loader instances
    train_loader = DataLoader(train_set,
                              batch_size=config["data_loader"]["args"]["batch_size"],
                              num_workers=config["data_loader"]["args"]["num_workers"],
                              shuffle=config["data_loader"]["args"]["shuffle"],
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set,
                            batch_size=config["data_loader"]["args"]["batch_size"],
                            num_workers=config["data_loader"]["args"]["num_workers"],
                            shuffle=False, pin_memory=True, drop_last=False)
    # data_loader = config.init_obj('data_loader', module_data)
    # valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    # model = resnet34(pretrained = True)
    model.fc = torch.nn.Linear(model.fc.in_features, config["n_classes"])
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    # criterion = getattr(module_loss, config['loss'])
    criterion = torch.nn.CrossEntropyLoss()
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    # lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    # optimizer = torch.optim.SGD(lr=config["optimizer"]["args"]["lr"],
    #                            weight_decay=config["optimizer"]["args"]["weight_decay"],
    #                            momentum=config["optimizer"]["args"]["momentum"],
    #                            params=trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', scheduler, optimizer)
    # lr_scheduler = scheduler.CosineAnnealingWarmupRestarts(optimizer,
    #                first_cycle_steps=100, cycle_mult=1.0, max_lr=0.01, 
    #                min_lr=0.001, warmup_steps=10, gamma=1.0)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=train_loader,
                      valid_data_loader=val_loader,
                      lr_scheduler=lr_scheduler, fold=fold)
    trainer.train()
    # if config["pseudo"]:


def kfold_main(config):
    for i in range(config["data_loader"]["args"]["fold"]):
        main(config, fold=i)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    if config["kfold"]:
        kfold_main(config)
    else:
        main(config)
