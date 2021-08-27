import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
import json
from utils import read_json, write_json
from data_loader.g2dataset import G2Dataset, G2Transforms
from torch.utils.data import DataLoader
from model.resnet import resnet50
import random


def main(config, fold=None):
    logger = config.get_logger('test')

    # setup data_loader instances
    #    data_loader = getattr(module_data, config['data_loader']['type'])(
    #        config['data_loader']['args']['data_dir'],
    #        batch_size=512,
    #        shuffle=False,
    #        validation_split=0.0,
    #        training=False,
    #        num_workers=2
    #    )
    if config["data_loader"]["args"]["tta"]:
        test_set = G2Dataset(root_dir=config["data_loader"]["args"]["data_dir"],
                             phase="tta",
                             ratio=config["data_loader"]["args"]["validation_split"],
                             transform=G2Transforms)
    else:
        test_set = G2Dataset(root_dir=config["data_loader"]["args"]["data_dir"],
                             phase="test",
                             ratio=config["data_loader"]["args"]["validation_split"],
                             transform=G2Transforms)
    test_loader = DataLoader(test_set,
                             batch_size=config["data_loader"]["args"]["batch_size"],
                             num_workers=config["data_loader"]["args"]["num_workers"],
                             shuffle=False, pin_memory=True, drop_last=False)

    # build model architecture
    # model = config.init_obj('arch', module_arch)
    model = config.init_obj('archtest', module_arch)
    # model = resnet34(pretrained = True)
    model.fc = torch.nn.Linear(model.fc.in_features, config["n_classes"])
    logger.info(model)

    # get function handles of loss and metrics
    # loss_fn = getattr(module_loss, config['loss'])
    loss_fn = torch.nn.CrossEntropyLoss()
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    ntta = config["data_loader"]["args"]["ntta"] if config["data_loader"]["args"]["tta"] else 1
    tta_ouptut = []
    with torch.no_grad():
        for itta in range(ntta):
            for i, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)

                if i == 0:
                    tot_output = output.clone()
                else:
                    tot_output = torch.cat((tot_output, output), 0)
                #
                # save sample images, or do something with output here
                #

                # computing loss, metrics on test set
                loss = loss_fn(output, target)
                batch_size = data.shape[0]
                total_loss += loss.item() * batch_size
                for i, metric in enumerate(metric_fns):
                    total_metrics[i] += metric(output, target) * batch_size

            tot_output = torch.nn.functional.softmax(tot_output, dim=1)
            tta_ouptut.append(tot_output)
            del tot_output
    total_loss = total_loss / ntta
    for i in range(len(metric_fns)):
        total_metrics[i] = total_metrics[i] / ntta
    tta_ouptut = torch.stack(tta_ouptut)
    tta_output = torch.mean(tta_ouptut, 0)

    output_result = {}
    datum = test_set.datum()
    types = ["single", "double", "triple", "quadruple", "T-shape"]
    # tot_output=torch.nn.functional.softmax(tot_output,dim=1)
    tot_argmax = torch.argmax(tta_output, dim=1)
    tta_output = tta_output.cpu().detach().numpy()
    tot_argmax = tot_argmax.cpu().detach().numpy()
    # ['T型顺转1-1', '三柱顺转1-1', '单柱顺转1-1', '双柱顺转1-1', '四柱顺转1-1']
    for i, da in enumerate(datum):
        # print(da)
        # print(da.split('images_test_B')[-1])
        img_key = da.split('images_test_B/')[-1].split('.jpg')[0]
        prob = {}
        for j, ty in enumerate(types):
            prob[ty] = tta_output[i, j].astype(float)
        output_result[img_key] = prob

    fold = fold if fold is not None else ''
    write_json(output_result, config.save_dir / 'prediction{}.json'.format(fold))

    output_result = {}
    for i, da in enumerate(datum):
        img_key = da.split('images_test_B/')[-1].split('.jpg')[0]
        prob = {}
        for j, ty in enumerate(types):
            prob[ty] = 0.0
        prob[types[tot_argmax[i]]] = 1.0
        output_result[img_key] = prob

    write_json(output_result, config.save_dir / 'prediction_argmax{}.json'.format(fold))

    n_samples = len(test_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


def kfold_main(config):
    ensemble_result = {}
    resume = config.resume
    for i in range(config["data_loader"]["args"]["fold"]):
        config.resume = resume / str("{}/model_best.pth".format(i))
        main(config, fold=i)
        with open(config.save_dir / 'prediction{}.json'.format(i), 'r') as f:
            result_now = json.load(f)

        if i == 0:
            for img in result_now:
                ensemble_result[img] = result_now[img]
        else:
            for img in result_now:
                for ty in result_now[img]:
                    ensemble_result[img][ty] += result_now[img][ty]
    for img in ensemble_result:
        for ty in ensemble_result[img]:
            ensemble_result[img][ty] /= config["data_loader"]["args"]["ntta"]
    write_json(ensemble_result, config.save_dir / 'prediction_kfold.json')
    # ensemble_result={}


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    if config["kfold"]:
        kfold_main(config)
    else:
        main(config)
