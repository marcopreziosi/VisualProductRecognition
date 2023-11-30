import argparse
import os
import os.path as osp
import random
import sys
import yaml

import torch
import numpy as np

import utils

from tqdm import tqdm

from data_utils import get_dataloader
from models import models
from train import train, validation
from utils import convert_dict_to_tuple

def main(args: argparse.Namespace) -> None:
    """
    Run train process of classification model
    :param args: all parameters necessary for launch
    :return: None
    """
    with open(args.cfg) as f:
        data = yaml.safe_load(f)

    config = convert_dict_to_tuple(data)

    seed = config.dataset.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    outdir = osp.join(config.outdir, config.exp_name)
    print("Savedir: {}".format(outdir))
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    train_loader, val_loader = get_dataloader.get_dataloaders(config)
    n_epoch = 0
    print("Loading model...")
    if config.train.resume:
        net = models.load_model(config)
        checkpoint_path = '/home/paperspace/visual-product-recognition-2023-starter-kit/experiments/densenet121_crop/model_0028.pth'
        # print (checkpoint)
        # net.load_state_dict(torch.load(checkpoint_path, map_location="cuda")) 
        ckpt = torch.load(checkpoint_path, map_location='cpu') 
        net.load_state_dict(ckpt['state_dict'])
        start_epoch = ckpt['epoch']
       
        n_epoch = config.train.n_epoch - start_epoch
        end_epoch = config.train.n_epoch

    else: 
        best_acc = 0
        start_epoch = 0
        n_epoch = config.train.n_epoch
        end_epoch = n_epoch
        net = models.load_model(config)
        if config.num_gpu > 1:
            net = torch.nn.DataParallel(net)
    print("Done!")

    criterion, optimizer, scheduler = utils.get_training_parameters(config, net)
    
    if config.train.resume:
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])

    
    train_epoch = tqdm(range(start_epoch, end_epoch), dynamic_ncols=True, 
                       desc='Epochs', position=0)

    

    # main process
    best_acc = 0
    for epoch in train_epoch:
        print (epoch)
        # sys.exit()
        train(net, train_loader, criterion, optimizer, config, epoch)
        epoch_avg_acc = validation(net, val_loader, criterion, epoch)
        if epoch_avg_acc >= best_acc:
            utils.save_checkpoint(net, optimizer, scheduler, epoch, outdir)
            best_acc = epoch_avg_acc
        scheduler.step()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='Path to config file.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
