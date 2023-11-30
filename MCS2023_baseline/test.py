import argparse
import os
import os.path as osp
import random
import sys
import yaml

import torch
import numpy as np
import torch.nn as nn
import utils

from tqdm import tqdm

from data_utils import get_dataloader,  dataset, augmentations
from models import models
from train import  validation
from utils import convert_dict_to_tuple

def main(args: argparse.Namespace) -> None:

    with open(args.cfg) as f:
            data = yaml.safe_load(f)

    config = convert_dict_to_tuple(data)

    print("Preparing test reader...")
    test_dataset = dataset.Product10KDatasetGroup(
        root="/home/paperspace/visual-product-recognition-2023-starter-kit/Product10K/test_origin/", annotation_file="/home/paperspace/visual-product-recognition-2023-starter-kit/Product10K/validation.csv",
        transforms=augmentations.get_val_aug(config), is_val=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
        drop_last=False,
        pin_memory=True
    )
    print("Done.")

    net, is_autoencoder = models.load_model(config)
    checkpoint_path = './experiments/resnet50_2/model_0011.pth'
    # print (checkpoint)
    # net.load_state_dict(torch.load(checkpoint_path, map_location="cuda")) 
    ckpt = torch.load(checkpoint_path, map_location='cpu') 
    net.load_state_dict(ckpt['state_dict'])
    criterion , _, _ = utils.get_training_parameters(config, net)
    acc = validation(net, False, test_loader, criterion, config, 0)

    
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='Path to config file.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
