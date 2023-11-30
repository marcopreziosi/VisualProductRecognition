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

from data_utils import get_dataloader
from models import models
from train import train, validation, calculate_map_epoch
from utils import convert_dict_to_tuple
from loss import TripletLoss, ContrastiveLoss, CombinedLoss, AutoencoderLoss
from ssim import *
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim import RMSprop
from torch.optim.lr_scheduler import CosineAnnealingLR
from collections import OrderedDict
# sys.path.append('./MCS2023_baseline/')
# MCS2023_baseline/models_transformer
# from models_transformer import VisionTransformer
# from visualization_functions import generate_sim_maps, show_cam_on_image, norm
# from dataset_norms import dataset_norms
from autoencoder import ConvEncoder, ConvDecoder
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
        net, is_autoencoder = models.load_model(config)
        checkpoint_path = './experiments/resnet50_2/model_0021.pth'
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
        
        
        net, is_autoencoder = models.load_model(config)
        # checkpoint_path = "/home/paperspace/visual-product-recognition-2023-starter-kit/experiments/CombinedLoss_allTrain_addcrop/model_0.pth"
        # checkpoint = torch.load(checkpoint_path, map_location='cuda')['state_dict']
        # new_state_dict = OrderedDict()
        # for k, v in checkpoint.items():
        #     name = k.replace("module.", "")
        #     new_state_dict[name] = v
        # net.load_state_dict(new_state_dict)
        print (net)
        
        # for i, param in enumerate(net.parameters()):
        #     if i<150:
        #         param.requires_grad = False
        
        net.cuda()
        
        if config.num_gpu > 1:
            net = torch.nn.DataParallel(net)
    print("Done!")
    # optimizer = torch.optim.SGD(net.parameters(),
    #                             lr=config.train.learning_rate,
    #                             momentum=config.train.momentum,
    #                             weight_decay=config.train.weight_decay)
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            step_size=config.train.lr_schedule.step_size,
                                            gamma=config.train.lr_schedule.gamma)
    if config.dataset.contrastive_all:
        criterion = ContrastiveLoss(config.dataset.batch_size, 1.0)
    elif config.dataset.triplete:
        # criterion = nn.TripletMarginLoss(margin=1.0, p=2) 
        criterion = TripletLoss() 
    elif config.dataset.combinedLoss:
        criterion =CombinedLoss()
    elif config.dataset.autoencoder:
        criterion = AutoencoderLoss()
        
    else:
        criterion = None
    

    # if not is_autoencoder:
    #     if config.dataset.triplete: 
    #         print ("---LossTriplete---")

    #         _ , optimizer, scheduler = utils.get_training_parameters(config, net)
            
    #         # criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    #         criterion = nn.MSELoss().to('cuda')
    #         # criterion = nn.HingeEmbeddingLoss()
    #     else:
    #         print ("---LossGroup---")
    #         criterion , optimizer, scheduler = utils.get_training_parameters(config, net)
    # else:
    #     _, optimizer, scheduler = utils.get_training_parameters(config, net)
    #     criterion = torch.nn.MSELoss().to('cuda')
    
    if config.train.resume:
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
    
    train_epoch = tqdm(range(start_epoch, end_epoch), dynamic_ncols=True, 
                       desc='Epochs', position=0)

    print (f"learning: {config.train.learning_rate}")
    best_loss = np.inf
    for epoch in train_epoch:
        print (epoch)
        train(net, train_loader, criterion, optimizer, config, epoch)
        # epoch_avg_loss = validation(net, val_loader, criterion, epoch)
        map_result = calculate_map_epoch(net,epoch, config)
        if best_acc <= map_result:
            utils.save_checkpoint(net, optimizer, scheduler, epoch, outdir)
            best_acc = map_result
        scheduler.step()



def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='Path to config file.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
