import numpy as np

import yaml

import numpy as np
import torch
import torchvision.models as models

from collections import OrderedDict

from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from tqdm import tqdm

import sys
sys.path.append('./MCS2023_baseline/')
from autoencoder import ConvAutoencoder, TransferConvAutoencoder

from data_utils import get_dataloader
from data_utils import dataset
from data_utils.augmentations import get_val_aug
from data_utils.dataset import SubmissionDataset, TestDataset
from utils import convert_dict_to_tuple
from train import train, validation
from utils import AverageMeter
from torch import nn

from models.net import *

import sys
class MCS_Ranker:
    def __init__(self, dataset_path, gallery_csv_path, queries_csv_path):
        """
        Initialize your model here
        Inputs:
            dataset_path
            gallery_csv_path
            queries_csv_path
        """
        # Try not to change
        self.dataset_path = dataset_path
        self.gallery_csv_path = gallery_csv_path
        self.queries_csv_path = queries_csv_path
        self.max_predictions = 1000

        # Add your code below

        print (dataset_path)
        print (gallery_csv_path)
        print (queries_csv_path)
        # checkpoint_path = 'experiments/Contrastive_bk/net_0.pth'
        # print (checkpoint_path)
        self.batch_size = 100

        self.exp_cfg = './MCS2023_baseline/config/ranker_cfg.yml'
        self.inference_cfg = './MCS2023_baseline/config/inference_config.yml'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with open(self.exp_cfg) as f:
            data = yaml.safe_load(f)
        self.exp_cfg = convert_dict_to_tuple(data)

        with open(self.inference_cfg) as f:
            data = yaml.safe_load(f)
        self.inference_cfg = convert_dict_to_tuple(data)
        
        print('Creating model and loading checkpoint')
        # self.model = models.__dict__["efficientnet_v2_s"](
        #     num_classes=9691
        # )
        # # # self.model = TransferConvAutoencoder()
        # # # self.embedding_shape = 1280
        # checkpoint = torch.load(checkpoint_path, map_location='cuda')['state_dict']
        # new_state_dict = OrderedDict()
        # for k, v in checkpoint.items():
        #     name = k.replace("module.", "")
        #     new_state_dict[name] = v
        # # num_classes = 361
        # # self.model.load_state_dict(new_state_dict)
        # # self.model = TripletNet(resnet101())
        # # checkpoint = torch.load(checkpoint_path, map_location='cuda')['state_dict']
        # # new_state_dict = OrderedDict()
        # # for k, v in checkpoint.items():
        # #     name = k.replace("module.", "")
        # #     new_state_dict[name] = v
        # self.model.load_state_dict(new_state_dict)
        # self.embedding_shape = 1280
        # self.model.classifier[1] = torch.nn.Identity()
        self.embedding_shape = 1280 #efficinetnet
        # # self.embedding_shape = 1024 #densenet
        # # self.model.classifier = torch.nn.Identity()
        
        # self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        self.model = ImageEmbedding()
        checkpoint_path = "submission_model/loss/model_4.pth"
        checkpoint = torch.load(checkpoint_path, map_location='cuda')['state_dict']
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v
        # num_classes = 361
        self.model.load_state_dict(new_state_dict)
        # print (self.model)
        self.model.eval()
        self.model.to(self.device)
        print('Weights are loaded, fc layer is deleted!')


    def raise_aicrowd_error(self, msg):
        """ Will be used by the evaluator to provide logs, DO NOT CHANGE """
        raise NameError(msg)
    
    def predict_product_ranks(self):
        """
        This function should return a numpy array of shape `(num_queries, 1000)`. 
        For ach query image your model will need to predict 
        a set of 1000 unique gallery indexes, in order of best match first.

        Outputs:
            class_ranks - A 2D numpy array where the axes correspond to:
                          axis 0 - Batch size
                          axis 1 - An ordered rank list of matched image indexes, most confident prediction first
                            - maximum length of this should be 1000
                            - predictions above this limit will be dropped
                            - duplicates will be dropped such that the lowest index entry is preserved
        """

        gallery_dataset = SubmissionDataset(
            root=self.dataset_path, annotation_file=self.gallery_csv_path,
            transforms=get_val_aug(self.exp_cfg)
        )

        gallery_loader = torch.utils.data.DataLoader(
            gallery_dataset, batch_size=self.batch_size,
            shuffle=False, pin_memory=True, num_workers=self.batch_size
        )

        query_dataset = SubmissionDataset(
            root=self.dataset_path, annotation_file=self.queries_csv_path,
            transforms=get_val_aug(self.exp_cfg), with_bbox=True
        )

        query_loader = torch.utils.data.DataLoader(
            query_dataset, batch_size=self.batch_size,
            shuffle=False, pin_memory=True, num_workers=self.batch_size
        )

        # print (self.dataset_path)
        # print (self.gallery_csv_path)
        # test_dataset = TestDataset(
        #     root=self.dataset_path, annotation_file=self.queries_csv_path,
        #     transforms=get_val_aug(self.exp_cfg)
        # )
        
        # test_loader = torch.utils.data.DataLoader(
        #     test_dataset,
        #     batch_size=256,
        #     shuffle=False,
        #     num_workers=8,
        #     drop_last=False,
        #     pin_memory=True
        # )
        # BUONO
        # test_dataset = dataset.Product10KDataset(
        #     root="/home/paperspace/visual-product-recognition-2023-starter-kit/Product10K/test/", annotation_file="/home/paperspace/visual-product-recognition-2023-starter-kit/Product10K/test_kaggletest.csv",
        #     transforms=get_val_aug(self.exp_cfg)
        #     )
        # test_loader = torch.utils.data.DataLoader(
        #     test_dataset,
        #     batch_size=128,
        #     shuffle=False,
        #     num_workers=8,
        #     drop_last=False,
        #     pin_memory=True
        # )
        # print('Calculating accuracy')

        # loss_stat = AverageMeter('Loss')
        # acc_stat = AverageMeter('Acc.')
        # criterion = torch.nn.CrossEntropyLoss().to('cuda')
        # with torch.no_grad():
        #     test_iter = tqdm(test_loader, desc='Test', dynamic_ncols=True, position=2)

        #     for step, (x, y) in enumerate(test_iter):
        #         out = self.model(x.cuda().to(memory_format=torch.contiguous_format))
        #         loss = criterion(out, y.cuda())
        #         num_of_samples = x.shape[0]
        #         # print (num_of_samples)
               
        #         loss_stat.update(loss.detach().cpu().item(), num_of_samples)

        #         scores = torch.softmax(out, dim=1).detach().cpu().numpy()
        #         predict = np.argmax(scores, axis=1)
        #         gt = y.detach().cpu().numpy()

        #         acc = np.mean(gt == predict)
        #         acc_stat.update(acc, num_of_samples)

        #         acc_val, acc_avg = acc_stat()
        #         loss_val, loss_avg = loss_stat()
        #     print('Validation of test is done; \n loss: {:.4f}; acc: {:.2f}'.format(loss_avg, acc_avg))
        # FINE
        # with torch.no_grad():
        #     for i, res in tqdm(enumerate(test_loader),
        #                         total=len(test_loader)):
        #         images = res[0]
        #         gt = res[1]
        #         gt = gt.detach().cpu().numpy()
        #         # print ("gt: ", gt)
        #         # print ("shape: ", images.shape)
        #         tensor = images.view(1, 3, 224, 224)
        #         out = self.model(tensor.cuda().to(memory_format=torch.contiguous_format))
        #         scores = torch.softmax(out, dim=1).detach().cpu().numpy()
        #         predict = np.argmax(scores, axis=1)
        #         if gt[0] == predict[0]:
        #             print ("classe: ", predict)
        # print ("acc: ", correct/len(test_loader) )
        # sys.exit()
        print('Calculating embeddings')
        print ("Len dataloader gallery: ", len(gallery_loader))
        print ("Len dataloader query: ", len(query_loader))
        # print (self.device)
        gallery_embeddings = np.zeros((len(gallery_dataset), self.embedding_shape))
        query_embeddings = np.zeros((len(query_dataset), self.embedding_shape))

        print (self.model)
        with torch.no_grad():
            for i, images in tqdm(enumerate(gallery_loader),
                                total=len(gallery_loader)):
                # print (images.shape)
                images = images.to(self.device)
                outputs, _= self.model(images)
                # print (imgs.shape)
                # print (embs.shape)
                # sys.exit()
                outputs = outputs.data.cpu().numpy()
                # print (outputs[0])
                gallery_embeddings[
                    i*self.batch_size:(i*self.batch_size + self.batch_size), :
                ] = outputs
            
            for i, images in tqdm(enumerate(query_loader),
                                total=len(query_loader)):
                # print (images.shape)
                images = images.to(self.device)
                outputs, _ = self.model(images)


                outputs = outputs.data.cpu().numpy()
                query_embeddings[
                    i*self.batch_size:(i*self.batch_size + self.batch_size), :
                ] = outputs
        
        print('Normalizing and calculating distances')
        gallery_embeddings = normalize(gallery_embeddings)
        query_embeddings = normalize(query_embeddings)
        distances = pairwise_distances(query_embeddings, gallery_embeddings, "cityblock")
        sorted_distances = np.argsort(distances, axis=1)[:, :1000]

        class_ranks = sorted_distances
        return class_ranks