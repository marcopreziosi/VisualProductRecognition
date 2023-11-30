import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from utils import AverageMeter
from utils import convert_dict_to_tuple


from data_utils import get_dataloader
from data_utils import dataset
from data_utils.augmentations import get_val_aug
from data_utils.dataset import SubmissionDataset, TestDataset
import yaml
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
import pandas as pd
import os
from mean_average_precision import calculate_map
import cv2
# from visualization_functions import generate_sim_maps
# import statistics

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()

def train(model: torch.nn.Module,
          train_loader: torch.utils.data.DataLoader,
          criterion: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          config, epoch) -> None:
    """
    Model training function for one epoch
    :param model: model architecture
    :param train_loader: dataloader for batch generation
    :param criterion: selected criterion for calculating the loss function
    :param optimizer: selected optimizer for updating weights
    :param config: train process configuration
    :param epoch (int): epoch number
    :return: None
    """

    if config.dataset.contrastive_all:
        print ("---TrainContrastive---")
        model.train()

        loss_stat = AverageMeter('Loss')

        train_iter = tqdm(train_loader, desc='Train', dynamic_ncols=True, position=1)

        for step, (img_train, img_positive, class_id) in enumerate(train_iter):
            
            optimizer.zero_grad()

            embX, projectionX = model(img_train.cuda())
            embY, projectionY = model(img_positive.cuda())

            loss = criterion(projectionX, projectionY)
            loss.backward()
            num_of_samples = img_train.shape[0]
            loss_stat.update(loss.detach().cpu().item(), num_of_samples)
            # print (loss)
            optimizer.step()

            if step % config.train.freq_vis == 0 and not step == 0:
                loss_val, loss_avg = loss_stat()
                print('Epoch: {}; step: {}; loss: {:.4f}; '.format(epoch, step, loss_avg))
        
        loss_val, loss_avg = loss_stat()
        print('Train process of epoch: {} is done; \n loss: {:.4f};'.format(epoch, loss_avg))
   
    elif config.dataset.triplete:
        print ("---Triplete---")
        model.train()

        loss_stat = AverageMeter('Loss')

        train_iter = tqdm(train_loader, desc='Train', dynamic_ncols=True, position=1)

        for step, (img, target, group, img_negative, img_positive) in enumerate(train_iter):
            
            optimizer.zero_grad()

            embedded_a, embedded_p, embedded_n = model(img.cuda(), img_positive.cuda(), img_negative.cuda())
            

            loss = criterion(embedded_a, embedded_p, embedded_n)
            loss.backward()
            num_of_samples = img.shape[0]
            loss_stat.update(loss.detach().cpu().item(), num_of_samples)
            # print (loss)
            optimizer.step()

            if step % config.train.freq_vis == 0 and not step == 0:
                loss_val, loss_avg = loss_stat()
                print('Epoch: {}; step: {}; loss: {:.4f}; '.format(epoch, step, loss_avg))
        
        loss_val, loss_avg = loss_stat()
        print('Train process of epoch: {} is done; \n loss: {:.4f};'.format(epoch, loss_avg)) 
    
    elif config.dataset.vit:
        print ("---TrainVit---")
        model.train()

        loss_stat = AverageMeter('Loss')

        train_iter = tqdm(train_loader, desc='Train', dynamic_ncols=True, position=1)

        for step, (img_train, img_positive, class_id) in enumerate(train_iter):
            # print (img_train.shape)
            # loss = model(img_train.cuda())
            im = torch.FloatTensor(20, 3, 256, 256).uniform_(0., 1.)
            loss = model(im)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num_of_samples = img_train.shape[0]
            loss_stat.update(loss.detach().cpu().item(), num_of_samples)
            if step % config.train.freq_vis == 0 and not step == 0:
                loss_val, loss_avg = loss_stat()
                print('Epoch: {}; step: {}; loss: {:.4f}; '.format(epoch, step, loss_avg))
        
        loss_val, loss_avg = loss_stat()
        print('Train process of epoch: {} is done; \n loss: {:.4f};'.format(epoch, loss_avg))           
    
    elif config.dataset.combinedLoss:
        print ("---TrainCombinedLoss---")
        model.train()

        loss_stat = AverageMeter('Loss')
        acc_stat = AverageMeter('Acc.')
        train_iter = tqdm(train_loader, desc='Train', dynamic_ncols=True, position=1)

        for step, (img_train, img_loss, y_true, target) in enumerate(train_iter):
            
            optimizer.zero_grad()

            embedding_anchor, x = model(img_train.cuda())
            embedding_loss, _ = model(img_loss.cuda())

            loss = criterion(x, target.cuda(), embedding_anchor, embedding_loss, y_true.cuda())
            loss.backward()
            num_of_samples = img_train.shape[0]
            loss_stat.update(loss.detach().cpu().item(), num_of_samples)
            # print (loss)
            optimizer.step()

            scores = torch.softmax(x, dim=1).detach().cpu().numpy()
            predict = np.argmax(scores, axis=1)
            gt = target.detach().cpu().numpy()

            acc = np.mean(gt == predict)
            acc_stat.update(acc, num_of_samples)

            if step % config.train.freq_vis == 0 and not step == 0:
                loss_val, loss_avg = loss_stat()
                acc_val, acc_avg = acc_stat()
                print('Epoch: {}; step: {}; loss: {:.4f}; acc: {:.4f}'.format(epoch, step, loss_avg, acc_avg))
        
        acc_val, acc_avg = acc_stat()
        loss_val, loss_avg = loss_stat()
        print('Train process of epoch: {} is done; \n loss: {:.4f}; acc: {:.2f}'.format(epoch, loss_avg, acc_avg))

    elif config.dataset.autoencoder:
        print ("---TrainAutoencoderLoss---")
        model.train()

        loss_stat = AverageMeter('Loss')
        acc_stat = AverageMeter('Acc.')
        train_iter = tqdm(train_loader, desc='Train', dynamic_ncols=True, position=1)

        for step, (img_train, img_loss, y_true, target) in enumerate(train_iter):
            
            optimizer.zero_grad()
            embedding_anchor, img = model(img_train.cuda())
            embedding_loss, _ = model(img_loss.cuda())
            # print (f"img: {img_train.shape}, img': {img.shape}")
            # sys.exit()
            loss, sim_value = criterion(img_train.cuda(), embedding_anchor, embedding_loss, y_true.cuda(), img)
            loss.backward()
            num_of_samples = img_train.shape[0]
            loss_stat.update(loss.detach().cpu().item(), num_of_samples)
            # print (loss)
            optimizer.step()

            # scores = torch.softmax(x, dim=1).detach().cpu().numpy()
            # predict = np.argmax(scores, axis=1)
            # gt = target.detach().cpu().numpy()

            # acc = np.mean(gt == predict)
            # acc_stat.update(acc, num_of_samples)

            # if step % config.train.freq_vis == 0 and not step == 0:
            #     loss_val, loss_avg = loss_stat()
            #     acc_val, acc_avg = acc_stat()
            print('Epoch: {}; step: {}; loss: {:.4f}; sim_value: {:.4f}'.format(epoch, step, loss_avg, sim_value))
        
        # acc_val, acc_avg = acc_stat()
        loss_val, loss_avg = loss_stat()
        print('Train process of epoch: {} is done; \n loss: {:.4f}'.format(epoch, loss_avg))
   

def validation(model: torch.nn.Module,
               val_loader: torch.utils.data.DataLoader,
               criterion: torch.nn.Module,
               epoch) -> None:
    """
    Model validation function for one epoch
    :param model: model architecture
    :param val_loader: dataloader for batch generation
    :param criterion: selected criterion for calculating the loss function
    :param epoch (int): epoch number
    :return: float: avg acc
    """
    
    loss_stat = AverageMeter('Loss')
    with torch.no_grad():
        model.eval()
        val_iter = tqdm(val_loader, desc='Val', dynamic_ncols=True, position=2)

        for step, (img_priv, img_pub, class_id) in enumerate(val_iter):
            embX, projectionX = model(img_priv.cuda())
            embY, projectionY = model(img_pub.cuda())

            loss = criterion(projectionX, projectionY)
            num_of_samples = img_priv.shape[0]
            loss_stat.update(loss.detach().cpu().item(), num_of_samples)

        loss_val, loss_avg = loss_stat()
        print('Validation of epoch: {} is done; \n loss: {:.4f}'.format(epoch, loss_avg))
        return loss_avg
    
def calculate_map_epoch(model, epoch, config):
    if config.dataset.contrastive_all:
        model.eval()
        print ("---ValContrastive---")
        datafolder = "/home/paperspace/visual-product-recognition-2023-starter-kit/development_test_data"
        gallery_csv_path = "/home/paperspace/visual-product-recognition-2023-starter-kit/development_test_data/gallery.csv"
        queries_csv_path = "/home/paperspace/visual-product-recognition-2023-starter-kit/development_test_data/queries.csv"
        exp_cfg = '/home/paperspace/visual-product-recognition-2023-starter-kit/MCS2023_baseline/config/ranker_cfg.yml'
        inference_cfg = '/home/paperspace/visual-product-recognition-2023-starter-kit/MCS2023_baseline/config/inference_config.yml'


        with open(exp_cfg) as f:
            data = yaml.safe_load(f)
        exp_cfg = convert_dict_to_tuple(data)
        gallery_dataset = SubmissionDataset(
            root=datafolder, annotation_file=gallery_csv_path,
            transforms=get_val_aug(exp_cfg)
        )

        gallery_loader = torch.utils.data.DataLoader(
            gallery_dataset, batch_size=64,
            shuffle=False, pin_memory=True, num_workers=8
        )

        query_dataset = SubmissionDataset(
            root=datafolder, annotation_file=queries_csv_path,
            transforms=get_val_aug(exp_cfg), with_bbox=True
        )

        query_loader = torch.utils.data.DataLoader(
            query_dataset, batch_size=64,
            shuffle=False, pin_memory=True, num_workers=8
        )


        print('Calculating embeddings')
        embedding_shape = 1024
        gallery_embeddings = np.zeros((len(gallery_dataset), embedding_shape))
        query_embeddings = np.zeros((len(query_dataset), embedding_shape))
        with torch.no_grad():
            for i, images in tqdm(enumerate(gallery_loader),
                                total=len(gallery_loader)):
                images = images.to("cuda")
                _, outputs = model(images)
                outputs = outputs.data.cpu().numpy()
                gallery_embeddings[
                    i*64:(i*64 + 64), :
                ] = outputs
            
            for i, images in tqdm(enumerate(query_loader),
                                total=len(query_loader)):
                images = images.to("cuda")
                _, outputs = model(images)
                outputs = outputs.data.cpu().numpy()
                query_embeddings[
                    i*64:(i*64 + 64), :
                ] = outputs

        print('Normalizing and calculating distances')
        gallery_embeddings = normalize(gallery_embeddings)
        query_embeddings = normalize(query_embeddings)

        similarities = ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan','braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'minkowski', 'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
        class_ranks = []
        maps = {}
        seller_gt = pd.read_csv(os.path.join(datafolder, 'gallery.csv'))
        gallery_labels = seller_gt['product_id'].values
        user_gt = pd.read_csv(os.path.join(datafolder, 'queries.csv'))
        query_labels = user_gt['product_id'].values

        for similarity in similarities:
            distances = pairwise_distances(query_embeddings, gallery_embeddings, "braycurtis")
            sorted_distances = np.argsort(distances, axis=1)[:, :1000]
            class_ranks.append(sorted_distances)
            maps[similarity] = calculate_map(sorted_distances, query_labels, gallery_labels)
        # max_map = calculate_map(sorted_distances, query_labels, gallery_labels)

        # # Evalaute metrics
        # print("Evaluation Results")
        max_map = 0
        for k, v in maps.items():
            if v > max_map:
                max_map = v
                similarity = k
        results = (f"mAP epoch {epoch}: {max_map}, ottenuto con la {similarity} ")
        print(results)
        return max_map
    
    elif config.dataset.triplete:
        model.eval()
        print ("---MapTriplete---")
        datafolder = "/home/paperspace/visual-product-recognition-2023-starter-kit/development_test_data"
        gallery_csv_path = "/home/paperspace/visual-product-recognition-2023-starter-kit/development_test_data/gallery.csv"
        queries_csv_path = "/home/paperspace/visual-product-recognition-2023-starter-kit/development_test_data/queries.csv"
        exp_cfg = '/home/paperspace/visual-product-recognition-2023-starter-kit/MCS2023_baseline/config/ranker_cfg.yml'
        inference_cfg = '/home/paperspace/visual-product-recognition-2023-starter-kit/MCS2023_baseline/config/inference_config.yml'


        with open(exp_cfg) as f:
            data = yaml.safe_load(f)
        exp_cfg = convert_dict_to_tuple(data)
        gallery_dataset = SubmissionDataset(
            root=datafolder, annotation_file=gallery_csv_path,
            transforms=get_val_aug(exp_cfg)
        )

        gallery_loader = torch.utils.data.DataLoader(
            gallery_dataset, batch_size=64,
            shuffle=False, pin_memory=True, num_workers=8
        )

        query_dataset = SubmissionDataset(
            root=datafolder, annotation_file=queries_csv_path,
            transforms=get_val_aug(exp_cfg), with_bbox=True
        )

        query_loader = torch.utils.data.DataLoader(
            query_dataset, batch_size=64,
            shuffle=False, pin_memory=True, num_workers=8
        )


        print('Calculating embeddings')
        embedding_shape = 1280
        gallery_embeddings = np.zeros((len(gallery_dataset), embedding_shape))
        query_embeddings = np.zeros((len(query_dataset), embedding_shape))
        with torch.no_grad():
            for i, images in tqdm(enumerate(gallery_loader),
                                total=len(gallery_loader)):
                images = images.to("cuda")
                outputs, _, _ = model(images, images, images)
                outputs = outputs.data.cpu().numpy()
                gallery_embeddings[
                    i*64:(i*64 + 64), :
                ] = outputs
            
            for i, images in tqdm(enumerate(query_loader),
                                total=len(query_loader)):
                images = images.to("cuda")
                _, outputs, _ = model(images, images, images)
                outputs = outputs.data.cpu().numpy()
                query_embeddings[
                    i*64:(i*64 + 64), :
                ] = outputs

        print('Normalizing and calculating distances')
        gallery_embeddings = normalize(gallery_embeddings)
        query_embeddings = normalize(query_embeddings)

        similarities = ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan','braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'minkowski', 'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
        class_ranks = []
        maps = {}
        seller_gt = pd.read_csv(os.path.join(datafolder, 'gallery.csv'))
        gallery_labels = seller_gt['product_id'].values
        user_gt = pd.read_csv(os.path.join(datafolder, 'queries.csv'))
        query_labels = user_gt['product_id'].values

        for similarity in similarities:
            distances = pairwise_distances(query_embeddings, gallery_embeddings, "braycurtis")
            sorted_distances = np.argsort(distances, axis=1)[:, :1000]
            class_ranks.append(sorted_distances)
            maps[similarity] = calculate_map(sorted_distances, query_labels, gallery_labels)
        # max_map = calculate_map(sorted_distances, query_labels, gallery_labels)

        # # Evalaute metrics
        # print("Evaluation Results")
        max_map = 0
        for k, v in maps.items():
            if v > max_map:
                max_map = v
                similarity = k
        results = (f"mAP epoch {epoch}: {max_map}, ottenuto con la {similarity} ")
        print(results)
        return max_map
    
    elif config.dataset.combinedLoss:
        model.eval()
        print ("---MapCombinedLoss---")
        datafolder = "/home/paperspace/visual-product-recognition-2023-starter-kit/development_test_data"
        gallery_csv_path = "/home/paperspace/visual-product-recognition-2023-starter-kit/development_test_data/gallery.csv"
        queries_csv_path = "/home/paperspace/visual-product-recognition-2023-starter-kit/development_test_data/queries.csv"
        exp_cfg = '/home/paperspace/visual-product-recognition-2023-starter-kit/MCS2023_baseline/config/ranker_cfg.yml'
        inference_cfg = '/home/paperspace/visual-product-recognition-2023-starter-kit/MCS2023_baseline/config/inference_config.yml'


        with open(exp_cfg) as f:
            data = yaml.safe_load(f)
        exp_cfg = convert_dict_to_tuple(data)
        gallery_dataset = SubmissionDataset(
            root=datafolder, annotation_file=gallery_csv_path,
            transforms=get_val_aug(exp_cfg)
        )

        gallery_loader = torch.utils.data.DataLoader(
            gallery_dataset, batch_size=64,
            shuffle=False, pin_memory=True, num_workers=8
        )

        query_dataset = SubmissionDataset(
            root=datafolder, annotation_file=queries_csv_path,
            transforms=get_val_aug(exp_cfg), with_bbox=True
        )

        query_loader = torch.utils.data.DataLoader(
            query_dataset, batch_size=64,
            shuffle=False, pin_memory=True, num_workers=8
        )


        print('Calculating embeddings')
        embedding_shape = 1280
        gallery_embeddings = np.zeros((len(gallery_dataset), embedding_shape))
        query_embeddings = np.zeros((len(query_dataset), embedding_shape))
        with torch.no_grad():
            for i, images in tqdm(enumerate(gallery_loader),
                                total=len(gallery_loader)):
                images = images.to("cuda")
                outputs, _,= model(images)
                outputs = outputs.data.cpu().numpy()
                gallery_embeddings[
                    i*64:(i*64 + 64), :
                ] = outputs
            
            for i, images in tqdm(enumerate(query_loader),
                                total=len(query_loader)):
                images = images.to("cuda")
                outputs, _ = model(images)
                outputs = outputs.data.cpu().numpy()
                query_embeddings[
                    i*64:(i*64 + 64), :
                ] = outputs

        print('Normalizing and calculating distances')
        gallery_embeddings = normalize(gallery_embeddings)
        query_embeddings = normalize(query_embeddings)

        similarities = ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan','braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'minkowski', 'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
        class_ranks = []
        maps = {}
        seller_gt = pd.read_csv(os.path.join(datafolder, 'gallery.csv'))
        gallery_labels = seller_gt['product_id'].values
        user_gt = pd.read_csv(os.path.join(datafolder, 'queries.csv'))
        query_labels = user_gt['product_id'].values

        for similarity in similarities:
            distances = pairwise_distances(query_embeddings, gallery_embeddings, "braycurtis")
            sorted_distances = np.argsort(distances, axis=1)[:, :1000]
            class_ranks.append(sorted_distances)
            maps[similarity] = calculate_map(sorted_distances, query_labels, gallery_labels)
        # max_map = calculate_map(sorted_distances, query_labels, gallery_labels)

        # # Evalaute metrics
        # print("Evaluation Results")
        max_map = 0
        for k, v in maps.items():
            if v > max_map:
                max_map = v
                similarity = k
        results = (f"mAP epoch {epoch}: {max_map}, ottenuto con la {similarity} ")
        print(results)
        return max_map

    elif config.dataset.autoencoder:
        model.eval()
        print ("---MapAutoEncoderLoss---")
        datafolder = "/home/paperspace/visual-product-recognition-2023-starter-kit/development_test_data"
        gallery_csv_path = "/home/paperspace/visual-product-recognition-2023-starter-kit/development_test_data/gallery.csv"
        queries_csv_path = "/home/paperspace/visual-product-recognition-2023-starter-kit/development_test_data/queries.csv"
        exp_cfg = '/home/paperspace/visual-product-recognition-2023-starter-kit/MCS2023_baseline/config/ranker_cfg.yml'
        inference_cfg = '/home/paperspace/visual-product-recognition-2023-starter-kit/MCS2023_baseline/config/inference_config.yml'


        with open(exp_cfg) as f:
            data = yaml.safe_load(f)
        exp_cfg = convert_dict_to_tuple(data)
        gallery_dataset = SubmissionDataset(
            root=datafolder, annotation_file=gallery_csv_path,
            transforms=get_val_aug(exp_cfg)
        )

        gallery_loader = torch.utils.data.DataLoader(
            gallery_dataset, batch_size=64,
            shuffle=False, pin_memory=True, num_workers=8
        )

        query_dataset = SubmissionDataset(
            root=datafolder, annotation_file=queries_csv_path,
            transforms=get_val_aug(exp_cfg), with_bbox=True
        )

        query_loader = torch.utils.data.DataLoader(
            query_dataset, batch_size=64,
            shuffle=False, pin_memory=True, num_workers=8
        )


        print('Calculating embeddings')
        embedding_dim = (1, 256, 14, 14)
        embedding_gallery = torch.randn(embedding_dim)
        
        print('Calculating embeddings')
        # embedding_shape = 256,14,14
        # gallery_embeddings = np.zeros((len(gallery_dataset), 256,14,14))
        # query_embeddings = np.zeros((len(query_dataset), 256,14,14))

        with torch.no_grad():
            for i, images in tqdm(enumerate(gallery_loader),
                                total=len(gallery_loader)):
                images = images.to("cuda")
                outputs, _ = model(images).cpu()
                embedding_gallery = torch.cat((embedding_gallery, outputs), 0)

                # outputs = outputs.data.cpu().numpy()
                # gallery_embeddings[
                #     i*64:(i*64 + 64), :
                # ] = outputs
            embedding_gallery = embedding_gallery[1:]
            print (embedding_gallery.shape)
            
            numpy_embedding = embedding_gallery.cpu().detach().numpy()
            num_images = numpy_embedding.shape[0]

            # Dump the embeddings for complete dataset, not just train
            flattened_embedding_gallery = numpy_embedding.reshape((num_images, -1))
            print (flattened_embedding_gallery.shape)

            embedding_query = torch.randn(embedding_dim)
            for i, images in tqdm(enumerate(query_loader),
                                total=len(query_loader)):
                images = images.to("cuda")
                outputs, _= model(images).cpu()
                embedding_query = torch.cat((embedding_query, outputs), 0)
                # query_embeddings[
                #     i*64:(i*64 + 64), :
                # ] = outputs
            embedding_query = embedding_query[1:]
        numpy_embedding = embedding_query.cpu().detach().numpy()
        num_images = numpy_embedding.shape[0]

        # Dump the embeddings for complete dataset, not just train
        flattened_embedding_query = numpy_embedding.reshape((num_images, -1))
        print (flattened_embedding_query.shape)
        print('Normalizing and calculating distances')
        gallery_embeddings = normalize(flattened_embedding_gallery)
        query_embeddings = normalize(flattened_embedding_query)

        similarities = ['cosine', 'euclidean', 'correlation']
        class_ranks = []
        maps = {}
        seller_gt = pd.read_csv(os.path.join(datafolder, 'gallery.csv'))
        gallery_labels = seller_gt['product_id'].values
        user_gt = pd.read_csv(os.path.join(datafolder, 'queries.csv'))
        query_labels = user_gt['product_id'].values

        for similarity in similarities:
            distances = pairwise_distances(query_embeddings, gallery_embeddings, similarity)
            sorted_distances = np.argsort(distances, axis=1)[:, :1000]
            # class_ranks.append(sorted_distances)
            maps[similarity] = calculate_map(sorted_distances, query_labels, gallery_labels)

        # Evalaute metrics
        print("Evaluation Results")
        max_map = 0
        for k, v in maps.items():
            if v > max_map:
                max_map = v
                similarity = k
        results = (f"mAP epoch {epoch}: {max_map}, ottenuto con la {similarity} ")
        print(results)
        
        return max_map