
import torch
import numpy as np
import torch.nn as nn
import math 
from torch import nn
import torch.nn.functional as F
from utils import LabelSmoothingCrossEntropy
from ssim import *
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu((0.8)*distance_positive - (0.2)*distance_negative + self.margin)
        return losses.mean() 

class CustomLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(CustomLoss, self).__init__()
        self.margin = 1.0
    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        losses = torch.relu(distance_positive + self.margin)
        return losses.mean() 


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())
            
    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask.cuda() * torch.exp(similarity_matrix / self.temperature)
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss

class CombinedLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(CombinedLoss, self).__init__()
        self.label_loss = LabelSmoothingCrossEntropy()
    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)
    def forward(self, x, target, anchor, loss, y_true):
        margin = 1.0
        y_pred = self.calc_euclidean(anchor, loss)
        loss_contrastive = torch.mean((y_true * torch.pow(y_pred, 2)) + ((1 - y_true) * torch.pow(torch.clamp(margin - y_pred, min=0.0), 2)))
        label_loss = self.label_loss(x, target)
        losses = torch.relu(loss_contrastive)
        return losses.mean() 

class AutoencoderLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(AutoencoderLoss, self).__init__()
        # self.label_loss = LabelSmoothingCrossEntropy()
        self.ssim_loss = SSIM()
    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)
    def forward(self, img_train, anchor, loss, y_true, img):
        margin = 1.0
        embedding_anchor = anchor.reshape((anchor.shape[0], -1))
        embedding_loss = loss.reshape((loss.shape[0], -1))
        y_pred = self.calc_euclidean(embedding_anchor, embedding_loss)
        loss_contrastive = torch.mean((y_true * torch.pow(y_pred, 2)) + ((1 - y_true) * torch.pow(torch.clamp(margin - y_pred, min=0.0), 2)))
        # label_loss = self.label_loss(x, target)
        ssim_out = -self.ssim_loss(img_train, img)
        ssim_value = - ssim_out.data[0]
        losses = torch.relu(loss_contrastive + ssim_out)
        return losses.mean(), ssim_value