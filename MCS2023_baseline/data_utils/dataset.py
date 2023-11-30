import os

import cv2
import pandas as pd
import torch.utils.data as data
import numpy as np
from PIL import Image


def read_image(image_file):
    img = cv2.imread(
        image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
    )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img is None:
        raise ValueError('Failed to read {}'.format(image_file))
    return img


class Product10KDataset(data.Dataset):
    def __init__(self, triplete, root, annotation_file, transforms, root_val, annotation_file_val, is_inference=False,
                 with_bbox=False, transforms_positive=None):
        self.root = root
        self.root_val = root_val
        self.imlist = pd.read_csv(annotation_file)
        self.imlist_val = pd.read_csv(annotation_file_val)

        self.transforms = transforms
        self.transforms_positive = transforms_positive
        self.is_inference = is_inference
        self.with_bbox = with_bbox
        self.triplete = triplete
        if self.triplete:
            self.dict_val = {}
            for i in range(9691):
                self.dict_val[i] = []
            for impath, class_id, usage in self.imlist_val.iloc:
                self.dict_val[class_id].append((impath, usage))

    def __getitem__(self, index):
        cv2.setNumThreads(6)

        if self.is_inference:
            impath, _, _ = self.imlist.iloc[index]
        else:
            impath, target, group = self.imlist.iloc[index]

        full_imname = os.path.join(self.root, impath)
        img = read_image(full_imname)
        if self.triplete:
            index_negative = np.random.randint(len(self.imlist))

            while self.imlist.iloc[index_negative][2] == group or self.imlist.iloc[index_negative][0] == impath:
                index_negative = np.random.randint(len(self.imlist))
            
            full_imname_neg = os.path.join(self.root, self.imlist.iloc[index_negative][0])

            img_negative = read_image(full_imname_neg)
        

            list_class_id_val = self.dict_val[target]

            index_positive = np.random.randint(len(list_class_id_val))
            impath_pos = list_class_id_val[index_positive][0]
            full_imname_pos = os.path.join(self.root_val, impath_pos)

            img_positive = read_image(full_imname_pos)
            img_positive = Image.fromarray(img_positive)
            
            img_positive = self.transforms_positive(img_positive)
            img_negative = Image.fromarray(img_negative)
            img_negative = self.transforms(img_negative)

        if self.with_bbox:
            x, y, w, h = self.table.loc[index, 'bbox_x':'bbox_h']
            img = img[y:y+h, x:x+w, :]

        img = Image.fromarray(img)
        img = self.transforms(img)

        
        if self.is_inference:
            return img
        else:
            if self.triplete:
                return img, target, group, img_negative, img_positive
            else:
                img, target

    def __len__(self):
        return len(self.imlist)

class Product10KDatasetGroup(data.Dataset):
    def __init__(self, root, annotation_file, transforms, is_inference=False,
                 with_bbox=False, is_val=False):
        self.root = root
        self.imlist = pd.read_csv(annotation_file)
        self.transforms = transforms
        self.is_inference = is_inference
        self.with_bbox = with_bbox
        self.is_val = is_val

    def __getitem__(self, index):
        cv2.setNumThreads(6)

        if self.is_inference:
            impath, _, _ = self.imlist.iloc[index]
        else:
            if not self.is_val:
                impath, target, group = self.imlist.iloc[index]
            else:
                impath, target, group, _ = self.imlist.iloc[index]

                

        full_imname = os.path.join(self.root, impath)
        img = read_image(full_imname)

        if self.with_bbox:
            x, y, w, h = self.table.loc[index, 'bbox_x':'bbox_h']
            img = img[y:y+h, x:x+w, :]

        img = Image.fromarray(img)
        img = self.transforms(img)

        if self.is_inference:
            return img
        else:
            return img, group


    def __len__(self):
        return len(self.imlist)

class SubmissionDataset(data.Dataset):
    def __init__(self, root, annotation_file, transforms, with_bbox=False):
        self.root = root
        self.imlist = pd.read_csv(annotation_file)
        self.transforms = transforms
        self.with_bbox = with_bbox

    def __getitem__(self, index):
        cv2.setNumThreads(6)

        full_imname = os.path.join(self.root, self.imlist['img_path'][index])
        img = read_image(full_imname)

        if self.with_bbox:
            x, y, w, h = self.imlist.loc[index, 'bbox_x':'bbox_h']
            img = img[y:y+h, x:x+w, :]

        img = Image.fromarray(img)
        img = self.transforms(img)
        return img

    def __len__(self):
        return len(self.imlist)

class TestDataset(data.Dataset):
    def __init__(self, root, annotation_file, transforms, with_bbox=False):
        self.root = root
        self.imlist = pd.read_csv(annotation_file)
        self.transforms = transforms
        self.with_bbox = with_bbox

    def __getitem__(self, index):
        cv2.setNumThreads(6)

        full_imname = os.path.join(self.root, self.imlist['img_path'][index])
        img = read_image(full_imname)
        class_name = self.imlist.loc[index, 'product_id']

        if self.with_bbox:
            x, y, w, h = self.imlist.loc[index, 'bbox_x':'bbox_h']
            img = img[y:y+h, x:x+w, :]

        img = Image.fromarray(img)
        img = self.transforms(img)
        return img, class_name

    def __len__(self):
        return len(self.imlist)

class Product10KDatasetOrigin(data.Dataset):
    def __init__(self, root, annotation_file, transforms, is_inference=False,
                 with_bbox=False):
        self.root = root
        self.imlist = pd.read_csv(annotation_file)
        self.transforms = transforms
        self.is_inference = is_inference
        self.with_bbox = with_bbox

    def __getitem__(self, index):
        cv2.setNumThreads(6)

        if self.is_inference:
            impath, _, _ = self.imlist.iloc[index]
        else:
            impath, target, _ = self.imlist.iloc[index]

        full_imname = os.path.join(self.root, impath)
        img = read_image(full_imname)

        if self.with_bbox:
            x, y, w, h = self.table.loc[index, 'bbox_x':'bbox_h']
            img = img[y:y+h, x:x+w, :]

        img = Image.fromarray(img)
        img = self.transforms(img)



        if self.is_inference:
            return img
        else:
            return img, target

    def __len__(self):
        return len(self.imlist)

class Product10KDatasetLoss(data.Dataset):
    def __init__(self, root, annotation_file, root_val, annotation_file_val, transforms, is_inference=False,
                 with_bbox=False, transforms_positive=None):
        self.root = root
        self.imlist = pd.read_csv(annotation_file)
        self.transforms = transforms
        self.is_inference = is_inference
        self.with_bbox = with_bbox
        self.transforms_positive = transforms_positive
        self.root_val = root_val
        self.imlist_val = pd.read_csv(annotation_file_val)
        self.dict_val = {}
        for i in range(9691):
            self.dict_val[i] = []
        for impath, class_id, usage in self.imlist_val.iloc:
            self.dict_val[class_id].append((impath, usage, group))


    def __getitem__(self, index):
        cv2.setNumThreads(6)

        if self.is_inference:
            impath, _, _ = self.imlist.iloc[index]
        else:
            impath, target, _ = self.imlist.iloc[index]

        full_imname = os.path.join(self.root, impath)
        img = read_image(full_imname)

        if self.with_bbox:
            x, y, w, h = self.table.loc[index, 'bbox_x':'bbox_h']
            img = img[y:y+h, x:x+w, :]

        img = Image.fromarray(img)
        query = self.transforms(img)
        target_img = self.transforms_positive(img)

        # prendo un'immagine gallary positiva (stesso target)
        list_class_id_val = self.dict_val[target]
        index_positive = np.random.randint(len(list_class_id_val))
        while list_class_id_val[index_positive][1] != "Public":
            index_positive = np.random.randint(len(list_class_id_val))
        impath_pos = list_class_id_val[index_positive][0]
        full_imname_pos = os.path.join(self.root_val, impath_pos)
        img_positive = read_image(full_imname_pos)
        img_positive = Image.fromarray(img_positive)
        img_gallery = self.transforms_positive(img_positive)

        # prendo un'immagine gallery negativa (gruppo diverso)
        index_negative = np.random.randint(len(self.imlist))
        while self.imlist.iloc[index_negative][2] == group or self.imlist.iloc[index_negative][0] == impath:
            index_negative = np.random.randint(len(self.imlist))

        if self.is_inference:
            return img
        else:
            return query, target_img, img_gallery, target

    def __len__(self):
        return len(self.imlist)

class Product10KDatasetContrastive(data.Dataset):
    def __init__(self, root, annotation_file, transforms, is_inference=False,
                 with_bbox=False, transforms_positive=None):
        self.root = root
        self.imlist = pd.read_csv(annotation_file)
        self.transforms = transforms
        self.is_inference = is_inference
        self.with_bbox = with_bbox
        self.transforms_positive = transforms_positive
    def __getitem__(self, index):
        cv2.setNumThreads(6)

        if self.is_inference:
            impath, _, _ = self.imlist.iloc[index]
        else:
            path_priv,class_id,path_pub,duplicato = self.imlist.iloc[index]

        full_imname = os.path.join(self.root, path_priv)
        img_priv = read_image(full_imname)

        img_priv = Image.fromarray(img_priv)
        img_priv = self.transforms_positive(img_priv)

        full_imname = os.path.join(self.root, path_pub)
        img_pub = read_image(full_imname)

        img_pub = Image.fromarray(img_pub)
        img_pub = self.transforms_positive(img_pub)
        
        return img_priv, img_pub, class_id

    def __len__(self):
        return len(self.imlist)

class Product10KDatasetContrastiveAll(data.Dataset):
    def __init__(self, root, annotation_file, transforms, is_inference=False,
                 with_bbox=False, transforms_positive=None):
        self.root = root
        self.imlist = pd.read_csv(annotation_file)
        self.transforms = transforms
        self.is_inference = is_inference
        self.with_bbox = with_bbox
        self.transforms_positive = transforms_positive
        self.dict_train = {}
        for i in range(9691):
            self.dict_train[i] = []
        for impath, class_id, group in self.imlist.iloc:
            self.dict_train[class_id].append((impath, group))

    def __getitem__(self, index):
        cv2.setNumThreads(6)

        if self.is_inference:
            impath, _, _ = self.imlist.iloc[index]
        else:
            img_train, class_id, _ = self.imlist.iloc[index]
            

        full_imname = os.path.join(self.root, img_train)
        img_train = read_image(full_imname)

        img_train = Image.fromarray(img_train)
        img_train = self.transforms_positive(img_train)

        # prendo un'immagine positiva (stesso target)
        list_class_id_train = self.dict_train[class_id]
        index_positive = np.random.randint(len(list_class_id_train))
       
        impath_pos = list_class_id_train[index_positive][0]
        full_imname_pos = os.path.join(self.root, impath_pos)
        img_positive = read_image(full_imname_pos)
        img_positive = Image.fromarray(img_positive)
        img_positive = self.transforms_positive(img_positive)
        
        return img_train, img_positive, class_id

    def __len__(self):
        return len(self.imlist)


class Product10KDatasetCombinedLoss(data.Dataset):
    def __init__(self, root, annotation_file, transforms, is_inference=False,
                 with_bbox=False, transforms_positive=None):
        self.root = root
        self.imlist = pd.read_csv(annotation_file)
        self.transforms = transforms
        self.is_inference = is_inference
        self.with_bbox = with_bbox
        self.transforms_positive = transforms_positive
        self.dict_train = {}
        self.dict_train_group = {}
        for i in range(9691):
            self.dict_train[i] = []
        for i in range(361):
            if i != 359:
                self.dict_train_group[i] = []
        for impath, class_id, group in self.imlist.iloc:
            self.dict_train[class_id].append((impath, group))
        for impath, class_id, group in self.imlist.iloc:
            self.dict_train_group[group].append(class_id)

    def __getitem__(self, index):
        cv2.setNumThreads(6)

        if self.is_inference:
            impath, _, _ = self.imlist.iloc[index]
        else:
            img_train, class_id, group = self.imlist.iloc[index]
            

        full_imname = os.path.join(self.root, img_train)
        img_train = read_image(full_imname)

        img_train = Image.fromarray(img_train)
        img_train = self.transforms_positive(img_train)
        idx = np.random.randint(3)

        if idx != 0:
            index_neg_group = np.random.randint(361)
            while index_neg_group == group or index_neg_group==359:
                index_neg_group = np.random.randint(361)
            
            list_class_neg = self.dict_train_group[index_neg_group]
            class_id_neg = np.random.choice(list_class_neg)
            list_class_id_train_neg = self.dict_train[class_id_neg]
            index_negative = np.random.randint(len(list_class_id_train_neg))
            impath_neg = list_class_id_train_neg[index_negative][0]
            full_imname_neg = os.path.join(self.root, impath_neg)
            img_negative = read_image(full_imname_neg)
            img_negative = Image.fromarray(img_negative)
            img_loss = self.transforms_positive(img_negative)
            y_true = 0.0
        
        else: 
            # prendo un'immagine positiva (stesso target)
            list_class_id_train = self.dict_train[class_id]
            index_positive = np.random.randint(len(list_class_id_train))
        
            impath_pos = list_class_id_train[index_positive][0]
            full_imname_pos = os.path.join(self.root, impath_pos)
            img_positive = read_image(full_imname_pos)
            img_positive = Image.fromarray(img_positive)
            img_loss = self.transforms_positive(img_positive)
            y_true = 1.0
        return img_train, img_loss, y_true, class_id

    def __len__(self):
        return len(self.imlist)
