import torch

from . import dataset, augmentations


def get_dataloaders(config):
    """
    Function for creating training and validation dataloaders
    :param config:
    :return:
    """
    print("Preparing train reader...")
    if config.dataset.triplete: 
        print ("---Triplete---")
        train_dataset = dataset.Product10KDataset(
            triplete=True, root=config.dataset.train_prefix, annotation_file=config.dataset.train_list, root_val=config.dataset.val_prefix, annotation_file_val=config.dataset.val_list_triplete, 
            transforms=augmentations.get_train_aug(config),  transforms_positive=augmentations.get_train_aug_positive(config)
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.dataset.batch_size,
            shuffle=True,
            num_workers=config.dataset.num_workers,
            drop_last=True
        )
        print("Done.")
        return train_loader, None
    
    elif config.dataset.group:
        print ("---DataLoaderGroup---")
        train_dataset = dataset.Product10KDatasetGroup(
            root=config.dataset.train_prefix, annotation_file=config.dataset.train_list,
            transforms=augmentations.get_train_aug(config)
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.dataset.batch_size,
            shuffle=True,
            num_workers=config.dataset.num_workers,
            pin_memory=True,
            drop_last=True
        )
        print("Done.")

        print("Preparing valid reader...")
        val_dataset = dataset.Product10KDatasetGroup(
            root=config.dataset.val_prefix, annotation_file=config.dataset.val_list,
            transforms=augmentations.get_val_aug(config), is_val=True
        )
        valid_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.dataset.batch_size,
            shuffle=False,
            num_workers=config.dataset.num_workers,
            drop_last=False,
            pin_memory=True
        )
        return train_loader, valid_loader

    elif config.dataset.contrastive:
        print ("---LossDataloader---")
        # train_dataset = dataset.Product10KDatasetLoss(
        #     root=config.dataset.train_prefix, annotation_file=config.dataset.train_list,root_val=config.dataset.val_prefix, annotation_file_val=config.dataset.val_list,
        #     transforms=augmentations.get_train_aug(config), transforms_positive=augmentations.get_train_aug_positive(config)
        # )
        # train_loader = torch.utils.data.DataLoader(
        #     train_dataset,
        #     batch_size=config.dataset.batch_size,
        #     shuffle=True,
        #     num_workers=config.dataset.num_workers,
        #     pin_memory=True,
        #     drop_last=True
        # )
        
        print("Done.")
        train_dataset = dataset.Product10KDatasetContrastive(
            root=config.dataset.val_prefix, annotation_file=config.dataset.val_list,
            transforms=augmentations.get_train_aug(config), transforms_positive=augmentations.get_train_aug_positive(config)
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.dataset.batch_size,
            shuffle=True,
            num_workers=config.dataset.num_workers,
            pin_memory=True,
            drop_last=True
        )
        print("Preparing valid reader...")
        
        print("Done.")
        return train_loader, None
    elif config.dataset.contrastive_all:
        print ("---ContrastiveAllDataloader---")

        train_dataset = dataset.Product10KDatasetContrastiveAll(
            root=config.dataset.train_prefix, annotation_file=config.dataset.train_list,
            transforms=augmentations.get_train_aug(config), transforms_positive=augmentations.get_train_aug_positive(config)
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.dataset.batch_size,
            shuffle=True,
            num_workers=config.dataset.num_workers,
            pin_memory=True,
            drop_last=True
        )
        print("Preparing valid reader...")
        
        valid_dataset = dataset.Product10KDatasetContrastive(
            root=config.dataset.val_prefix, annotation_file=config.dataset.val_list,
            transforms=augmentations.get_train_aug(config), transforms_positive=augmentations.get_train_aug_positive(config)
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=config.dataset.batch_size,
            shuffle=True,
            num_workers=config.dataset.num_workers,
            pin_memory=True,
            drop_last=True
        )
        print("Done.")

        return train_loader, valid_loader
    
    elif config.dataset.vit:
        print ("---VitDataloader---")

        train_dataset = dataset.Product10KDatasetContrastiveAll(
            root=config.dataset.train_prefix, annotation_file=config.dataset.train_list,
            transforms=augmentations.get_train_aug(config), transforms_positive=augmentations.get_train_aug_positive(config)
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.dataset.batch_size,
            shuffle=True,
            num_workers=config.dataset.num_workers,
            pin_memory=True,
            drop_last=True
        )
        print("Preparing valid reader...")
        
        valid_dataset = dataset.Product10KDatasetContrastive(
            root=config.dataset.val_prefix, annotation_file=config.dataset.val_list,
            transforms=augmentations.get_train_aug(config), transforms_positive=augmentations.get_train_aug_positive(config)
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=config.dataset.batch_size,
            shuffle=True,
            num_workers=config.dataset.num_workers,
            pin_memory=True,
            drop_last=True
        )
        print("Done.")
        return train_loader, valid_loader
    elif config.dataset.combinedLoss or config.dataset.autoencoder :
        print ("---CombinedLossDataloader---")

        train_dataset = dataset.Product10KDatasetCombinedLoss(
            root=config.dataset.train_prefix, annotation_file=config.dataset.train_list,
            transforms=augmentations.get_train_aug(config), transforms_positive=augmentations.get_train_aug_positive(config)
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.dataset.batch_size,
            shuffle=True,
            num_workers=config.dataset.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        return train_loader, None
