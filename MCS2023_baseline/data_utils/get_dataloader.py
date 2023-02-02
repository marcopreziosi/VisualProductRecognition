import torch

from . import dataset, augmentations


def get_dataloaders(config):
    """
    Function for creating training and validation dataloaders
    :param config:
    :return:
    """

    train_dataset = dataset.Product10KDataset(
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
    val_dataset = dataset.Product10KDataset(
        root=config.dataset.val_prefix, annotation_file=config.dataset.val_list,
        transforms=augmentations.get_val_aug(config)
    )
    valid_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
        drop_last=False,
        pin_memory=True
    )
    print("Done.")
    return train_loader, valid_loader
