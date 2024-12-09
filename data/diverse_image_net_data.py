# returns a dataloader that returns 5% of the ImageNet dataset

from .base import GetData
from abc import ABC, abstractmethod
from pathlib import Path
import os
import tarfile
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import json
import scipy.io
from PIL import Image
import random
import matplotlib.pyplot as plt

class DiverseImageNetDataset(Dataset):
    def __init__(self, image_paths, transform_hr=None, transform_lr=None):
        self.image_paths = image_paths
        self.transform_hr = transform_hr
        self.transform_lr = transform_lr

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")  # Ensure RGB format

        # Apply transformations for HR and LR images
        high_res_image = self.transform_hr(image) if self.transform_hr else image
        low_res_image = self.transform_lr(image) if self.transform_lr else image

        del image

        return low_res_image, high_res_image


class DiverseImageNetData(GetData):
    """
    The "input" is the low res image, and the "output" is the high res image
    """

    def gen_data(batch_size = 32, transform = None, dataset_portion=0.04):
        """
        params
            dataset_portion (float): the portion of the entire ImageNet dataset you want to use. The entire imagenet dataset used has 1,200,000 images.
        """
        HIGH_RES_DIM = 224
        LOW_RES_DIM = 64

        train_image_paths, val_image_paths = DiverseImageNetData._collect_images("data/datasets/ImageNet/train", dataset_portion)

        # Define HR transformations
        transform_hr = transforms.Compose([
            transforms.Lambda(lambda img: DiverseImageNetData._make_square(img)),  # Make the image square
            transforms.Resize(HIGH_RES_DIM),  # Resize to high resolution
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
        ])

        # Define LR transformations
        transform_lr = transforms.Compose([
            transforms.Lambda(lambda img: DiverseImageNetData._make_square(img)),  # Make the image square
            transforms.Resize(LOW_RES_DIM),  # Resize to low resolution
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
        ])

        # Create dataset and dataloader
        train_dataset = DiverseImageNetDataset(train_image_paths, transform_hr=transform_hr, transform_lr=transform_lr)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = DiverseImageNetDataset(val_image_paths, transform_hr=transform_hr, transform_lr=transform_lr)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        return train_dataloader, val_dataloader

    def _collect_images(folder, dataset_portion):
        """
        Collects a portion of the photos from each subfolder of ImageNet
        """
        train_image_paths = []
        val_image_paths = []
        for subfolder in os.listdir(folder):
            subfolder_path = os.path.join(folder, subfolder)
            if os.path.isdir(subfolder_path):  # Ensure it's a directory
                all_images = [
                    os.path.join(subfolder_path, img)
                    for img in os.listdir(subfolder_path)
                    if img.lower().endswith((".png", ".jpg", ".jpeg"))
                ]
                num_images = max(1, int(len(all_images) * dataset_portion))  # Select at least one image
                selected_images = random.sample(all_images, num_images)
                random.shuffle(selected_images)
                # 90% training, 10% validation
                split_index = int(len(selected_images) * 0.9)
                train_image_paths.extend(selected_images[:split_index])
                val_image_paths.extend(selected_images[split_index:])
        return train_image_paths, val_image_paths
    
    def _make_square(img):
        width, height = img.size
        max_dim = max(width, height)
        padding = (
            (max_dim - width) // 2,  # Left padding
            (max_dim - height) // 2,  # Top padding
            (max_dim - width) - (max_dim - width) // 2,  # Right padding
            (max_dim - height) - (max_dim - height) // 2,  # Bottom padding
        )
        return transforms.functional.pad(img, padding, fill=0)  # Pad with black (value 0)