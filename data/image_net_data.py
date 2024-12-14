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

class ImageDataset(Dataset):
    def __init__(self, parent_folder, subfolders, transform=None):
        self.data = []
        self.labels = []
        self.transform = transform or transforms.ToTensor()  # Default to ToTensor if no transform provided

        for label, subfolder in enumerate(subfolders):
            folder_path = os.path.join(parent_folder, subfolder)
            for filename in os.listdir(folder_path):
                if filename.lower().endswith((".jpeg", ".jpg")):
                    image_path = os.path.join(folder_path, filename)
                    self.data.append(image_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx]
        label = self.labels[idx]

        # Open the image
        image = Image.open(image_path).convert("RGB")

        # Apply transformations (default or custom)
        image = self.transform(image)

        return image, label

class ImageNetData(GetData):
    """
    Concrete class for preparing and loading the ImageNet dataset.
    """

    @staticmethod
    def load_and_convert_mat_to_json(mat_file_path="data/datasets/ImageNet/meta.mat", json_file_path="data/datasets/ImageNet/meta.json"):
        """
        Checks if the json version of the metadata for ImageNet exists. If not, then create it.
        Exists because json files easier to read than .mat files
        """
        # Check if the JSON file already exists
        if os.path.exists(json_file_path):
            print(f"The ImageNet metadata JSON file already exists at: {json_file_path}")
        else:
            # Load the .mat file
            mat_contents = scipy.io.loadmat(mat_file_path)
            synsets_data = mat_contents['synsets']

            # Convert the 'synsets' data to a list of dictionaries
            def synsets_to_dict(synsets_array):
                synsets_dict = []
                for entry in synsets_array:
                    synset = {
                        'ILSVRC2012_ID': int(entry[0][0][0]),
                        'WNID': entry[0][1][0],
                        'words': entry[0][2][0],
                        'gloss': entry[0][3][0],
                        'num_children': int(entry[0][4][0][0]),
                        'children': entry[0][5][0].tolist() if entry[0][5].size > 0 else [],
                        'wordnet_height': int(entry[0][6][0][0]),
                        'num_train_images': int(entry[0][7][0][0])
                    }
                    synsets_dict.append(synset)
                return synsets_dict

            # Convert the synsets data
            synsets_list_of_dicts = synsets_to_dict(synsets_data)

            # Save the converted data to a JSON file
            with open(json_file_path, 'w') as f:
                json.dump(synsets_list_of_dicts, f, indent=4)

            print(f"JSON ImageNet metadata file created at: {json_file_path}")

    @staticmethod
    def examine_data():
        # File paths
        json_file_path = 'data/datasets/ImageNet/meta.json'  # Path to your JSON file
        dataset_folder = 'data/datasets/ImageNet/train'  # Path to your dataset folder

        # Load the JSON file with synsets data
        with open(json_file_path, 'r') as f:
            synsets = json.load(f)

        # Get the WNID values from the folders in the dataset
        existing_wnids = [d for d in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, d))]

        # Select 12 random WNID values from the existing ones
        random_wnids = random.sample(existing_wnids, 18)

        # Create a dictionary for quick lookup of WNID to "words" value
        wnid_to_words = {synset['WNID']: synset['words'] for synset in synsets}

        # Prepare the figure
        fig, axes = plt.subplots(3, 6, figsize=(15, 7))
        axes = axes.flatten()

        for i, wnid in enumerate(random_wnids):
            subfolder_path = os.path.join(dataset_folder, wnid)

            # Get a list of valid image files in the subfolder
            valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
            image_files = [
                f for f in os.listdir(subfolder_path)
                if f.lower().endswith(valid_extensions) and os.path.isfile(os.path.join(subfolder_path, f))
            ]

            if image_files:
                # Select a random image
                random_image = random.choice(image_files)
                image_path = os.path.join(subfolder_path, random_image)

                try:
                    # Load and display the image
                    img = Image.open(image_path)
                    axes[i].imshow(img)
                    axes[i].axis('off')
                    # Set title using the "words" value if available
                    title = wnid_to_words.get(wnid, wnid)  # Fallback to WNID if words not found
                    axes[i].set_title(title, fontsize=8)
                except Exception as e:
                    axes[i].axis('off')
                    axes[i].set_title(f"Error loading image in {wnid}", fontsize=8)
                    print(f"Error loading image: {image_path}\n{e}")
            else:
                axes[i].axis('off')
                axes[i].set_title(f"No images in {wnid}", fontsize=8)

        # Adjust layout
        plt.tight_layout()
        plt.show()

    @staticmethod
    def gen_data(batch_size: int = 32, classes=None, transform=None) -> pd.Series:

        # TODO: put logic here to make it so if classes = None, then that means get a dataloader with all the classes
        assert classes != None

        parent_folder = "data/datasets/ImageNet/train"
        dataset = ImageDataset(parent_folder, classes, transform=transform)

        # Calculate lengths for training and validation splits
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size

        # Split the dataset into training and validation sets
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

