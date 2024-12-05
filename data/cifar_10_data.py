from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as T
from typing import Optional, Tuple, List
from .base import GetData

class CIFAR10Data(GetData):
    @staticmethod
    def gen_data(batch_size: int = 32, transforms: Optional[List] = None) -> Tuple[DataLoader, DataLoader]:
        """
        Returns a tuple of train_dataloader, test_dataloader containing the corresponding data
        
        params:
            batch_size (int): The batch size of the data.
            transforms (Optional[List]): A list of torchvision transforms to be applied to the data.
        """
        if transforms is None:
            transforms = [T.ToTensor()]
        transform = T.Compose(transforms)
        
        # Load training data
        trainset = datasets.CIFAR10(root='./data/datasets', train=True, download=True, transform=transform)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        
        # Load test data
        testset = datasets.CIFAR10(root='./data/datasets', train=False, download=True, transform=transform)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
        
        return trainloader, testloader
