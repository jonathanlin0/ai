from abc import ABC, abstractmethod
import pandas as pd
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from typing import Optional, Tuple, List

class GetData(ABC):
    """
    Abstract base class for retrieving data.
    """
    @staticmethod
    @abstractmethod
    def gen_data(batch_size: int = 32, transform: Optional[List[transforms.Compose]] = None) -> Tuple[DataLoader]:
        """
        Returns a tuple of train_dataloader, test_dataloader containing the corresponding data
        
        params
            batch_size (int): The batch size of the data.
            transforms
        """
