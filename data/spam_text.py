import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch
from typing import Tuple

class SpamTextDataset(Dataset):
    def __init__(self, data):
        self.texts = data['text'].values
        self.labels = data['label'].values
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return torch.tensor(text, dtype=torch.long), torch.tensor(label, dtype=torch.long)

class SpamTextData:
    @staticmethod
    def get_data(batch_size=32) -> Tuple[DataLoader]:
        df = pd.read_csv("data/datasets/spam_text.tsv", delimiter='\t', header=None, names=['label', 'text'])

        # process the data
        df['text'] = df['text'].apply(SpamTextData.preprocess_text)
        df = df[['text', 'label']]

        # encode the labels from the text into a number between 0 and n_classes - 1 (inclusive)
        le = LabelEncoder()
        df['label'] = le.fit_transform(df['label'])

        # split the data into training and testing sets
        train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

        # encode the text into numbers
        vocab = set([word for phrase in df['text'] for word in phrase])
        word_to_idx = {word: idx for idx, word in enumerate(vocab, 1)}

        def encode_phrase(phrase):
            return [word_to_idx[word] for word in phrase]
        
        train_data['text'] = train_data['text'].apply(encode_phrase)
        test_data['text'] = test_data['text'].apply(encode_phrase)

        # padding sequences
        max_length = max(df['text'].apply(len))

        train_data['text'] = train_data['text'].apply(lambda x: SpamTextData.pad_sequence(x, max_length))
        test_data['text'] = test_data['text'].apply(lambda x: SpamTextData.pad_sequence(x, max_length))

        train_dataset = SpamTextDataset(train_data)
        test_dataset = SpamTextDataset(test_data)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader

    @staticmethod
    def pad_sequence(seq, max_length):
        """
        Adds padding to a sequence to make it the same length as max_length.
        """
        return seq + [0] * (max_length - len(seq))

    @staticmethod
    def preprocess_text(text):
        return text.lower().split()