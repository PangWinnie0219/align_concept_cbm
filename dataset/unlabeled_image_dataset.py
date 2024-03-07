import os
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

class UnlabeledImageDataset(Dataset):
    def __init__(self, df, label_encoder, image_dir="", image_key="path",  label_key="label", transform=None, multiply=1, loader=default_loader):
        '''
        df: pandas dataframe of this dataset
        img_key: column name of storing image path in the dataframe
        transform: preprocessing image transformation
        multiply: repeat the dataset with multiple copies of itself
        '''
        self.df = df
        self.image_dir = image_dir
        self.image_key = image_key
        self.label_encoder = label_encoder
        self.label_key = label_key
        self.transform = transform
        self.multiply = multiply
        self.loader = loader

    def __len__(self):
        return len(self.df) * self.multiply

    def __getitem__(self, idx):
        idx = idx % len(self.df)
        image_path = os.path.join(
            self.image_dir, self.df.loc[idx, self.image_key])
        image = self.loader(image_path)

        label_value = self.df.loc[idx, self.label_key]
        label = torch.tensor(self.label_encoder[label_value])

        sample = {
            'image': image, 
            'y_label': label,
            'img_path': image_path, 
            'idx': idx
        }

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample
