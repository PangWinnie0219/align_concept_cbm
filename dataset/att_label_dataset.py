import os
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader



class AttLabelDataset(Dataset):
    """
    Custom PyTorch Dataset for handling data with attributes.
    
    Args:
        df (pandas.DataFrame): DataFrame containing the dataset.
        attribute_encoders (dict): A dictionary where keys are attribute names and values are dictionaries mapping attribute values to unique integers.
        attribute_columns (list, optional): List of column names in the DataFrame corresponding to attributes. Defaults to None.
        image_dir (str, optional): Directory path where images are stored. Defaults to "".
        image_key (str, optional): Column name in the DataFrame containing image file names. Defaults to "path".
        label_key (str, optional): Column name in the DataFrame containing final class labels. Defaults to "label".
        transform (callable, optional): Optional transform to be applied to the image sample. Defaults to None.
        multiply (int, optional): Number of times dataset should be repeated for each epoch. Defaults to 1.
        loader (callable, optional): Function to load an image given its path. Defaults to torchvision's default_loader.
    """
    def __init__(self, df, attribute_encoders, label_encoder, attribute_columns=None, image_dir="", image_key="path", label_key="label", transform=None, multiply=1, loader=default_loader):
        self.df = df
        self.image_dir = image_dir
        self.image_key = image_key
        self.transform = transform
        self.multiply = multiply
        if attribute_columns is None:
            attribute_columns = list(attribute_encoders.keys())
        self.attribute_columns = attribute_columns
        self.attribute_encoders = attribute_encoders
        self.attribute_decoders = self.create_attribute_decoders(
            self.attribute_encoders)
        self.label_encoder = label_encoder
        self.label_key = label_key
        self.loader = loader  # Use default_loader from torchvision

    def __len__(self):
        return len(self.df) * self.multiply

    def create_attribute_encoders(self):
        attribute_encoders = {}
        for col in self.attribute_columns:
            attribute_encoders[col] = {
                value: idx for idx, value in enumerate(sorted(self.df[col].unique()))}
        return attribute_encoders
    
    def create_attribute_decoders(self, attribute_encoders):
        # Create reverse mapping for each dictionary
        attribute_decoders = {}
        for col, encoding in attribute_encoders.items():
            attribute_decoders[col] = {v: k for k, v in encoding.items()}
        return attribute_decoders

    def __getitem__(self, idx):
        idx = idx % len(self.df)

        image_path = os.path.join(
            self.image_dir, self.df.loc[idx, self.image_key])
        image = self.loader(image_path)

        attribute_values = [self.df.loc[idx, col]
                            for col in self.attribute_columns]
        # Convert attribute values to indices then to tensors
        attributes = torch.tensor([self.attribute_encoders[col][value] for col, value in zip(
            self.attribute_columns, attribute_values)], dtype=torch.long)
        
        label_value = self.df.loc[idx, self.label_key]
        label = torch.tensor(self.label_encoder[label_value])
        # one_hot_label = torch.nn.functional.one_hot(label, num_classes=len(self.label_encoder)).float()

        sample = {
            'image': image,
            'attributes': attributes,
            'attribute_values': attribute_values,
            'y_label': label,
            'img_path': image_path,
            'idx': idx,
        }

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample