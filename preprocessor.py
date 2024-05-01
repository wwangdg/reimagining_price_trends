import pandas as pd
import numpy as np
import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


# 自定义数据集类
class ImageDataset(Dataset):
    """
    Custom Dataset Class: Designed to be versatile, this class is suitable for both binary classification and regression problems. It facilitates the handling and processing of data specific to each type of task.
    """
    
    def __init__(self, image_files, label_files, transform=None, label_name='Ret_20d', bool_regression=False):
        self.image_files = image_files
        self.label_files = label_files
        self.label_name = label_name
        self.bool_regression = bool_regression
        self.transform = transform
        
        self.images = []
        self.labels = []
        
        for img_f, lbl_f in zip(image_files, label_files):
            img = np.memmap(img_f, dtype=np.uint8, mode='r').reshape((-1, 1, 64, 60))
            lbl = pd.read_feather(lbl_f)[self.label_name].values
            
            valid_idx = ~np.isnan(lbl) 
            self.images.append(img[valid_idx])
            self.labels.append(lbl[valid_idx])
        
        self.images = np.concatenate(self.images, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)
        
        if not self.bool_regression:
            self.labels = np.where(self.labels < 0, 0, 1).astype(np.int64)
        
        self.images = torch.tensor(self.images, dtype=torch.float32)
        if not self.bool_regression:
            self.labels = torch.tensor(self.labels, dtype=torch.long)
        else:
            self.labels = torch.tensor(self.labels, dtype=torch.float32)
        
    def __getitem__(self, index):
        x = self.images[index]
        y = self.labels[index]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y
    
    def __len__(self):
        return len(self.labels)

def load_data(batch_size=128, num_workers=4, transform=None, labelName='Ret_20d', bool_regression=False):
    """
    Divides the entire dataset into training, validation, and testing samples.Adheres to the methodology outlined in the original paper: Uses the first eight years of data (1993-2000) for training and validating the model,with 70% of this period randomly selected for training and the remaining 30% for validation. The subsequent 19 years of data are designated as the out-of-sample test dataset.
    """
    
    DATA_DIR = "./img_data/monthly_20d"
    
    image_files = sorted([os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('_images.dat')])
    label_files = sorted([os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('_labels_w_delay.feather')])
    
    split_index = 8
    
    dataset = ImageDataset(image_files[:split_index], label_files[:split_index], transform=transform, label_name=labelName, bool_regression=bool_regression)
    train_size = int(0.7 * len(dataset))
    valid_size = len(dataset) - train_size
    
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
    
    test_dataset = ImageDataset(image_files[split_index:], label_files[split_index:], transform=transform, label_name=labelName, bool_regression=bool_regression)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    print(f"Train size: {len(train_loader.dataset)}")
    print(f"Validation size: {len(valid_loader.dataset)}")
    print(f"Test size: {len(test_loader.dataset)}")
    
    return train_loader, valid_loader, test_loader