import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

#Create Flood class:
class FloodDataset(Dataset):
    def __init__(self, image_dir,mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)/255.0
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)/255.0
        mask = np.expand_dims(mask, axis=0)
        
        if self.transform:
            image = self.transform(image)
        
        return torch.tensor(image, dtype=torch.float32) , torch.tensor(mask, dtype=torch.float32)   
        