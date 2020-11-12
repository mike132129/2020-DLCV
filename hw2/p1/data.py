import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob, os
import pdb
import numpy as np

def get_dataloader(folder, batch_size=32):
    trans = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
            ])
    train_path, valid_path = os.path.join(folder, 'train_50/'), os.path.join(folder,'val_50/')
    trainset = ImageDataset(train_path, trans)
    validset = ImageDataset(valid_path, trans)
    trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
    validloader = DataLoader(dataset=validset, batch_size=batch_size, shuffle=True)
    return trainloader, validloader

def get_data_from_folder(path):

    images = []
    for file in globl.glob(path + '*'):
        image = Image.open(file)
        label = file.split('/')[3].split('_')[0]

class ImageDataset(Dataset):
    def __init__(self, f_path, transform):
        super().__init__()
        images, labels = [], []

        for file in glob.glob(f_path + '*'):
            image = Image.open(file)
            label = file.split('/')[3].split('_')[0]
            images.append(np.array(image))
            labels.append(int(label))

        self.images = np.array(images)
        self.labels = np.array(labels)
        self.transform = transform

    def __getitem__(self, idx):
        
        image, label = self.images[idx], self.labels[idx]
        image = self.transform(image)
        label = torch.tensor(label)
        return image, label

    def __len__(self):
        return len(self.images)












            
