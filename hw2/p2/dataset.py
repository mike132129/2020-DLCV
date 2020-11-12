import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob, os
import pdb
import numpy as np


def get_dataloader(folder, batch_size=8):
    trans = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
            ])
    train_path, valid_path = os.path.join(folder, 'train/'), os.path.join(folder,'validation/')
    trainset = ImageDataset(train_path, trans)
    validset = ImageDataset(valid_path, trans)
    trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
    validloader = DataLoader(dataset=validset, batch_size=1, shuffle=False)
    return trainloader, validloader, trainset.colormap

class ImageDataset(Dataset):
    def __init__(self, f_path, transform):
        super().__init__()
        sat, mask = [], []
        file_list = glob.glob(f_path + '*')
        file_list = sorted(file_list, key=lambda x: (float(x.split('/')[3].split('_')[0]), x))

        
        for i in range(len(file_list)):
            if i % 2 == 0:
                mask.append(file_list[i])
            else:
                sat.append(file_list[i])

        sat = self._get_image_from_list(sat)
        mask = self._get_image_from_list(mask)

        self.sat = sat
        self.mask = mask
        self.transform = transform
        self.colormap = [
            [0, 255, 255],    # urban
            [255, 255, 0],    # agriculture
            [255, 0, 255],    # rangeland
            [0, 255, 0],      # forest
            [0, 0, 255],      # water
            [255, 255, 255],  # barren
            [0, 0, 0]]        # unknown

        self.cm2lbl = np.zeros(256 ** 3)
        for i, cm in enumerate(self.colormap):
            self.cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i

    def __getitem__(self, idx):
        
        sat, mask = self.sat[idx], self.mask[idx]
        
        sat = self.transform(sat)
        mask = self._mask2label(mask)
        return sat, mask

    def _mask2label(self, mask):

        label = np.zeros((len(mask), len(mask[1])))
        idx = (mask[:, :, 0] * 256 + mask[:, :, 1]) * 256 + mask[:, :, 2]
            
        return torch.tensor(self.cm2lbl[idx])

    def _get_image_from_list(self, file_list):

        images = []
        for file in file_list:
            image = Image.open(file)
            images.append(np.array(image))

        return np.array(images)

    def __len__(self):
        return len(self.sat)