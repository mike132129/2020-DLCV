import sys
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from dataset import get_dataloader
from model import Segmenter
import pdb
from tqdm import tqdm
import torch.optim.lr_scheduler
from metric import IOUMetric
from PIL import Image
from eval import read_masks, mean_iou_score
import numpy as np

torch.manual_seed(1320)

def train(trainloader, validloader, colormap):
    model = Segmenter(n_class=7).cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.NLLLoss2d()
    activation = nn.LogSoftmax(dim=1)
    metri = IOUMetric(num_classes=7)
    colormap = np.array(colormap)
     
    for epoch in range(150):

        print('Epoch', epoch)

        train_loss = 0

        for batch, (x, label) in enumerate(tqdm(trainloader), 1):
            x, label = x.cuda(), label.cuda()
            model.train()
            model.zero_grad()

            output = model(x)
            loss = criterion(activation(output), label.long())
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                print('Loss: {}'.format(train_loss/batch))

        with torch.no_grad():
            valid_loss = 0
            model.eval()

            for batch, (x, label) in enumerate(validloader, 1):
                x, label = x.cuda(), label.cuda()
                output = model(x)
                output = activation(output)
                loss = criterion(activation(output), label.long())
                valid_loss += loss.item()
                predictions = output.argmax(dim=1).detach().cpu().numpy()[0]
                predictions = colormap[predictions]
                img = Image.fromarray(predictions.astype('uint8'))
                img = img.save('./data/p2_data/predict/' + str(batch) + '.png')
        
        evaluate()

        print('Epoch: {}, ValidLoss: {}'.format(epoch, valid_loss/batch))

def evaluate():
    # each epoch
    pred = read_masks('data/p2_data/predict/')
    label = read_masks('data/p2_data/validation/')

    mean_iou_score(pred, label)


if __name__ == '__main__':
    
    folder = sys.argv[1]
    trainloader, validloader, colormap = get_dataloader(folder)
    train(trainloader, validloader, colormap)
