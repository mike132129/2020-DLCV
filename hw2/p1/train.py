import sys
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from model import Classifier
from data import get_dataloader
import pdb
import torch.optim.lr_scheduler

torch.manual_seed(1320)
def train(trainloader, validloader):
    
    criterion = nn.CrossEntropyLoss()
    model = Classifier().cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) # SGD
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # for i in range(50):
    #     for para in optimizer.param_groups:
    #         scheduler.step()
    #         print(para['lr'])
    # exit()
    for epoch in range(80):
        print('Epoch', epoch)

        
        train_loss = 0

        for batch, (x, label) in enumerate(trainloader, 1):

            x, label = x.cuda(), label.cuda()
            model.train()
            model.zero_grad()
            output = model(x)
            loss = criterion(output, label)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            if batch % 300 == 0:
                print('Loss: {}'.format(train_loss/batch))

        # scheduler.step()

        with torch.no_grad():
            total_loss = 0
            model.eval()
            correct = 0
            total = 0
            for batch, (x, label) in enumerate(validloader, 1):
                x, label = x.cuda(), label.cuda()
                output = model(x)
                loss = criterion(output, label)
                total_loss += loss.item()
                pred = output.max(dim=1)[1]
                correct += (pred == label).sum().item()
                total += pred.size(0)
        print('EPOCH: {}, validloss: {}, Acc: {}'.format(epoch, total_loss/batch, correct/total))



if __name__ == '__main__':
    
    folder = sys.argv[1]
    trainloader, validloader = get_dataloader(folder)
    train(trainloader, validloader)
