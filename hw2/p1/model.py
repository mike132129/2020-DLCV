from torchvision import models
import torch.nn as nn
#model = models.vgg16(pretrained=True)
import pdb
class Classifier(nn.Module):
    
    def __init__(self, pretrained='vgg16'):
        super(Classifier, self).__init__()
        if pretrained == 'vgg16':
            self.model = models.vgg16(pretrained=True)
            self.model.classifier = nn.Sequential(nn.Linear(25088, 4096),
                                                     nn.BatchNorm1d(num_features=4096),
                                                     nn.ReLU(),
                                                     nn.Dropout(0.5),
                                                     nn.Linear(4096, 1024),
                                                     nn.BatchNorm1d(num_features=1024),
                                                     nn.ReLU(),
                                                     nn.Dropout(0.5),
                                                     nn.Linear(1024, 50)
                                                    )
        else:
            raise NotImplementedError

    def forward(self, x):
        
        x = self.model(x)
        
        return x
