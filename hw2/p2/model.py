from torchvision import models
import torch.nn as nn
import pdb

class Segmenter(nn.Module):

    def __init__(self, pretrained='vgg16', n_class=7):
        super(Segmenter, self).__init__()
        if pretrained == 'vgg16':
            self.model = models.vgg16(pretrained=True)
        else:
            raise NotImplementedError

        self.feature_extractor = self.model.features
        self.fcnn = nn.Sequential(
        #               nn.Conv2d(512, 4096, 7),
        #               nn.ReLU(inplace=True),
        #               nn.Dropout2d(),

        #               nn.Conv2d(4096, 4096, 1),
        #               nn.ReLU(inplace=True),
        #               nn.Dropout2d(),

        #               nn.Conv2d(4096, n_class, 1),
                        nn.ConvTranspose2d(512, n_class,
                                            36, stride=32,
                                            padding=2,
                                            bias=False))




        
    def forward(self, x):

        output = self.feature_extractor(x)
        fcnn_output = self.fcnn(output)
        # pdb.set_trace()

        return fcnn_output
