import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class FC_ResNet(nn.Module):

    def __init__(self, model, num_classes, selu):
        super(FC_ResNet, self).__init__()

        # feature encoding
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4)

        # classifier
        num_features = model.layer4[1].conv1.in_channels
        print('selu: ',selu)
        if selu:
            self.classifier = nn.Sequential(
                nn.Conv2d(num_features, num_classes, kernel_size=1, bias=True), nn.SELU())
        else:
            self.classifier = nn.Sequential(
                nn.Conv2d(num_features, num_classes, kernel_size=1, bias=True))

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        #print('relu')
        return x


class FC_VGG16(nn.Module):

    def __init__(self, model, num_classes):
        super(FC_VGG16, self).__init__()

        # feature encoding
        # self.features = nn.Sequential(
        #     model.features)
        self.features=nn.Sequential(*list(model.features.children())[:-1])
        # classifier
        num_features = 512

        self.classifier = nn.Sequential(
            nn.Conv2d(num_features, num_classes, kernel_size=1, bias=True))
 

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        #print('relu')
        return x


class FC_VGG16_2(nn.Module):

    def __init__(self, model, num_classes):
        super(FC_VGG16_2, self).__init__()

        # feature encoding
        # self.features = nn.Sequential(
        #     model.features)
        self.features=nn.Sequential(*list(model.features.children())[:])
        # classifier
        num_features = 512

        self.classifier = nn.Sequential(
            nn.Conv2d(num_features, num_classes, kernel_size=1, bias=True))
 

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        #print('relu')
        return x