from torch import nn
from torch.nn import functional as F
from torchvision import models
from nest import register


@register
def lenet5(num_classes: int = 10) -> nn.Module:
    """LeNet5.
    """

    class LeNet5(nn.Module):
        def __init__(self, num_classes=10):
            super(LeNet5, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 10, kernel_size=5),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.ReLU(),
                nn.Conv2d(10, 20, kernel_size=5),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.ReLU()
            )
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(320, 50),
                nn.Linear(50, num_classes)
            )

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), 320)
            x = self.classifier(x)
            return x

    return LeNet5(num_classes)


@register
def torchvision_model(
    model: str, 
    num_classes: int = 1000, 
    pretrained: bool = False) -> nn.Module:
    """Models from torchvision.
    """

    if hasattr(models, model):
        return getattr(models, model)(pretrained=pretrained, num_classes=num_classes)
    else:
        raise NotImplementedError('Invalid model name "%s".' % model)
