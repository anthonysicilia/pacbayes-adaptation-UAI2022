import torch
import torchvision as tv

class ImagesBase(torch.nn.Module):

    def __init__(self, num_inputs=2048):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, 1024),
            torch.nn.Dropout(),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.Dropout(),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.Dropout(),
            torch.nn.ReLU())
    
    def forward(self, x):
        return self.model(x)

class ImagesHead(torch.nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()
        self.model = torch.nn.Linear(256, num_classes)
    
    def forward(self, x):
        return self.model(x)

# Can Try Full Resnet if Time Permits

class ResNetBase(torch.nn.Module):

    def __init__(self, resnet):
        super().__init__()
        self.model = resnet(pretrained=True)
        self.model.fc = torch.nn.Identity()

    def forward(self, x):
        return self.model(x)

class ResNetHead(torch.nn.Module):

    def __init__(self, resnet, num_classes):
        super().__init__()
        num_inputs = resnet(pretrained=True).fc.in_features
        self.model = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, 256),
            torch.nn.Dropout(),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.Dropout(),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.Dropout(),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

class ResNet18Base(ResNetBase):

    def __init__(self):
        super().__init__(self, tv.models.resnet18)

class ResNet50Base(ResNetBase):

    def __init__(self):
        super().__init__(self, tv.models.resnet50)

class ResNet18Head(ResNetHead):

    def __init__(self, num_classes=10):
        super().__init__(self, tv.models.resnet18, num_classes)

class ResNet50Head(ResNetHead):

    def __init__(self, num_classes=10):
        super().__init__(self, tv.models.resnet50, num_classes)