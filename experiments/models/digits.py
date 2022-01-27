import torch

class DigitsBase(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(1)
        )
    
    def forward(self, x):
        return self.model(x)

class DigitsHead(torch.nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(9216, 128),
            torch.nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)