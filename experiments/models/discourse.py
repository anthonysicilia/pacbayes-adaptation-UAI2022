import torch

class DiscourseBase(torch.nn.Module):

    def __init__(self, num_inputs=768):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, 512),
            torch.nn.Dropout(),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.Dropout(),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.Dropout(),
            torch.nn.ReLU())
    
    def forward(self, x):
        return self.model(x)

class DiscourseHead(torch.nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()
        self.model = torch.nn.Linear(256, num_classes)
    
    def forward(self, x):
        return self.model(x)