from abc import *
import torch
import torch.nn as nn

from .builder import Builder

class SimclrLayer(nn.Module):
    def __init__(self, last_dim, simclr_dim):
        super(SimclrLayer, self).__init__()

        self.fc1 = nn.Linear(last_dim, last_dim)
        self.relu = nn.ReLU()
        self.last = nn.ModuleList()

        self.last = nn.Linear(last_dim, simclr_dim)

    def forward(self, features):
        out = self.fc1(features)
        out = self.relu(out)
        out = self.last(out)
        return out

class BaseModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, last_dim, num_classes=20, simclr_dim=128):
        super(BaseModel, self).__init__()
        builder = Builder()

        self.last_dim = last_dim

        self.joint_distribution_layer = builder.conv1x1(last_dim, 4 * num_classes, last_layer=True)

        self.linear = nn.Linear(last_dim, num_classes)
        self.simclr_layer = SimclrLayer(last_dim, simclr_dim)
        
    @abstractmethod
    def penultimate(self,inputs):
        pass

    def forward(self, inputs, penultimate=False, simclr=False, joint=False):
        _aux = {}
        _return_aux = False

        features = self.penultimate(inputs)


        output = self.linear(features.view(-1, self.last_dim))

        if penultimate:
            _return_aux = True
            _aux['penultimate'] = features

        if simclr:
            _return_aux = True
            out = self.simclr_layer(features.view(-1, self.last_dim))
            _aux['simclr'] = out

        if joint:
            _return_aux = True
            out = self.joint_distribution_layer(features)
            _aux['joint'] = out.view(out.size(0), -1)

        if _return_aux:
            return output, _aux
        return output

