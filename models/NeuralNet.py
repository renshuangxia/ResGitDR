import torch.nn as nn
import torch



class NN(nn.Module):
    def __init__(self, InputSize, hiddenSizes, OutputSize):
        super(NN, self).__init__()
        self.InputSize = InputSize
        self.hiddenSizes = hiddenSizes
        self.OutputSize = OutputSize
        self.hiddenLayers = nn.ModuleList()
        self.t = nn.Tanh()

        self.hiddenLayers.append(nn.Linear(self.InputSize, self.hiddenSizes[0], bias=True))
        for i in range(1, len(self.hiddenSizes)):
            self.hiddenLayers.append(nn.Linear(self.hiddenSizes[i - 1], self.hiddenSizes[i], bias=True))
        self.hiddenLayers.append(nn.Linear(self.hiddenSizes[-1], self.OutputSize, bias=True))

    def forward(self, x):
        hiddens = []
        for j in range(len(self.hiddenLayers) - 1):
            x = self.t(self.hiddenLayers[j](x.float()))
            hiddens.append(x)
        x = self.hiddenLayers[-1](x)
        return x, hiddens