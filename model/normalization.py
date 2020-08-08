import torch
import torch.nn as nn


class cLN(nn.Module):
    def __init__(self, dim):
        super(cLN, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1, dim, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, input):  # input :[batch_size, channels, time_step]
        '''
        cumulate over time.
        '''
        batch_size, channels, time = input.shape
        sum_over_step = torch.cumsum(input.sum(dim=1), dim=-1)
        pow_sum_over_step = torch.cumsum(input.pow(2).sum(dim=1), dim=-1)

        time_step = torch.linspace(1, time + 1, steps=time)
        mean = sum_over_step / (time_step * channels)
        var = (pow_sum_over_step -
               2 * sum_over_step * mean) / (time_step * channels) + mean.pow(2)
        mean = mean.unsqueeze(1)
        var = var.unsqueeze(1)
        input = (input - mean) / (var + 1e-8).sqrt()
        return input * self.gamma + self.beta


class gLN(nn.Module):
    def __init__(self, dim):
        super(gLN, self).__init__()
        self.gamma = nn.Parameter(torch.ones([1, dim, 1]))
        self.beta = nn.Parameter(torch.zeros([1, dim, 1]))

    def forward(self, input):  # input :[batch_size, channels, time_step]
        batch_size, channels, time = input.shape
        mean = input.mean(dim=[1, 2], keepdim=True)
        var = (input - mean).pow(2).mean(dim=[1, 2], keepdim=True)
        input = (input - mean) / (var + 1e-8).sqrt()
        return input * self.gamma + self.beta
