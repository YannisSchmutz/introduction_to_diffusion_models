import torch.nn as nn


class Downsample(nn.Module):

    def __init__(self, ch):
        """
        :param ch (int): number of input and output channels
        """
        super(Downsample, self).__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        bs, ch, h, w = x.shape
        x = self.conv(x)
        assert x.shape == (bs, ch, h // 2, w // 2)
        return x
