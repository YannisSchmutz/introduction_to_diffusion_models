import torch.nn as nn


class Upsample(nn.Module):

    def __init__(self, ch):
        """
        :param ch (int): number of input and output channels
        """
        super(Upsample, self).__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=1, padding=1)

    def forward(self, x):
        bs, ch, h, w = x.shape

        x = nn.functional.interpolate(x, size=None, scale_factor=2, mode='nearest')

        x = self.conv(x)
        assert x.shape == (bs, ch, h * 2, w * 2)
        return x
