import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        # Calculate queries, keys, values
        q = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)  # B, HW, C/8
        k = self.key(x).view(batch_size, -1, H * W)  # B, C/8, HW
        v = self.value(x).view(batch_size, -1, H * W)  # B, C, HW
        
        # Attention map
        attention = torch.bmm(q, k)  # B, HW, HW
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to values
        out = torch.bmm(v, attention.permute(0, 2, 1))  # B, C, HW
        out = out.view(batch_size, C, H, W)
        
        return self.gamma * out + x

class LightSiamAttention(nn.Module):
    def __init__(self):
        super(LightSiamAttention, self).__init__()
        self.feature = nn.Sequential(
            # conv1
            nn.Conv2d(3, 64, 11, 2),
            nn.BatchNorm2d(64, eps=1e-6, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            
            # conv2
            nn.Conv2d(64, 64, 5, 1, groups=64),  # depthwise
            nn.Conv2d(64, 128, 1, 1),            # pointwise
            nn.BatchNorm2d(128, eps=1e-6, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            
            # conv3
            nn.Conv2d(128, 128, 3, 1),
            nn.BatchNorm2d(128, eps=1e-6, momentum=0.05),
            nn.ReLU(inplace=True),
            
            # Self-attention after conv3
            SelfAttention(128),
            
            # conv4
            nn.Conv2d(128, 128, 3, 1, groups=128),  # depthwise
            nn.Conv2d(128, 128, 1, 1),               # pointwise
            nn.BatchNorm2d(128, eps=1e-6, momentum=0.05),
            nn.ReLU(inplace=True),
            
            # conv5
            nn.Conv2d(128, 256, 3, 1, groups=2)
        )
        self._initialize_weights()

    def forward(self, z, x):
        # Same forward pass as original
        z = self.feature(z)
        x = self.feature(x)

        # fast cross correlation
        n, c, h, w = x.size()
        x = x.view(1, n * c, h, w)
        out = F.conv2d(x, z, groups=n)
        out = out.view(n, 1, out.size(-2), out.size(-1))

        # adjust the scale of responses
        out = 0.001 * out + 0.0

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data, mode='fan_out',
                                     nonlinearity='relu')
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()