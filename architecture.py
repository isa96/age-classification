import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SimpleAttentionCNN(nn.Module):
    def __init__(self, in_channel, in_size, num_classes, spatial_attention=True, channel_attention=True):
        super(SimpleAttentionCNN, self).__init__()

        self.spatial_attention = spatial_attention
        self.channel_attention = channel_attention
        self.in_channel = in_channel
        self.in_size = in_size
        # Convolutional layers

        self.block = nn.Sequential(
            nn.Conv2d(in_channel, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, 1, 1),
        )

        # Spatial Attention
        if spatial_attention:
            self.spatial_attention_layer = SpatialAttention()

        # Channel Attention
        if channel_attention:
            self.channel_attention_layer = ChannelAttention(256)

        # Fully connected layers
        self.fc1 = nn.Linear(self.in_size, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        logits = self.block(x)
        # Apply Spatial Attention
        if self.spatial_attention:
            logits = self.spatial_attention_layer(logits)
        # Apply Channel Attention
        if self.channel_attention:
            logits = self.channel_attention_layer(logits)

        logits = logits.view(-1, self.in_size)
        logits = F.relu(self.fc1(logits))
        logits = self.fc2(logits)

        # return F.log_softmax(x, dim=1)
        return x

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.concat([avg_pool, max_pool], dim=1)
        out = self.conv(combined)
        return x * self.sigmoid(out)

class ChannelAttention(nn.Module):
    def __init__(self, in_channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channel, in_channel//reduction, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channel//reduction, in_channel, 1, bias=False)
        )
        self.sig = nn.Sigmoid()
    def forward(self, x):
        avgOut = self.fc(self.avg_pool(x))
        maxOut = self.fc(self.max_pool(x))
        out = avgOut + maxOut
        return x * self.sig(out)