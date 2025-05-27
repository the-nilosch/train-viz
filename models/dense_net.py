import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, dropout):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        out = self.dropout(out)
        return torch.cat([x, out], 1)

class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, dropout):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(DenseLayer(in_channels, growth_rate, dropout))
            in_channels += growth_rate
        self.block = nn.Sequential(*layers)
        self.out_channels = in_channels  # after growth

    def forward(self, x):
        return self.block(x)

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.layer(x)

class DenseNet(nn.Module):
    def __init__(
        self,
        block_config=(6, 12, 24),  # number of layers per dense block
        growth_rate=12,
        num_classes=10,
        input_channels=3,
        fc_hidden_dims=[],
        dropout=0.2,
    ):
        super().__init__()

        num_init_features = 2 * growth_rate
        self.conv1 = nn.Conv2d(input_channels, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)

        channels = num_init_features
        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()

        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers, channels, growth_rate, dropout)
            self.blocks.append(block)
            channels = block.out_channels
            if i != len(block_config) - 1:
                trans = TransitionLayer(channels, channels // 2)
                self.transitions.append(trans)
                channels = channels // 2

        self.bn_final = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.embedding_dim = channels
        fc_layers = []
        prev_dim = self.embedding_dim
        for dim in fc_hidden_dims:
            fc_layers.append(nn.Linear(prev_dim, dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout))
            prev_dim = dim
        self.fc_layers = nn.Sequential(*fc_layers)
        self.head = nn.Linear(prev_dim, num_classes)
        self.emb_dim = prev_dim

    def forward(self, x, return_embedding=False):
        x = self.conv1(x)
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i < len(self.transitions):
                x = self.transitions[i](x)
        x = self.relu(self.bn_final(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        h = self.fc_layers(x)
        out = self.head(h)

        if return_embedding:
            return out, h
        return out
