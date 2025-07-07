import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, conv_dims=[64, 128], kernel_sizes=[3, 3], hidden_dims=[128], num_classes=10, input_channels=3,
                 dropout=0.2, use_residual=True):
        super(CNN, self).__init__()
        self.emb_dim = hidden_dims[-1]
        self.use_residual = use_residual

        # Validate kernel size length
        if len(kernel_sizes) != len(conv_dims):
            raise ValueError(f"Expected kernel_sizes of length {len(conv_dims)}, got {len(kernel_sizes)}")

        # Construct residual blocks
        conv_layers = []
        prev_channels = input_channels

        for out_channels, kernel_size in zip(conv_dims, kernel_sizes):
            conv_layers.append(ResidualBlock(prev_channels, out_channels, kernel_size, stride=1,
                                             dropout=dropout, use_residual=use_residual))
            conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev_channels = out_channels

        self.conv_layers = nn.Sequential(*conv_layers)

        # Calculate input size for fully connected layers
        with torch.no_grad():
            dummy_input = torch.randn(1, input_channels, 32, 32)
            dummy_output = self.conv_layers(dummy_input)
            fc_input_dim = dummy_output.view(1, -1).shape[1]

        # Construct fully connected layers
        fc_layers = []
        prev_dim = fc_input_dim

        for dim in hidden_dims:
            fc_layers.append(nn.Linear(prev_dim, dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout))
            prev_dim = dim

        self.fc_layers = nn.Sequential(*fc_layers)
        self.head = nn.Linear(prev_dim, num_classes)

        self.cfg = dict(conv_dims=conv_dims,
                        kernel_sizes=kernel_sizes,
                        hidden_dims=hidden_dims,
                        dropout=dropout,
                        residual=use_residual)

    def __repr__(self):
        fields = ", ".join(f"{k}={v}" for k,v in self.cfg.items())
        return f"{self.__class__.__name__}({fields})"

    def forward(self, x, return_embedding=False):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        h = self.fc_layers(x)
        out = self.head(h)

        if return_embedding:
            return out, h
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout=0.2, use_residual=True):
        super(ResidualBlock, self).__init__()
        padding = kernel_size // 2
        self.use_residual = use_residual

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Ensure residual connection can handle dimension change
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride) if in_channels != out_channels else None

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))

        # Apply residual connection
        if self.use_residual:
            if self.residual_conv:
                identity = self.residual_conv(identity)
            out += identity

        out = self.relu(out)
        return out