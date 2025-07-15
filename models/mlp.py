import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, hidden_dims=[128, 64], input_size=784, num_classes=10, dropout=0.2):
        super(MLP, self).__init__()
        self.emb_dim = hidden_dims[-1]

        # store config for reproducibility and string representation
        self.cfg = dict(
            hidden_dims=hidden_dims,
            input_size=input_size,
            num_classes=num_classes,
            dropout=dropout
        )

        layers = []
        prev_dim = input_size
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = dim

        self.feature_extractor = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, num_classes)

        # Apply Kaiming Initialization
        self.apply(init_weights_kaiming)

    def forward(self, x, return_embedding=False):
        x = x.view(x.size(0), -1)
        h = self.feature_extractor(x)
        out = self.head(h)
        if return_embedding:
            return out, h
        return out

    def __repr__(self):
        # build a concise string from the config
        fields = ", ".join(f"{k}={v}" for k, v in self.cfg.items())
        return f"{self.__class__.__name__}({fields})"

def init_weights_kaiming(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)