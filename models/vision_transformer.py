import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, emb_dim=128, in_channels=3):
        super().__init__()
        assert img_size % patch_size == 0

        self.num_patches = (img_size // patch_size) ** 2
        self.projection = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, emb_dim))

        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.projection(x).flatten(2).transpose(1, 2)  # [B, num_patches, emb_dim]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, emb_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, num_patches+1, emb_dim]
        x = x + self.pos_embedding
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, emb_dim=128, num_heads=4, depth=6, mlp_dim=256, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        return self.transformer(x)

class ViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, num_classes=10, emb_dim=128, depth=6, num_heads=4, mlp_dim=256, dropout=0.1, input_channels=3):
        super().__init__()
        self.emb_dim = emb_dim
        self.patch_embed = PatchEmbedding(img_size, patch_size, emb_dim, in_channels=input_channels)
        self.transformer_encoder = TransformerEncoder(emb_dim, num_heads, depth, mlp_dim, dropout)
        self.head = nn.Linear(emb_dim, num_classes)
        self.norm = nn.LayerNorm(emb_dim)

        # Initialization
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

        # store config
        self.cfg = dict(
            img_size=img_size,
            patch_size=patch_size,
            num_classes=num_classes,
            emb_dim=emb_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            input_channels=input_channels
        )

    def forward(self, x, return_embedding=False):
        x = self.patch_embed(x)  # [B, num_patches + 1, emb_dim]
        x = self.transformer_encoder(x)  # [B, num_patches + 1, emb_dim]
        cls_token = self.norm(x[:, 0])  # [B, emb_dim]
        out = self.head(cls_token)  # [B, num_classes]
        return (out, cls_token) if return_embedding else out

    def __repr__(self):
        # build a concise string from the config
        fields = ", ".join(f"{k}={v}" for k, v in self.cfg.items())
        return f"{self.__class__.__name__}({fields})"


