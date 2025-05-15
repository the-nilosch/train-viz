import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, emb_dim=128):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.projection = nn.Conv2d(3, emb_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, emb_dim))

    def forward(self, x):
        B = x.shape[0]
        x = self.projection(x).flatten(2).transpose(1, 2)  # [B, num_patches, emb_dim]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, emb_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, num_patches+1, emb_dim]
        x = x + self.pos_embedding
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, emb_dim=128, num_heads=4, depth=6, mlp_dim=256):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=num_heads, dim_feedforward=mlp_dim, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        return self.transformer(x)

class ViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, num_classes=10, emb_dim=128, depth=6, num_heads=4, mlp_dim=256):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, emb_dim)
        self.transformer_encoder = TransformerEncoder(emb_dim, num_heads, depth, mlp_dim)
        self.head = nn.Linear(emb_dim, num_classes)

    def forward(self, x, return_embedding=False):
        x = self.patch_embed(x)
        x = self.transformer_encoder(x)
        cls_token_embedding = x[:, 0]  # Use class token embedding
        out = self.head(cls_token_embedding)
        if return_embedding:
            return out, cls_token_embedding
        return out
