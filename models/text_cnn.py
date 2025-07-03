import torch
import torch.nn as nn
import torch.nn.functional as F

class ParallelTextCNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=300,
        num_classes=10,
        kernel_sizes=[2, 3, 4, 5],
        num_filters=512,
        dropout=0.5,
        pad_idx=0,
        pretrained_embeddings=None,
        freeze_embeddings=False,
        multi_label=False  # Use sigmoid for multi-label, softmax for multi-class
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = not freeze_embeddings

        # Four parallel convolutional layers with different kernel sizes (filter widths)
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=1,
                out_channels=num_filters,
                kernel_size=(k, embedding_dim)
            )
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(num_filters * len(kernel_sizes), 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.multi_label = multi_label

    def forward(self, input_ids, attention_mask=None, return_embedding=False):
        # input_ids: [batch, seq_len]
        x = self.embedding(input_ids)           # [batch, seq_len, emb_dim]
        x = x.unsqueeze(1)                      # [batch, 1, seq_len, emb_dim]
        pooled_outputs = []
        for conv in self.convs:
            c = F.relu(conv(x)).squeeze(3)      # [batch, num_filters, seq_len - k + 1]
            p = F.max_pool1d(c, c.size(2)).squeeze(2)  # [batch, num_filters]
            pooled_outputs.append(p)
        x = torch.cat(pooled_outputs, 1)        # [batch, num_filters * num_kernels]
        x = self.dropout(x)
        emb_out = F.relu(self.fc1(x))           # [batch, 256]
        emb_out = self.dropout(emb_out)
        logits = self.fc2(emb_out)              # [batch, num_classes]
        if self.multi_label:
            logits = torch.sigmoid(logits)
        else:
            logits = torch.softmax(logits, dim=1)
        if return_embedding:
            return logits, emb_out
        return logits
