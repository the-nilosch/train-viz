import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTMAttentionClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_layers,
        num_classes,
        dropout=0.3,
        pad_idx=None,
        use_attention=True
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.use_attention = use_attention
        if use_attention:
            self.attn_linear = nn.Linear(hidden_dim * 2, hidden_dim * 2)
            self.context_vector = nn.Linear(hidden_dim * 2, 1, bias=False)
        self.output = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask=None, return_embedding=False):
        emb = self.embedding(input_ids)  # [batch, seq, emb_dim]
        emb = self.dropout(emb)
        outputs, _ = self.bilstm(emb)    # [batch, seq, 2*hidden]

        if self.use_attention:
            # Attention scoring
            u = torch.tanh(self.attn_linear(outputs))  # [batch, seq, 2*hidden]
            attn_scores = self.context_vector(u).squeeze(-1)  # [batch, seq]
            if attention_mask is not None:
                attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e9)
            attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)  # [batch, seq, 1]
            emb_out = torch.sum(outputs * attn_weights, dim=1)  # [batch, 2*hidden]
        else:
            # Use mean pooling instead of attention
            if attention_mask is not None:
                # Mask out padded positions
                lengths = attention_mask.sum(dim=1, keepdim=True)  # [batch, 1]
                emb_out = torch.sum(outputs * attention_mask.unsqueeze(-1), dim=1) / lengths
            else:
                emb_out = outputs.mean(dim=1)  # [batch, 2*hidden]

        emb_out = self.dropout(emb_out)
        logits = self.output(emb_out)
        if return_embedding:
            return logits, emb_out
        return logits
