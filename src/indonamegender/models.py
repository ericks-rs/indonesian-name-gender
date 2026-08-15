import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, rnn_out, mask):
        scores = self.attn(rnn_out).squeeze(-1).masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=1)
        pooled = (rnn_out * weights.unsqueeze(-1)).sum(dim=1)
        return pooled, weights

class BiRNNAttn(nn.Module):
    RNN_CLASSES = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}

    def __init__(self, vocab_size, emb_dim, hidden_dim, dropout, rnn_type="lstm"):
        super().__init__()
        self.rnn_type = rnn_type
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = self.RNN_CLASSES[rnn_type](
            emb_dim, hidden_dim // 2, batch_first=True, bidirectional=True
        )
        self.attention = Attention(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, return_attention=False):
        mask = (x != 0).float()
        emb = self.embedding(x)
        out, _ = self.rnn(emb)
        pooled, weights = self.attention(out, mask)
        logit = self.fc(self.dropout(pooled)).squeeze(1)
        if return_attention:
            return logit, weights
        return logit

class TransformerClf(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, ff_dim, max_len, dropout):
        super().__init__()
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.attention = Attention(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x, return_attention=False):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        pad_mask = (x == 0)
        out = self.encoder(self.tok_emb(x) + self.pos_emb(pos),
                           src_key_padding_mask=pad_mask)
        pooled, weights = self.attention(self.norm(out), (~pad_mask).float())
        logit = self.fc(self.dropout(pooled)).squeeze(1)
        if return_attention:
            return logit, weights
        return logit
