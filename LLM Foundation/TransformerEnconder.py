import torch 
from torch import nn
from torch.nn import functional as F


class EncoderLayer(nn.Module):
    def __init__(self, 
                d_model,
                n_heads,
                dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.layernorm = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), 
                                nn.GELU(),
                                nn.Linear(4 * d_model, d_model))
        self.trans_block = nn.MultiheadAttention(d_model, n_heads, dropout)


    def forward(self, x):
        resid = x
        norm = self.layernorm(x)
        logits, _ = self.trans_block(norm, norm, norm)
        hidden = logits + resid

        norm = self.layernorm2(hidden)
        output = self.ffn(norm)
        output = output + hidden
        return output


class TransformerEncoder(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, dropout):
        super().__init__()

        self.final_norm = nn.LayerNorm(d_model)
        self.encoderLayer = nn.ModuleList([EncoderLayer(d_model, n_heads, dropout) for _ in range(n_layers)])


    def forward(self, x):
        for layer in self.encoderLayer:
            x = layer(x)

        logits = self.final_norm(x)
        return logits
