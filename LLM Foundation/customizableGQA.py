import torch 
from torch import nn 


def softmax(x: torch.Tensor):
  '''numerical stable softmax'''
    shifted_x = x -  x.max(dim=-1, keepdims=True)
    exp_x = torch.exp(shifted_x)
    return exp_x / torch.sum(exp_x, dim = -1, keepdims = True)

def scaled_dot_product(Q: torch.Tensor,
                       K: torch.Tensor, 
                       V: torch.Tensor,
                       mask: torch.Tensor,
                       dropout: nn.Dropout):
    
    attention_score = Q @ K.transpose(2, 3) / torch.sqrt(torch.tensor(K.shape[-1], dtype=Q.dtype, device=Q.device))
    attention_score = torch.where(condition=mask == 1, input=torch.tensor(-1e9, dtype=Q.dtype, device=Q.device), other=attention_score)
    attention_weights = softmax(dropout(attention_score))
    context_aware = attention_weights @ V
    return context_aware

def scaled_dot_product_alibi(Q: torch.Tensor, 
                            K: torch.Tensor, 
                            V: torch.Tensor, 
                            slope: float,
                            mask: torch.Tensor, 
                            dropout: nn.Dropout):
    seq_len = K.shape[1]
    pos = torch.arange(seq_len, device=K.device)
    distance = torch.abs(pos[None, None, :] - pos[:, None, None])
    
    if isinstance(slope, float):
        slope = torch.tensor([slope], device=K.device)
        
    bias = -slope[:, None, None] * distance.float()
    bias = torch.where(condition=mask == 1, input=torch.tensor(-1e9, dtype=Q.dtype, device=Q.device), other=bias)

    attention_score = Q @ K.transpose(2, 3) / torch.sqrt(torch.tensor(K.shape[-1], dtype=Q.dtype, device=Q.device))
    attention_score = attention_score + bias 
    attention_weights = softmax(dropout(attention_score))
    context_aware = attention_weights @ V
    return context_aware

class GroupedQueryAttentionCustomizable(nn.Module):
    def __init__(self,
                d_model: int,
                n_heads: int,
                seq_len: int,
                n_kv_heads: int,
                dropout: float = 0.1
                ):
        super().__init__()

        assert(d_model % n_heads == 0), "d_model must be divisible by n_heads."
        assert(n_heads % n_kv_heads == 0), "n_heads must be divisible by the n_kv_heads."

        self.d_model = d_model 
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dims = d_model // n_heads
        self.q_per_kv = n_heads // n_kv_heads
        self.dropout = nn.Dropout(dropout)
        
        self.register_buffer('casual_mask', 
                             torch.triu(torch.ones(seq_len, seq_len), diagonal=1))

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, n_kv_heads * self.head_dims)
        self.w_v = nn.Linear(d_model, n_kv_heads * self.head_dims)
        self.w_o = nn.Linear(d_model, d_model)

    def apply_rope(self, x, base=10000):
        B, H, seq_len, head_dim = x.shape
        out = torch.empty_like(x)

        pos = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        i = torch.arange(head_dim // 2, device=x.device, dtype=torch.float32)
        freq = 1.0 / (base ** (2 * i / head_dim))
        theta = torch.outer(pos, freq)
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        
        cos_t = cos_t.view(1, 1, seq_len, -1)
        sin_t = sin_t.view(1, 1, seq_len, -1)

        even, odd = x[:, :, :, 0::2], x[:, :, :, 1::2]
        
        out[..., 0::2] = even * cos_t - odd * sin_t
        out[..., 1::2] = even * sin_t + odd * cos_t
        return out

    def forward(self, x, position='rope'):
        B, T, E = x.shape 

        q = self.w_q(x).view(B, T, self.n_heads, self.head_dims).transpose(1, 2)
        k = self.w_k(x).view(B, T, self.n_kv_heads, self.head_dims).transpose(1, 2)
        v = self.w_v(x).view(B, T, self.n_kv_heads, self.head_dims).transpose(1, 2)

        k = torch.repeat_interleave(k, repeats=self.q_per_kv, dim=1)
        v = torch.repeat_interleave(v, repeats=self.q_per_kv, dim=1)
        
        if position == 'rope':
            rotated_queries = self.apply_rope(q)
            rotated_keys = self.apply_rope(k)
            context_aware = scaled_dot_product(rotated_queries, rotated_keys, v, mask=self.casual_mask, dropout=self.dropout)
        else: 
            context_aware = scaled_dot_product_alibi(q, k, v, slope=0.5, mask=self.casual_mask, dropout=self.dropout)

        return self.w_o(context_aware.transpose(1, 2).contiguous().view(B, T, E))


  
                              
