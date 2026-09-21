## DESCRIPTION
- In this second part of the Make-an-LLM-like-playing-with-legos. We'll talk about and assemble the part of a LlaMA architecture.
- I will summarize the main components in a unicode diagram.

````
  Input Text
    │
    ▼
┌─────────────────┐
│   Tokenizer     │  "hello" → [20, 43, 50, 50, 53]  (character-level)
└────────┬────────┘
         ▼
┌─────────────────┐
│  Transformer    │  × n_layer
│  Block:         │
│  ┌────────────┐ │
│  │ RMSNorm    │ |                 ______________________________
│  │ GQA attn   │ <-----------------| Rotary positional encoding | rotate the keys and query based on their position(idx)
│  │ + Residual │ │                 |____________________________|
│  ├────────────┤ │
│  │ RMSNorm    │ │
│  │ MLP (FFN)  │ │  expand 4x, SWIGLU, project back
│  │ + Residual │ │
│  └────────────┘ │
└────────┬────────┘
         ▼
┌─────────────────┐
│   RMS  Norm     │
│  Linear → logits│  vocab_size outputs + softmax (probability over next token) + s
└─────────────────┘

````

# CONFIGURATION 
- before we move ahead, we need to configure our model's constraints. This can be done via dataclass or python dictionary
```python
from dataclasses import dataclass
@dataclass
class LlaMAConfig:
    vocab_size: int = 10000 # vocab size.
    hidden: int = 512 # embedding dimension.
    context_length: int = 128 #number of words our model can attend to at a time.
    dropout_rate: int = 0.1 # drop out ratio.
    q_heads: int = 8 # number of queries heads.
    n_layers: int = 4 # number of layers of our transformer. 
    kv_heads: int = 4 # number of key and value heads.
    bias: bool = False # we remove the bias from our transformer. 
```

# Root-Mean-Squared Normalization
- RMSNorm is a simplified version of the Layer Normalization
- it is computationally efficient as it only re-scaled our data as compared to the Layer Normalization which re-centers then re-scales data.
```python
class RMSNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(config.hidden))
        self.eps = 1e-05

    def forward(self, x: torch.tensor) -> torch.tensor:
        mean_square = x.pow(2).mean(-1, keepdims = True)
        rms = torch.sqrt(mean_square + self.eps)
        return self.gamma + x / rms
#the self.eps is essential for avoiding square root of zero.
#self.gamma is a learnable parameter which the model learns during the backward pass. 
```
# FULLY CONNECTED LAYER WITH SILU (SWIGLU)
- SWIGLU is simply a gated feed forward network.
- it computes two projection of the same token represantion.
- applied a silu activation function to one and performs element-wise multiplication with the second projection.
- i.e SILU(ffn(x)) * ffn(x)
```python
class SwiGLUFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w_1 = nn.Linear(config.hidden, config.hidden, bias = config.bias)
        self.w_2 = nn.Linear(config.hidden, config.hidden, bias = config.bias)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.out_proj = nn.Linear(config.hidden, config.hidden, bias = config.bias)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = F.silu(self.w_1(x)) * self.w_2(x)
        x = self.dropout(x)
        return self.out_proj(x)
```

# ROTARY POSITIONAL ENCODING 
- Another cardinal feature of an LLM is how it understanding the order in a sequence.
- Transformer looks at all words at the same time. The model does not know which word comes first and which word comes second.
- The sentence “Man speaks to woman” and “Woman speaks to amn” have the same words, but very different meanings!
- So we need to tell the model: “This word is in position 1, this word is in position 2,” and so on. This is what positional embeddings do.
- In this tutorial I will only discuss RoPE and not relative or absolute encoding and how they are added to the token embedding. I will put a link to that below, if you want to compare.
- Instead of adding position information, *RoPE* rotates the query and key vectors by an angle that depends on their position.
- I have written a simple utility class that rotates the keys and queries right in the attention mechanism.

```python
class RotaryPositionalEncoding(nn.Module):
    def __init__(self, head_dim, base: int = 10000):
        super().__init__()
        self.head_dim = head_dim
        self.base = base

    def rotation_angles(self, x: torch.Tensor, y: torch.Tensor, theta: torch.Tensor):
        '''2D rotation matrices: this is a stantard formula from linear Algebra'''
        return (x * torch.cos(theta) - y * torch.sin(theta),
                x * torch.sin(theta) + y * torch.cos(theta))

    def forward(self, pos: int, query: torch.Tensor, key: torch.Tensor):
        '''
        freq: computes the various rotation speeds.
        theta: angle at a particular position.
        .squeeze() : to match dimensions.
        torch.stack(): to put them back together and .reshape_as() takes them back to the required dims
        '''
        freq = self.base ** (-2 * torch.arange(0, self.head_dim // 2) / self.head_dim)
        theta = torch.outer(pos, freq)
        theta = theta.to(query.device)  
        theta = theta.unsqueeze(0).unsqueeze(0)    

        q1, q2 = query[..., 0::2], query[..., 1::2]
        k1, k2 = key[..., 0::2], key[..., 1::2]

        rq1, rq2 = self.rotation_angles(q1, q2, theta)
        rk1, rk2 = self.rotation_angles(k1, k2, theta)

        rotated_query = torch.stack([rq1, rq2], dim=-1).reshape_as(query)
        rotated_key   = torch.stack([rk1, rk2], dim=-1).reshape_as(key)
        return rotated_query, rotated_key

```

# GROUPED QUERY ATTENTION MECHANISM
- Before we discuss the theory of GQA and what renders it computationally efficient and expressive. We need to discuss two other variants of attention mechanisms.
- (i) *Multi Head Attention:* In this type of attention mechanism, each query head has its own key and value. Which makes it computationally expensive but very expressive(learns diverse patterns).
- (ii) *Multi Query Attention:* This variant of attention mechanism, prioritizes computational efficiency as it only has one key - value head and multiple query heads.However, it is less expressive.
- (iii) *Grouped Query Attention:* The subject we are discussion, this variant, bring a balance between computational efficiency and model expressiveness. It organizes queries into groups, and each group shares a key - value head.
```
GQA (8 query heads, 2 groups - 1 KV set per group):

  [Q1] [Q2] [Q3] [Q4]       [Q5] [Q6] [Q7] [Q8]
    \   |    |   /             \   |    |   /
     +--+----+--+                +-+----+-+
            |                          |
            v                          v
       [K_group1]                 [K_group2]
       [V_group1]                 [V_group2]
```
- pyTorch implementation below, I fused in a rotatary positional encoding.

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_queries, num_kv_groupes, dropout):
        super().__init__()
        assert d_model % num_queries == 0, f'{d_model} must be divible by {num_queries}'
        assert num_queries % num_kv_groupes == 0, f'{num_queries} must be divisible by {num_kv_groupes}'

        self.d_model = d_model
        self.num_queries = num_queries
        self.kv_heads = num_kv_groupes
        self.head_dim = d_model // num_queries
        self.query_per_kv = num_queries // num_kv_groupes
        self.kv_dims = self.head_dim * num_kv_groupes
        self.drop_out = dropout

        self.w_q = nn.Linear(d_model, d_model, bias = False)
        self.w_k = nn.Linear(d_model, self.kv_dims, bias = False)
        self.w_v = nn.Linear(d_model, self.kv_dims, bias = False)
        self.output_proj = nn.Linear(d_model, d_model, bias = False)

        #applying RoPE in GQA
        self.rope = RotaryPositionalEncoding(self.head_dim)

    def forward(self, x: torch.tensor) -> torch.tensor:
        B, T, E = x.shape
        q = self.w_q(x).view(B, T, self.num_queries, self.head_dim)
        k = self.w_k(x).view(B, T, self.kv_heads, self.head_dim)
        v = self.w_v(x).view(B, T, self.kv_heads, self.head_dim)

        #repeat the number of keys and values to match the queries
        k = k.repeat_interleave(self.query_per_kv, dim = 2)
        v = v.repeat_interleave(self.query_per_kv, dim = 2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        #apply RoPE to the keys and queries, and get the positions with torch.arange()
        pos = torch.arange(T, device=x.device)  
        q, k = self.rope(pos=pos, query=q, key=k)
        
        context = F.scaled_dot_product_attention(q, k, v, is_causal = True, dropout_p = self.drop_out)
        context = context.transpose(1, 2).contiguous().view(B, T, E)
        return self.output_proj(context)

```

- we have reached the end of the second part.
- for further study read the articles below:
- RoPE: https://outcomeschool.com/blog/math-behind-rope-rotary-position-embedding
- grouped query attention: https://saeedmehrang.github.io/blogs/language-modeling/llm-2025-overview/grouped-query-attention/
- swiglu: https://sebastianraschka.com/faq/docs/swiglu-modern-llms.html
- rms normalization: https://docs.pytorch.org/docs/2.14/generated/torch.nn.RMSNorm.html
