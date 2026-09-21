# Complete Mini LLaMA

This is the final assembly stage before we begin training.  
Think of it as snapping together all the Lego bricks we’ve built so far — embeddings, attention layers, feedforward blocks (like SwiGLU), and normalization — into one cohesive engine.  

At this point:
- **Embeddings** convert tokens into dense vectors.  
- **Transformer Blocks** (attention + feedforward) process those vectors, layer by layer.  
- **Stacking multiple blocks** gives the model depth and the ability to capture complex language patterns.  
- **Output layers** map the processed representations back into predictions (like the next token).  

By placing each component in the right spot, we create a miniature version of LLaMA’s architecture. It’s simple compared to full‑scale LLMs, but it captures the essence of how they work. Once assembled, this model is ready to be trained on text data — turning our Lego‑like construction into a functioning language model.


```python
class LlaMASimple(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_emb = nn.Embedding(config.vocab_size, config.hidden)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.trans_block = nn.ModuleList(
            [LlamaTransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.rms_norm = RMSNorm(config)
        self.output_head = nn.Linear(config.hidden, config.vocab_size, bias = False)

    def forward(self, x: torch.tensor) -> torch.tensor:
        token_emb = self.token_emb(x)
        x = self.dropout(token_emb)
        for layer in self.trans_block:
            x = layer(x)
        x = self.rms_norm(x)
        logits = self.output_head(x)
        return logits
```
