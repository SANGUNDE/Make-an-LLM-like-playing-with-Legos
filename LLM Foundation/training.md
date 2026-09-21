# DECODING STRATEGIES
- Decoding strategies are text generation strategies
- Here we are going to talk about there; Greedy decoding, top-k and Temperature
- *Greed decoding:* which selects the token with the highest probabilities.
- *Top-K:* We truncate our selection to K tokens only, works well with temperature, as it provides a probabilistic selection to the K tokens with high probability. 
- *Temperature:* reshape the probability distribution to increase the probability of the high probability tokens and decrease the probability of the low probability tokens.
- 
```python
import torch
from torch import nn

def text_generation(model,
                    idx,
                    max_new_token, 
                    context_size,
                    temperature = 0.0,
                    eos_id = None,
                    k = 30):
  
    
    for _ in range(max_new_token):
        idx_cond = idx[:, -context_size:] #focus only on the context constraint. 
        with torch.inference_mode():
            logits = model(idx_cond)
        logits = logits[:, -1, :] #select the last predicted token. 

        if k is not None:
            top_logits, _ = torch.topk(logits, k) # select the K tokens and zero out low probability tokens. 
            logits = torch.where(
                condition = logits < top_logits[:, -1],
                input = torch.tensor(-torch.inf),
                other = logits
            )

        if temperature > 0.0: 
            logits = logits / temperature
            probs = torch.softmax(logits, -1)
            id_next = torch.multinomial(probs, num_samples = 1)
        else:
            id_next = torch.argmax(logits, -1, keepdim = True) #resort to greedy decoding. 
        if id_next == eos_id:
            break
        idx = torch.concat([idx, id_next], dim = -1)
    return idx

```
# UTILITY FUNCTIONS FOR LOSS CALCULATION AND TOKEN - TEXT CONVERSATION

```python
def calc_loss_batch(input_batch, output_batch, model):
    '''
    remember: predicted logits shape -> [batch, seq_len, vocab_size] e.g [2, 4, 10000]
              target tokens shape -> [batch, seq_len] e.g [2, 4]
              flatten them in the batch. 
    '''
    logits = model(input_batch)
    loss = F.cross_entropy(logits.flatten(0, 1), output_batch.flatten())
    return loss 


def text_to_ids(text, tokenizer):
    ''' utility function to change text to ids'''
    encoded = tokenizer.encode(text) 
    encoded = torch.tensor(encoded).unsqueeze(0)
    return encoded

def ids_to_text(ids, tokenizer):
    ''' utility function to change ids to text'''
    ids = ids.squeeze(0).detach().tolist()
    return tokenizer.decode(ids)


def generate_text(model, tokenizer, start_context):
    model.eval()
    context_size = LlaMAConfig.context_length
    encoded = text_to_ids(start_context, tokenizer)
    with torch.inference_mode():
        token_ids = text_generation(model, encoded, max_new_token=20, context_size = context_size, temperature = 0.5, k = 30)

    decode = ids_to_text(token_ids, tokenizer)
    decoded_text = decode.replace('\n', ' ')
    print(decoded_text)
    model.train()

```

# THE TRAINING LOOP
LR: learning rate which controls the number of steps we take when minimizing the loss. 
epochs: number of time run.
optimizer: tuning the weights. 
scheduler: controls the learning rate.  

```python
lr = 0.0004
epochs = 50
optimizer = AdamW(llama_model.parameters(), lr)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max = 10)
```



```python
def training_loop(
        model,
        train_loader,
        num_epochs,
        lr_scheduler,
        optimizer
):
    pbar = tqdm(range(num_epochs), desc = 'Training') 

    for _ in pbar:
        model.train()
        training_loss = 0.0

        for word_batch, target_batch in train_loader:
            optimizer.zero_grad(set_to_none = True)
            loss = calc_loss_batch(word_batch, target_batch, model)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
            optimizer.step()
            training_loss += loss.item()

        lr_scheduler.step()

        pbar.set_postfix(loss=f'{training_loss / len(train_loader)}')
        generate_text(model, tokenizer, 'Maybe I should')

training_loop(llama_model, train_loader, epochs, scheduler, optimizer)

```

- After 50 passes through the dataset, I only use a training data for lack of data, low RAM and educational purposes.
- Given a context *is it*, the trained LlaMA was able to generate the following text

```python
gen_ids = text_generation(llama_model, 
                idx = text_to_ids('is it', tokenizer),
                max_new_token = 50, 
                context_size = LlaMAConfig.context_length,
                temperature =0.8,
                k = 20)
gen_text = ids_to_text(gen_ids, tokenizer)
print(gen_text)

#is it happened after" For Jack himself," I had alwaysed up at the picture above the chimney--piece. "I like to crumental easel placed them of their savour--room. It
```

- You can include a validation step as well.
- I trained the model on *the verdict.txt and shakespear* I have provided it as well. 
- and collect the statistics as well.
- I thought include all that would make the tutorial difficult to follow. So, I just wrote a beginner friendly training loop.
