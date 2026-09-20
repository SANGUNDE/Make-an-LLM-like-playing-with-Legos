## LESSON 1: TOKENISATION
---
The first step in modern NLP (Natural Language Processing) is tokenization. 
By definition Tokenization is the task of separating out or tokenizing words from running text. 
*Tokens* can be words, subwords, or even characters depending on the approach.

---

## Why Tokenisation Matters
- Models cannot work directly with raw text.
- Tokenisation converts text into numerical representations.
- Different strategies (word-level, subword-level, character-level) affect vocabulary size and model performance.

---

## Example 1: Simple Word Tokenisation
```python
# Split text into words using Python's split()
text = "Transformers are powerful models"
tokens = text.split()
print(tokens)
```
## Example 2: Extended WordLevel Tokenizer.

```python
import re
class WordLevelTokenizer:
    def __init__(self):
        self.pattern = re.compile(r'[\w]+')
        self.map = {}

    def train(self, text: str):
        tokens = self.pattern.findall(text)
        for idx, token in enumerate(tokens):
            if token not in self.map:
                self.map[token] = idx 

    def encode(self, text: str) -> list[int]:
        ids = []
        splitted_text = text.split()
        for word in splitted_text:
            if word in self.map:
                ids.append(self.map[word])
            else:
                ids.append(-1)
        return ids

    def decode(self, ids: list[int]) -> str:
        tokens = []
        for idx in ids:
            if idx in self.map.values():
                token = list(self.map.keys())[list(self.map.values()).index(idx)]
                tokens.append(token)
            else:
                tokens.append('[unk]')
        return ' '.join(tokens)

    def show_ids(self):
        return self.map

#example usage
#train it on a small corpora
runing_text = 'This is a tokenizer'
tokenizer = WordLevelTokenizer()
tokenizer.train(runing_text)

#encoding
sample_text = 'this is a that'
tokenizer.encode(sample_text)

#decoding
tokenizer.decode([-1, 1, 2, -1])
```


## DISADVANTAGE OF USING WORD - LEVEL ENCODING
- UKNOWN WORDS: words that our model didn't see during training.
- Hard to define. reference: Speech and Language processing by Daniel Jurafsky and James H. Martin


## ALTERNATIVE - BPE ALGORITHM
- Use algorithms that use subword tokenization.
- e.g
- tiktoken (OpenAI)
- SentencePiece (Google) uses BPE and Unigram
- WordPiece (Google)

## Example usage

```python
import sentencepiece as spm

tokenizer = spm.SentencePieceProcessor(model_file = 'llaMA_tokenizer.model')
text = 'Try training the sentence pieces and follow this example for encoding and decoding'
print(f'ENCODING\n')
pieces = tokenizer.encode(text, out_type = str)
ids = tokenizer.encode(text, out_type = int)
print(f'Pieces: {pieces}')
print(f'ids: {ids}')
print('DECODING\n')
token_ids: int = [384, 9949, 5134, 3422, 442, 267, 593, 403, 9948, 48, 290, 6083, 300, 2341, 359, 335, 551, 50, 9158, 261, 279, 1185, 862, 395]
print(tokenizer.decode(token_ids))
print(len(ids))

```
- reference: check the sentencepiece github repo.

thank you for reading. leave a like
