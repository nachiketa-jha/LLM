import torch
import torch.nn as nn
from torch.nn import functional as F
device = 'cpu'


block_size = 8
batch_size = 4

with open("book.txt", "r", encoding="utf-8")as f:
    text = f.read()
chars = sorted(set(text))
vocab_size = len(chars)

string_to_int = {ch:i for i, ch in enumerate(chars)}
int_to_string = {i:ch for i, ch in enumerate(chars)}

encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join(int_to_string[i] for i in l) 

data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.8*len(data))
train_data = data[:n]
val_data = data[n:]
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix=torch.randint(len(data) - block_size, (batch_size,))
    x= torch.stack([data[i:i+block_size]for i in ix])
    y= torch.stack([data[i:i+block_size+1]for i in ix])
    return x.to(device), y.to(device)


x, y = get_batch('train')


x= train_data[:block_size]
y= train_data[1:block_size+1]

for t in range(block_size):
    context = x[:t+1]
    target = y[t]


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, index, targets=None):
        logits = self.token_embedding_table(index)
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, target)
        
        return loss, logits

    def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self.forward(index)
            logits = logits[:, -1, :] 
            probability = F.softmax(logits, dim = -1)
            index_next = torch.multinomial(probability, num_samples=1)
            index = torch.cat((index, index_next), dim=1)
        return index

model = BigramLanguageModel(vocab_size)
m=model.to(device)

context = torch.zeros((1,1), dtype = torch.long, device=device)
generated_chars = decode(m.generate(context, max_new_tokens=500)[0].tolist())
print(generated_chars)


