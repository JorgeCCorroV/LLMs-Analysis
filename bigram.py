# conda install PyPDF2
# conda install torch

import PyPDF2
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 500000
eval_interval = 50000
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 10000
# ------------

torch.manual_seed(123)

# wget https://canonburyprimaryschool.co.uk/wp-content/uploads/2016/01/Joanne-K.-Rowling-Harry-Potter-Book-1-Harry-Potter-and-the-Philosophers-Stone-EnglishOnlineClub.com_.pdf
with open('Joanne-K.-Rowling-Harry-Potter-Book-1-Harry-Potter-and-the-Philosophers-Stone-EnglishOnlineClub.com_.pdf', 'rb') as f:
  reader = PyPDF2.PdfReader(f)

  full_text = ""
  for page in reader.pages:
    full_text += page.extract_text()

# After examine the book, We keep just the text, filtering out the facebook, prologue, greetings, and others.
full_text = full_text[3315:-1408]

# here are all the unique characters that occur in this text
chars = sorted(list(set(full_text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(full_text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

"""

step 0: train loss 4.9105, val loss 4.9060
step 50000: train loss 2.4464, val loss 2.4452
step 100000: train loss 2.4466, val loss 2.4454
step 150000: train loss 2.4466, val loss 2.4455
step 200000: train loss 2.4464, val loss 2.4452
step 250000: train loss 2.4467, val loss 2.4455
step 300000: train loss 2.4466, val loss 2.4450
step 350000: train loss 2.4465, val loss 2.4459
step 400000: train loss 2.4462, val loss 2.4445
step 450000: train loss 2.4466, val loss 2.4462

 twide heve n watioony vif nd honk Hag d w  ind. ownt Tho Ithe wanon tw ‘D rs we thnn.’ brtey tncrrerengrel aniparergak-e I’ w therattsugeelouen nd asad 
 wonorr merrmosu, terst fintr orichome he OWorom,’ tapothelelay’s h, cl-p!’ as ns the, ‘Wesheyofomil Hadast or futopeanupe fombeghed tenk Nororyo is w bem 
 Hat hofl POn.. bary t t am; w housil d stce s w sioad, ‘Hararest hath,’ He T as Buldsin ,’ mis BENEROhend londinanking whe s gry penthet. asoo ager t ER bey 
 termeasatheroust Wo lat he an ‘Loyou

"""
