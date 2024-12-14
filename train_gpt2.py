from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

# -----------------------------------------------------------------------------
# GPT2
# -----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a bias, more of an attention mask, just following OpenAI/HF's naming convention
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                                .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is number of heads, hs is head size, and C is embedding dimensionality (n_embd = nh * hs)
        # in GPT2-124M, nh = 12, hs = 64, C = 768
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # attention (materializes the large (T, T) attention matrix for all the queries and keys)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 500,000 BPE merges + 256 byte tokens + 1 |endoftext| token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of attention heads
    n_embd: int = 768 # embedding dimension


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                wpe = nn.Embedding(config.block_size, config.n_embd),
                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f = nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    
    def forward(self, idx, targets=None):
        # idx is in the shape of (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # foward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos).unsqueeze(0) # position embeddings of shape (1, T, n_embd)
        x = tok_emb + pos_emb # (B, T, n_embd); there is hidden broadcasting because the shape of pos_emb is (1, T, n_embd)

        # forward the transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # forward the final layer norm
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None: 
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits.view(B*T, logits.size(-1)), targets)
        return logits, loss
    
    @classmethod
    def from_pretrained(cls, model_type: str):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel
        model = GPT2LMHeadModel.from_pretrained(model_type)
        print("loading weights from pretrained %s" % model_type)

        # n_layer, n_head, n_embd are determined from the model type
        config_args = {
            'gpt2':        dict(n_layer=12, n_head=12, n_embd=768), # 124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':  dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':     dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        
        # the sd works for both our model and the huggingface model
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [key for key in sd_keys if not key.endswith('.attn.bias')] # discard this mask / buffer, they are not parameters

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_hf_keys = sd_hf.keys()
        sd_hf_keys = [key for key in sd_hf_keys if not key.endswith('.attn.masked_bias')]
        sd_hf_keys = [key for key in sd_hf_keys if not key.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']


        # the openai tensorflow checkpoints use a 'Conv1D' module, but we only need the vanilla Linear module
        # this means we have to transpose the weights from Conv1D to Linear
        assert(len(sd_hf_keys) == len(sd_keys)), f"mismatched keys: {len(sd_hf_keys)} != {len(sd_keys)}"
        for key in sd_hf_keys:
            if any(key.endswith(w) for w in transposed):
                # special case for the Conv1D to Linear transposition
                assert sd[key].shape == sd_hf[key].t().shape
                with torch.no_grad():
                    sd[key].copy_(sd_hf[key].t())
            else:
                # vanilla copy over the other parameters
                assert sd[key].shape == sd_hf[key].shape
                with torch.no_grad():
                    sd[key].copy_(sd_hf[key])
        
        return model

# -----------------------------------------------------------------------------
device = 'cuda'
num_return_sequences = 3
max_length = 30

import tiktoken
enc = tiktoken.get_encoding("gpt2")
with open("dataset/input.txt", "r") as f:
    text = f.read()
text = text[:1000]
tokens = enc.encode(text)

B, T = 4, 32
buf = torch.tensor(tokens[:B*T + 1], dtype=torch.long)
buf = buf.to(device)
x = buf[:-1].view(B, T)
y = buf[1:].view(B, T)

# get the logits
model = GPT.from_pretrained('gpt2')
model.to(device)
# logits, loss = model(x, y)
# assert logits.size() == (B, T, 50257), logits.size()
# print(loss)
 
# optimize for 50 steps
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f"step {i}: loss: {loss.item():.8f}")


import sys; sys.exit(0)

while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x) # (B, T, vocab_size) = (3, 8, 50257)
        # take the logits from the last position
        logits = logits[:, -1, :] # (3, 50257)
        
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1) # (3, 50257)
        
        # do top-k sampling of 50 (huggingface default)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)  # Use probs instead of logits
        
        # select the next token from the topk
        ix = torch.multinomial(topk_probs, num_samples=1) # (3, 1)
        # gather the selected indices
        xcol = torch.gather(topk_indices, -1, ix) # (3, 1)
        # append to the sequence and continue
        x = torch.cat((x, xcol), dim=1) # (3, 9)

# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print('>', decoded)