# Solving for residual std scaling issue
import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.amp import autocast, GradScaler  # Add GradScaler back
from pathlib import Path
import io
import zipfile
import pickle
import gzip
from datetime import datetime, timedelta
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm, trange


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

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
    # block_size: int = 1024 # max sequence length
    # vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    # n_layer: int = 12 # number of layers
    # n_head: int = 12 # number of heads
    # n_embd: int = 768 # embedding dimension
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12  # Reduced number of layers
    n_head: int = 12   # Reduced number of heads
    n_embd: int = 768 # Reduced embedding dimension


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing
        self.transformer.wte.weight = self.lm_head.weight

        # weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean = 0.0, std = std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std = 0.02)



    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


def save_compressed_model(model, path):
    """Save model with compression"""
    state_dict = model.state_dict()
    # Convert tensors to half precision
    for key in state_dict:
        if state_dict[key].dtype == torch.float32:
            state_dict[key] = state_dict[key].half()
    # Compress and save
    with gzip.open(path, 'wb', compresslevel=9) as f:
        pickle.dump(state_dict, f)

def load_compressed_model(model, path):
    """Load compressed model"""
    with gzip.open(path, 'rb') as f:
        state_dict = pickle.load(f)
    # Convert back to float32 for training
    for key in state_dict:
        if state_dict[key].dtype == torch.float16:
            state_dict[key] = state_dict[key].float()
    model.load_state_dict(state_dict)
# model = GPT.from_pretrained('gpt2')

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

# SEED
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# STOP
num_return_sequences = 5
max_length = 30



import tiktoken

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # at init load tokens from disk and store them in memory
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2') 
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f'loaded {len(self.tokens)} tokens')
        print(f'1 epoch = {len(self.tokens) // (B * T)} batches')

        # state
        self.current_position = 0
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B*T
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y


model = GPT(GPTConfig())
model.to(device)


# Print model architecture and parameter count
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

max_lr = 6e-4 
min_lr = max_lr * 0.1
warmup_steps = 10
target_loss = 0.099999
best_loss = float('inf')

# Update hyperparameters
initial_lr = 1e-3
min_lr = 1e-4
batch_size = 16  # Reduced batch size for better gradient updates
context_length = 32  # Reduced context length
accumulation_steps = 2  # Reduced accumulation steps

# Initialize data loader with new batch size
train_loader = DataLoaderLite(B=batch_size, T=context_length)
num_epochs = 10  # Define number of epochs
steps_per_epoch = len(train_loader.tokens) // (train_loader.B * train_loader.T)

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > num_epochs:
        return min_lr
    decay_ratio = (it - warmup_steps) / (num_epochs - warmup_steps)
    assert 0 <= decay_ratio <=1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)




# Create optimizer with modified parameters
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=initial_lr,
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=0.1
)

# First, modify the scaler initialization based on device
if device == 'cuda':
    scaler = GradScaler()
    use_scaler = True
    ctx_manager = autocast(device_type='cuda')
else:
    scaler = None
    use_scaler = False
    ctx_manager = autocast(device_type='cpu')

# Better learning rate scheduler
scheduler = OneCycleLR(
    optimizer,
    max_lr=initial_lr,
    total_steps=num_epochs * steps_per_epoch,
    pct_start=0.1,  # Warm up for 10% of training
    div_factor=10,  # min_lr = initial_lr/10
    final_div_factor=10,  # final_lr = min_lr/10
    anneal_strategy='cos'
)

# Add before training loop
checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)
best_model_path =  os.path.join(checkpoint_dir, "best_model.pth")
final_model_path = os.path.join(checkpoint_dir, "final_model.pth")


# Load previous best model if exists
if os.path.exists(best_model_path):
    print(f"Loading previous best model from {best_model_path}")
    model = GPT(GPTConfig())
    load_compressed_model(model, best_model_path)
    model.to(device)
else:
    model = GPT(GPTConfig())
    model.to(device)


total_params, trainable_params = count_parameters(model)

print("\nModel Architecture:")
print("=" * 50)
print(model)
print("\nParameter Count:")
print("=" * 50)
print(f"Total Parameters: {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}")
print(f"Non-trainable Parameters: {total_params - trainable_params:,}")
print("=" * 50)

# Add before training loop
rmat_time(seconds):
    """Convert seconds to human readable string"""
    return str(timedelta(seconds=int(seconds)))def fo

# Modified training loop with timing
print(f"\nStarting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
training_start = time.time()

# Create epoch progress bar
for epoch in range(num_epochs):
    epoch_start = time.time()
    model.train()
    epoch_loss = 0
    optimizer.zero_grad(set_to_none=True)
    accumulated_loss = 0
    
    for step in range(steps_per_epoch):
        step_start = time.time()
        t0 = time.time()  # Add this line to define t0
        
        x, y = train_loader.next_batch()
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass
        with ctx_manager:
            logits, loss = model(x, y)
        
        # Backward pass and optimization
        if use_scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        # Learning rate adjustment
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Metrics
        current_loss = loss.item()
        best_loss = min(best_loss, current_loss)
        t1 = time.time()
        dt = (t1 - t0) * 1000
        tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - step_start)
        
        # Logging
        print(f'step {step:4d} | loss: {current_loss:.4f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f} | norm: {norm:.2f} | lr: {lr:.2e}')
        
    # Calculate average loss and epoch timing
    epoch_time = time.time() - epoch_start
    avg_loss = epoch_loss / (steps_per_epoch / accumulation_steps)
    
    print(f'\nEpoch {epoch+1} | Average Loss: {avg_loss:.4f} | Best Loss: {best_loss:.4f}| Epoch Time: {format_time(epoch_time)} | Total Time: {format_time(time.time() - training_start)}')
    # print("=" * 50)
    
    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        print(f"New best loss: {best_loss:.4f}")
        save_compressed_model(model, best_model_path)
        print(f"Best model saved at epoch {epoch+1}")
        
    # Early stopping check
    if best_loss < target_loss:
        print(f"Reached target loss of {target_loss} at epoch {epoch+1}")
        break

# Training summary
total_time = time.time() - training_start
print("\nTraining Summary:")
print("=" * 50)
print(f"Total Training Time: {format_time(total_time)}")
print(f"Final Loss: {loss.item():.6f}")
print(f"Best Loss Achieved: {best_loss:.6f}")
print(f"Final Model: {final_model_path}.gz")
print(f"Best Model: {best_model_path}.gz")
print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 50)
