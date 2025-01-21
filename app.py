import os
import gradio as gr
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch.nn as nn
import pickle
import gzip
from dataclasses import dataclass
from pathlib import Path

# Define the model loading function
def load_compressed_model(model, path):
    """Load compressed model"""
    with gzip.open(path, 'rb') as f:
        state_dict = pickle.load(f)
    # Convert back to float32 for training
    for key in state_dict:
        if state_dict[key].dtype == torch.float16:
            state_dict[key] = state_dict[key].float()
    model.load_state_dict(state_dict)

# Define the model configuration
@dataclass
class GPTConfig:
    n_embd: int = 768
    n_head: int = 12
    block_size: int = 1024

# Define the GPT model
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleList([CausalSelfAttention(config) for _ in range(12)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        for block in self.transformer:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)

# Define the CausalSelfAttention class
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / torch.sqrt(torch.tensor(k.size(-1), dtype=torch.float32)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = torch.nn.functional.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

# Load the model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT(GPTConfig())
load_compressed_model(model, 'checkpoints/best_model.pth')
model.eval()

def generate_text(input_text, max_length, num_samples):
    inputs = tokenizer.encode(input_text, return_tensors='pt')
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=num_samples)
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

# Create Gradio interface
iface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter text here..."),
        gr.Slider(minimum=10, maximum=100, default=50, label="Maximum Length"),
        gr.Slider(minimum=1, maximum=5, default=1, label="Number of Samples")
    ],
    outputs=gr.Textbox()
)

if __name__ == "__main__":
    iface.launch()