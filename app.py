import os
import gradio as gr
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from train import load_compressed_model, GPTConfig, GPT

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

# Load the model and tokenizer
checkpoint_dir = Path("checkpoints")
best_model_path =  os.path.join(checkpoint_dir, "best_model.pth")
model = GPT(GPTConfig())
load_compressed_model(model, best_model_path)
model.to(device)
model.train(False)

def generate_text(input_text, max_length, num_samples):
    inputs = tokenizer.encode(input_text, return_tensors='pt')
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=num_samples)
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

# Create Gradio interface
iface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.inputs.Textbox(lines=2, placeholder="Enter text here..."),
        gr.inputs.Slider(minimum=10, maximum=100, default=50, label="Maximum Length"),
        gr.inputs.Slider(minimum=1, maximum=5, default=1, label="Number of Samples")
    ],
    outputs=gr.outputs.Textbox()
)

if __name__ == "__main__":
    iface.launch()