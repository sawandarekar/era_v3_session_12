# Training Transformers from Scratch

## Model Architecture

The model is based on the GPT architecture. It consists of the following components:
- Embedding layers for tokens and positions
- Multiple transformer blocks, each containing:
  - Layer normalization
  - Causal self-attention
  - Feed-forward neural network
- Final layer normalization
- Linear layer for language modeling

## Model Details

The model is implemented in the `GPT` class in [train.py](train.py) and [train_transformers.py](train_transformers.py). It uses the following configuration:
- Vocabulary size: 50257
- Block size: 1024
- Number of layers: 12
- Number of heads: 12
- Embedding dimension: 768

## Model Parameters

The model has the following parameters:
- Total Parameters: 124M
- Trainable Parameters: 124M
- Non-trainable Parameters: 0

## Features

- Token and position embeddings
- Causal self-attention mechanism
- Layer normalization
- Feed-forward neural network
- Weight sharing between embedding and output layers
- Gradient clipping
- Learning rate scheduling with cosine annealing

## Model Training Logs from Colab

```
```

## Colab Link

You can run the training script on Colab using the following link: [Colab Notebook](https://colab.research.google.com/drive/1VWw6lJM14K9QxvavfbEe2jIdO99jF340#scrollTo=mc9xOCbsWQLx).

## Deployment to Hugging Face

The model is deployed on Hugging Face Spaces. You can access the deployed application using the following link: [Deployed Application](https://huggingface.co/spaces/sawandarekar/session_12_transformer_part_1).

## Deployed Application Link

[Deployed Application](https://huggingface.co/spaces/sawandarekar/session_12_transformer_part_1)
