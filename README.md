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
using device: cuda

Model Architecture:
==================================================
GPT(
  (transformer): ModuleDict(
    (wte): Embedding(50257, 768)
    (wpe): Embedding(1024, 768)
    (h): ModuleList(
      (0-11): 12 x Block(
        (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (attn): CausalSelfAttention(
          (c_attn): Linear(in_features=768, out_features=2304, bias=True)
          (c_proj): Linear(in_features=768, out_features=768, bias=True)
        )
        (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (mlp): MLP(
          (c_fc): Linear(in_features=768, out_features=3072, bias=True)
          (gelu): GELU(approximate='tanh')
          (c_proj): Linear(in_features=3072, out_features=768, bias=True)
        )
      )
    )
    (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=768, out_features=50257, bias=False)
)

Parameter Count:
==================================================
Total Parameters: 124,439,808
Trainable Parameters: 124,439,808
Non-trainable Parameters: 0
==================================================
loaded 338025 tokens
1 epoch = 330 batches
Training for 200 epochs with 330 steps per epoch on cuda with best loss: inf
--------------------------------------------------
Epoch 0/200, Step 0/330, Loss: 11.0180
Epoch 0/200, Step 100/330, Loss: 6.1865
Epoch 0/200, Step 200/330, Loss: 5.5199
Epoch 0/200, Step 300/330, Loss: 5.7399
Epoch 0, avg_loss: 5.957654 | Total Time: 0:01:24 False
--------------------------------------------------
Epoch 1/200, Step 0/330, Loss: 5.7158
Epoch 1/200, Step 100/330, Loss: 4.9735
Epoch 1/200, Step 200/330, Loss: 4.8578
Epoch 1/200, Step 300/330, Loss: 5.0835
Epoch 1, avg_loss: 4.903122 | Total Time: 0:01:27 False
--------------------------------------------------
Epoch 2/200, Step 0/330, Loss: 5.2292
Epoch 2/200, Step 100/330, Loss: 4.5046
Epoch 2/200, Step 200/330, Loss: 4.4865
Epoch 2/200, Step 300/330, Loss: 4.7315
Epoch 2, avg_loss: 4.504619 | Total Time: 0:01:30 False
--------------------------------------------------
Epoch 3/200, Step 0/330, Loss: 4.8582
Epoch 3/200, Step 100/330, Loss: 4.1512
Epoch 3/200, Step 200/330, Loss: 4.1716
Epoch 3/200, Step 300/330, Loss: 4.4149
Epoch 3, avg_loss: 4.201948 | Total Time: 0:01:32 False
--------------------------------------------------
Epoch 4/200, Step 0/330, Loss: 4.5652
Epoch 4/200, Step 100/330, Loss: 3.8396
Epoch 4/200, Step 200/330, Loss: 3.8258
Epoch 4/200, Step 300/330, Loss: 4.1069
Epoch 4, avg_loss: 3.912101 | Total Time: 0:01:32 False
--------------------------------------------------
Epoch 5/200, Step 0/330, Loss: 4.2137
Epoch 5/200, Step 100/330, Loss: 3.5451
Epoch 5/200, Step 200/330, Loss: 3.5012
Epoch 5/200, Step 300/330, Loss: 3.7738
Epoch 5, avg_loss: 3.601704 | Total Time: 0:01:32 False
--------------------------------------------------
Epoch 6/200, Step 0/330, Loss: 3.8896
Epoch 6/200, Step 100/330, Loss: 3.2075
Epoch 6/200, Step 200/330, Loss: 3.1787
Epoch 6/200, Step 300/330, Loss: 3.4679
Epoch 6, avg_loss: 3.271078 | Total Time: 0:01:32 False
--------------------------------------------------
Epoch 7/200, Step 0/330, Loss: 3.6523
Epoch 7/200, Step 100/330, Loss: 2.8554
Epoch 7/200, Step 200/330, Loss: 2.7905
Epoch 7/200, Step 300/330, Loss: 3.1004
Epoch 7, avg_loss: 2.923672 | Total Time: 0:01:32 False
--------------------------------------------------
Epoch 8/200, Step 0/330, Loss: 3.4182
Epoch 8/200, Step 100/330, Loss: 2.5043
Epoch 8/200, Step 200/330, Loss: 2.4583
Epoch 8/200, Step 300/330, Loss: 2.6833
Epoch 8, avg_loss: 2.559714 | Total Time: 0:01:32 False
--------------------------------------------------
Epoch 9/200, Step 0/330, Loss: 3.0484
Epoch 9/200, Step 100/330, Loss: 2.1391
Epoch 9/200, Step 200/330, Loss: 2.0910
Epoch 9/200, Step 300/330, Loss: 2.2818
Epoch 9, avg_loss: 2.207648 | Total Time: 0:01:32 False
--------------------------------------------------
Epoch 10/200, Step 0/330, Loss: 2.6285
Epoch 10/200, Step 100/330, Loss: 1.8002
Epoch 10/200, Step 200/330, Loss: 1.7751
Epoch 10/200, Step 300/330, Loss: 1.9596
Epoch 10, avg_loss: 1.876537 | Total Time: 0:01:32 False
--------------------------------------------------
Epoch 11/200, Step 0/330, Loss: 2.2280
Epoch 11/200, Step 100/330, Loss: 1.5152
Epoch 11/200, Step 200/330, Loss: 1.4787
Epoch 11/200, Step 300/330, Loss: 1.6792
Epoch 11, avg_loss: 1.569216 | Total Time: 0:01:32 False
--------------------------------------------------
Epoch 12/200, Step 0/330, Loss: 1.9088
Epoch 12/200, Step 100/330, Loss: 1.2595
Epoch 12/200, Step 200/330, Loss: 1.2093
Epoch 12/200, Step 300/330, Loss: 1.3728
Epoch 12, avg_loss: 1.287274 | Total Time: 0:01:32 False
--------------------------------------------------
Epoch 13/200, Step 0/330, Loss: 1.7057
Epoch 13/200, Step 100/330, Loss: 0.9881
Epoch 13/200, Step 200/330, Loss: 0.9110
Epoch 13/200, Step 300/330, Loss: 1.1250
Epoch 13, avg_loss: 1.029201 | Total Time: 0:01:32 False
--------------------------------------------------
Epoch 14/200, Step 0/330, Loss: 1.3704
Epoch 14/200, Step 100/330, Loss: 0.8084
Epoch 14/200, Step 200/330, Loss: 0.6870
Epoch 14/200, Step 300/330, Loss: 0.8392
Epoch 14, avg_loss: 0.818182 | Total Time: 0:01:32 False
--------------------------------------------------
Epoch 15/200, Step 0/330, Loss: 1.1138
Epoch 15/200, Step 100/330, Loss: 0.6385
Epoch 15/200, Step 200/330, Loss: 0.5539
Epoch 15/200, Step 300/330, Loss: 0.6462
Epoch 15, avg_loss: 0.640581 | Total Time: 0:01:32 False
--------------------------------------------------
Epoch 16/200, Step 0/330, Loss: 0.8999
Epoch 16/200, Step 100/330, Loss: 0.5186
Epoch 16/200, Step 200/330, Loss: 0.4848
Epoch 16/200, Step 300/330, Loss: 0.4835
Epoch 16, avg_loss: 0.499057 | Total Time: 0:01:32 False
--------------------------------------------------
Epoch 17/200, Step 0/330, Loss: 0.7384
Epoch 17/200, Step 100/330, Loss: 0.3840
Epoch 17/200, Step 200/330, Loss: 0.3264
Epoch 17/200, Step 300/330, Loss: 0.4425
Epoch 17, avg_loss: 0.394073 | Total Time: 0:01:32 False
--------------------------------------------------
Epoch 18/200, Step 0/330, Loss: 0.5826
Epoch 18/200, Step 100/330, Loss: 0.3006
Epoch 18/200, Step 200/330, Loss: 0.2481
Epoch 18/200, Step 300/330, Loss: 0.3360
Epoch 18, avg_loss: 0.314237 | Total Time: 0:01:32 False
--------------------------------------------------
Epoch 19/200, Step 0/330, Loss: 0.4604
Epoch 19/200, Step 100/330, Loss: 0.2506
Epoch 19/200, Step 200/330, Loss: 0.2284
Epoch 19/200, Step 300/330, Loss: 0.2708
Epoch 19, avg_loss: 0.254093 | Total Time: 0:01:32 False
--------------------------------------------------
Epoch 20/200, Step 0/330, Loss: 0.4310
Epoch 20/200, Step 100/330, Loss: 0.2032
Epoch 20/200, Step 200/330, Loss: 0.1655
Epoch 20/200, Step 300/330, Loss: 0.2332
Epoch 20, avg_loss: 0.213714 | Total Time: 0:01:32 False
--------------------------------------------------
Epoch 21/200, Step 0/330, Loss: 0.3680
Epoch 21/200, Step 100/330, Loss: 0.1898
Epoch 21/200, Step 200/330, Loss: 0.1426
Epoch 21/200, Step 300/330, Loss: 0.2245
Epoch 21, avg_loss: 0.185703 | Total Time: 0:01:32 False
--------------------------------------------------
Reached target loss of 0.099999 at epoch 22

Training Summary:
==================================================
Total Training Time: 2032.6384
Final Loss: 0.164893
Best Loss Achieved: 0.080016
Final Model: checkpoints/final_model.pth.gz
Best Model: checkpoints/best_model.pth.gz
Completed at: 2025-01-21 06:51:06
==================================================
```

## Colab Link

You can run the training script on Colab using the following link: [Colab Notebook](https://colab.research.google.com/drive/1VWw6lJM14K9QxvavfbEe2jIdO99jF340#scrollTo=mc9xOCbsWQLx).

## Deployment to Hugging Face

The model is deployed on Hugging Face Spaces. You can access the deployed application using the following link: [Deployed Application](https://huggingface.co/spaces/sawandarekar/session_12_transformer_part_1).

## Deployed Application Link

[Deployed Application](https://huggingface.co/spaces/sawandarekar/session_12_transformer_part_1)
