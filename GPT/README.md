# GPT-1: Improving Language Understanding by Generative Pre-Training — From Scratch

PyTorch implementation of GPT-1 from *Improving Language Understanding by Generative Pre-Training* (Radford et al., OpenAI 2018), with layer transfer and zero-shot experiments mirroring Figure 2 of the paper.

## What this project does

Builds GPT-1 from scratch in PyTorch including:

- `GPTConfig` - all hyperparameters in one place
- Decoder-only Transformer with causal self-attention
- BPE tokenizer following the same approach (`<start>`, `<extract>`, `<delim>` tokens)
- Unsupervised pre-training via next word prediction
- Supervised fine-tuning with task-aware input transformations
- Auxiliary language modeling loss during fine-tuning
- Layer transfer experiment mirroring Figure left
- Zero-shot evaluation mirroring right

## Scaled-down config

| | Paper (GPT-1) | This project |
|---|---|---|
| Layers | 12 | 4 |
| Hidden size | 768 | 256 |
| Attention heads | 12 | 4 |
| Feed-forward dim | 3072 | 512 |
| Parameters | 117M | 9.8M |
| Dataset | BooksCorpus (800M words) | WikiText-2 (~2M tokens) |
| Tokenizer | BPE (40K merges on BooksCorpus) | BPE (30K merges on WikiText-2) |

## Results

### Zero-shot evaluation (Figure 2 right from the paper)

| Checkpoint | Val Loss | SST-2 Zero-shot Acc |
|---|---|---|
| Epoch 2 | 9.114 | 49.5% |
| Epoch 4 | 7.593 | 49.5% |
| Epoch 6 | 7.295 | 49.5% |
| Epoch 8 | 6.960 | 49.5% |
| Epoch 10 | 6.790 | 49.5% |

### Layer transfer experiment (Figure 2 left from the paper)

| Layers transferred | SST-2 Val Acc |
|---|---|
| 0 (embeddings only) | 77.6% |
| 1 layer | **78.4%** |
| 2 layers | 76.9% |
| 3 layers | 77.3% |
| 4 layers | 76.8% |
| Full fine-tuning (all layers) | 78.3% |


**Zero-shot capability does not emerge at small scale***
Zero-shot accuracy was stuck at 49/.5% at all checkpoints regardless of how long pre-training continued. The paper showed only marginal zero-shot improvement at 117M parameters on 800M words. But at 9.8M  parameters on 2M tokens the capability simply does not emerge. 

**Early layers capture the most transferable representations**
Transferring just 1 layer gave the best results (78.4%). Adding more layers actually hurt. The first layer learned general language patterns that transfer well, but deeper layers became too focused on the pre-training task to help with sentiment.

**Auxiliary language modeling loss matters**
Adding the LM objective during fine-tuning improved training stability and generalization, consistent with the paper's finding that it helps on most tasks.

**Scale is the dominant factor**
The 400x difference in training data (800M vs 2M words) and 12x difference in parameters (117M vs 9.8M) explains most of the gap between this implementatation results and the paper's.




