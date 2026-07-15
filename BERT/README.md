# BERT: Pre-training of Deep Bidirectional Transformers — From Scratch

PyTorch implementation of BERT from *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* (Devlin et al., 2018), with a full ablation study like Table 5 of the paper.

## What this project does

Builds BERT from scratch in PyTorch including:

- `BertConfig` — like the original Google repo design
- Full BERT architecture with token, segment, and learned positional embeddings
- Masked Language Modeling (MLM) pre-training task
- Next Sentence Prediction (NSP) pre-training task
- Fine-tuning on SST-2 sentiment classification
- Full ablation study like Table 5 of the paper

I structured this notebook like the original Google repo - shared componenets defined once and 4 experiments run controlled by flags (`do_nsp`, `ltr`, `use_bilstm`).

## Scaled-down config used in this notebook:

| | Paper (BERT Base) | This project |
|---|---|---|
| Layers | 12 | 6 |
| Hidden size | 768 | 512 |
| Attention heads | 12 | 8 |
| Feed-forward dim | 3072 | 1024 |
| Vocab size | 30,522 | ~30K |
| Pre-training data | Wikipedia + BooksCorpus | WikiText-103 |

## Ablation Study — Like Table 5 of the Paper

Four configurations varying one component at a time, evaluated on SST-2 sentiment classification.

| Run | NSP | Bidirectional | Additions | SST-2 Val Acc |
|---|---|---|---|---|
| Run 1 — Full BERT | Yes | Yes | — | 72.6% |
| Run 2 — No NSP | No | Yes | — | 66.4% |
| Run 3 — LTR & No NSP | No | No | — | 50.9% |
| Run 4 — LTR & No NSP + BiLSTM | No | No | BiLSTM | 50.9% |

## Observations:

**Bidirectionality is the most critical component** - Removing Bidirectionality dropped accuracy from 66.4% to 50.9%. The model stopped learning altogether. This directly validates BERT's core argument against GPT's left-to-right approach.

**NSP adds meaningful improvement** - Removing NSP dropped accuracy from 72.6% to 66.4%, a 6.2 point drop even on a single sentence task. 

**BiLSTM cannot recover lost bidirectionality** — Run 4 scored identical to Run 3 at 50.9%. Adding a BiLSTM on top of a left-to-right pre-trained model does not fix representations that were never bidirectional to begin with. This directly proves Table 5 row 4 of the paper.

**Pre-training quality is the bottleneck** — Fine-tuning on a poorly pre-trained model produces weak results regardless of the fine-tuning architecture.

## How to run

Open `BERT_Ablation_Study.ipynb` in Google Colab, enable GPU under Runtime → Change runtime type, and run all cells in order.

The notebook runs all four experiments sequentially:
```
pretrain(model, do_nsp=True,  ltr=False)  # Run 1 — Full BERT
pretrain(model, do_nsp=False, ltr=False)  # Run 2 — No NSP
pretrain(model, do_nsp=False, ltr=True)   # Run 3 — LTR
finetune(model, use_bilstm=True)          # Run 4 — BiLSTM
