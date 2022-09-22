# MASR: Memory Bank Augmented Long-tail Sequential Recommendation
This repository contains the source code of our paper, Memory Bank Augmented Long-tail Sequential Recommendation, which is accepted for publication at CIKM 2022.
The implement of the sequence encoder follows [BERT4Rec](https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch.git).



## Requirements

- Python 3.8.3
- Pytorch 1.10.2
- Transformers 4.15.0
- wget
- Tqdm

Training:
```
python --template train_bert main.py --use_memory_loss_tail -copy_tail_label
```
