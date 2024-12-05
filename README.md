# Rapid Histopathology Pre-training (RHP)
The official code of 《Efficient Self-Supervised Learning for Pathology Image Analysis via Masking》


# Train
#### Stage 1: masked pre-training
python train.py --db_path your_db_path --mask_ratio 0.75 --bs your_batch_size

#### Stage 2: unmasked tuning
python train.py --db_path your_db_path --ckpt ckpt_stage1 --mask_ratio 0 --bs int(your_batch_size*0.25)


# Acknowledgement
This repository is built using the [BYOL](https://github.com/lucidrains/byol-pytorch) repository and the [MAE](https://github.com/facebookresearch/mae) repository.
