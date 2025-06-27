## Introduction
This is the source code of [Cephalometric Landmark Detection by Attentive Feature Pyramid Fusion and Regression-Voting](https://arxiv.org/pdf/1908.08841.pdf). The paper is early accepted in MICCAI 2019.



## Prerequistes
- Python 3.8
- PyTorch 1.0.0-1.7.0

## Dataset and setup
- Download the dataset from the official [webside](https://figshare.com/s/37ec464af8e81ae6ebbf) and put it in the root. We also provide the processed [dataset](https://drive.google.com/file/d/1fGoaoZbp04NVi41jcEoEmbbCxrUcpjUO/view?usp=sharing) for a quick start.

## Training and validation
- python main.py

## Continue Training
The system now supports continuing training from an existing model checkpoint. This is useful for:
- Resuming training after interruption
- Extending training beyond the original number of epochs
- Fine-tuning with different parameters

### Usage
1. **Continue training (default behavior):**
   ```bash
   python main.py
   ```

2. **Start training from scratch (ignore existing model):**
   ```bash
   python main.py --continue_training 0
   ```

3. **Continue training with custom parameters:**
   ```bash
   python main.py --epochs 500 --batchSize 2 --continue_training 1
   ```

### How it works
- If `model/model.pth` exists and `--continue_training` is enabled (default), the system will:
  - Load the model state from the checkpoint
  - Load training history from `model/loss.npy` (if available)
  - Load optimizer state from `model/model_optimizer.pth` (if available)
  - Resume training from the last completed epoch
- The system saves checkpoints every 10 epochs and at the end of training
- Training history is preserved and extended

### Files created during training
- `model/model.pth`: Model state dictionary
- `model/model_optimizer.pth`: Optimizer state dictionary
- `model/loss.npy`: Training loss history

## Reference

If you found this code useful, please cite the following paper:

```
@inproceedings{chen2019cephalometric,
  title={Cephalometric landmark detection by attentive feature pyramid fusion and regression-voting},
  author={Chen, Runnan and Ma, Yuexin and Chen, Nenglun and Lee, Daniel and Wang, Wenping},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={873--881},
  year={2019},
  organization={Springer}
}
```