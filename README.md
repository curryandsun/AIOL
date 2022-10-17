# Adaptive In-Out-aware Learning (AIOL)
This is a PyTorch implementation of the Adaptive In-Out-aware Learning (AIOL) method described in the paper [Exploiting Mixed Unlabeled Data for Detecting Samples of Seen and Unseen Out-of-Distribution Classes](https://arxiv.org/abs/2210.06833).

## Abstract
Out-of-Distribution (OOD) detection is essential in real-world applications, which has attracted increasing attention in recent years. However, most existing OOD detection methods require many labeled In-Distribution (ID) data, causing a heavy labeling cost. In this paper, we focus on the more realistic scenario, where limited labeled data and abundant unlabeled data are available, and these unlabeled data are mixed with ID and OOD samples. We propose the Adaptive In-Out-aware Learning (AIOL) method, in which we employ the appropriate temperature to adaptively select potential ID and OOD samples from the mixed unlabeled data and consider the entropy over them for OOD detection. Moreover, since the test data in realistic applications may contain OOD samples whose classes are not in the mixed unlabeled data (we call them unseen OOD classes), data augmentation techniques are brought into the method to further improve the performance. The experiments are conducted on various benchmark datasets, which demonstrate the superiority of our method.

## Usage

### Dataset
The used datasets in this paper can be found in:
- [ImageNet32](https://arxiv.org/abs/1707.08819)
- [iSUN and LSUN](https://github.com/facebookresearch/odin)
- [Texture and Places365](https://github.com/hendrycks/outlier-exposure)

### Train
Train the model by 1000 labeled data of CIFAR-10 dataset with ImageNet as seen OOD:

```
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --num-labeled 1000 --expand-labels --seed 5 --out <your out_dir> --seen-ood imagenet32 --calibrate --calibrate-start 40
```

### Evaluate
Evaluate the OOD detection performance of the trained model:

```
CUDA_VISIBLE_DEVICES=0 python eval.py --dataset cifar10 --seen-ood imagenet32 --seed 5 --resume <your ckpt_dir>
```

### Monitoring training progress
```
tensorboard --logdir=<your out_dir>
```

## Requirements
- python 3.6+
- torch 1.4
- torchvision 0.5
- tensorboard
- numpy
- tqdm
- apex (optional)


## References
- [Unofficial Pytorch implementation of FixMatch](https://github.com/kekmodel/FixMatch-pytorch)
- [Official Pytorch implementation of SSD: A Unified Framework for Self-Supervised Outlier Detection](https://github.com/inspire-group/SSD)


## Citations
If you find this repo is helpful for your experiment or your research paper, please think about kindly citing our paper as follow:
```
@article{sun2022exploiting,
  title={Exploiting Mixed Unlabeled Data for Detecting Samples of Seen and Unseen Out-of-Distribution Classes},
  author={Sun, Yi-Xuan and Wang, Wei},
  booktitle = {Proceedings of the 36th AAAI Conference on Artificial Intelligence},
  year={2022}
}
```
