# MLMCforDMs

Code for "Accelerating Bayesian
computation with deep
generative samplers by
Multilevel Monte Carlo" Haji-Ali et. al. (2024) adapted from various sources.

## Denoising
Adapted from https://github.com/yang-song/score_sde_pytorch associated to
```bib
@inproceedings{
  song2021scorebased,
  title={Score-Based Generative Modeling through Stochastic Differential Equations},
  author={Yang Song and Jascha Sohl-Dickstein and Diederik P Kingma and Abhishek Kumar and Stefano Ermon and Ben Poole},
  booktitle={International Conference on Learning Representations},
  year={2021},
  url={https://openreview.net/forum?id=PxTIG12RRHS}
}
```
Used under Apache-2.0 license.

Make sure that you download CIFAR10 to the appropriate directory (see ``main.py``) and then run e.g.
```
python main.py --workdir exp/checkpoints/cifar10_ddpmpp_continuous --acc 0.0086 --config configs/vp/cifar10_ddpmpp_continuous.py --eval_folder exp/eval/cifar10DenoisingSecondmoment_0.8 --MLMCsampler EXPINT --probflow=False --conditional=True --payoff secondmoment --conditional_noise .8
```
## Superresolution
Adapted from https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement associated to

[Paper](https://arxiv.org/pdf/2104.07636.pdf ) |  [Project](https://iterative-refinement.github.io/ )

Run e.g.
```
python MLMC.py -acc 0.018 -gpu 1
```

## Inpainting
Adapted from https://github.com/NVlabs/I2SB associated to 
```
@article{liu2023i2sb,
  title={I{$^2$}SB: Image-to-Image Schr{\"o}dinger Bridge},
  author={Liu, Guan-Horng and Vahdat, Arash and Huang, De-An and Theodorou, Evangelos A and Nie, Weili and Anandkumar, Anima},
  journal={arXiv preprint arXiv:2302.05872},
  year={2023},
}
```
Used under the licence: 
Copyright © 2023, NVIDIA Corporation. All rights reserved.

This work is made available under the Nvidia Source Code License-NC.

The model checkpoints are shared under [CC-BY-NC-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode). If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.

Run e.g.
```
python mlmc.py --dataset-dir datasetdir --ckpt inpaint-center --batch-size 12 
```
