# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 14:40:30 2023

@author: lshaw
"""

# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import copy
import argparse
import random
from pathlib import Path
from easydict import EasyDict as edict

import numpy as np
import datetime
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
from torch.utils.data import DataLoader, Subset
from torch_ema import ExponentialMovingAverage
import torchvision.utils as tu

from logger import Logger
import distributed_util as dist_util
from i2sb import MLMCRunner, download_ckpt
from corruption import build_corruption
from dataset import imagenet
from i2sb import ckpt_util

import colored_traceback.always
from ipdb import set_trace as debug

RESULT_DIR = Path("results")

def set_seed(seed):
    # https://github.com/pytorch/pytorch/issues/7068
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.

def build_dataset(opt, log, corrupt_type):
    if "sr4x" in corrupt_type:
        val_dataset = imagenet.build_lmdb_dataset(opt, log, train=False) # full 50k val
    elif "inpaint" in corrupt_type:
        mask = corrupt_type.split("-")[1]
        val_dataset = imagenet.InpaintingValSubset(opt, log, mask) # subset 10k val + mask
    elif corrupt_type == "mixture":
        from corruption.mixture import MixtureCorruptDatasetVal
        val_dataset = imagenet.build_lmdb_dataset(opt, log, train=False)
        val_dataset = MixtureCorruptDatasetVal(opt, val_dataset) # subset 10k val + mixture
    else:
        val_dataset = imagenet.build_lmdb_dataset(opt, log,train=False) # subset 10k val

    return val_dataset

def compute_batch(ckpt_opt, corrupt_type, corrupt_method, out):
    if "inpaint" in corrupt_type:
        clean_img, y, mask = out
        corrupt_img = clean_img * (1. - mask) + mask
        x1          = clean_img * (1. - mask) + mask * torch.randn_like(clean_img)
    elif corrupt_type == "mixture":
        clean_img, corrupt_img, y = out
        mask = None
    else:
        clean_img, y = out
        mask = None
        corrupt_img = corrupt_method(clean_img.to(opt.device))
        x1 = corrupt_img.to(opt.device)

    cond = x1.detach() if ckpt_opt.cond_x1 else None
    if ckpt_opt.add_x1_noise: # only for decolor
        x1 = x1 + torch.randn_like(x1)

    return corrupt_img, x1, mask, cond, y

@torch.no_grad()
def main(opt):
    log = Logger(0, ".log")

    # get (default) ckpt option
    ckpt_opt = ckpt_util.build_ckpt_option(opt, log, RESULT_DIR / opt.ckpt)
    corrupt_type = ckpt_opt.corrupt
    
    # build corruption method
    corrupt_method = build_corruption(opt, log, corrupt_type=corrupt_type)

    # build imagenet val dataset
    valdataset= build_dataset(opt, log, corrupt_type)
    
    # build runner
    imgnum=0
    sample_dir = RESULT_DIR / opt.ckpt / f'_{imgnum}_{opt.payoff}_temp'
    if opt.savedir is not None:
        sample_dir=opt.savedir
    os.makedirs(sample_dir, exist_ok=True)
    mlmcoptions=edict(M=opt.M,N0=opt.N0,Lmin=opt.Lmin,Lmax=opt.Lmax,payoff=opt.payoff,
                      accsplit=opt.accsplit,eval_dir=sample_dir,acc=opt.acc,batch_size=opt.batch_size)
    
    runner = MLMCRunner(ckpt_opt, log, mlmcoptions,save_opt=False)

    # handle use_fp16 for ema
    if opt.use_fp16:
        raise Exception('fp16 not enabled')
        runner.ema.copy_to() # copy weight from ema to net
        runner.net.diffusion_model.convert_to_fp16()
        runner.ema = ExponentialMovingAverage(runner.net.parameters(), decay=0.99) # re-init ema with fp16 weight
        
    val_loader = DataLoader(valdataset,
        batch_size=1, shuffle=False, pin_memory=True, num_workers=1, drop_last=False,
    )
    
    for it,out in enumerate(val_loader):
        if it==imgnum:
            corrupt_img, x1, mask, cond, y = compute_batch(ckpt_opt, corrupt_type, corrupt_method, out)
        
            y = y.to(opt.device)

            tu.save_image((out[0]+1)/2, os.path.join(mlmcoptions.eval_dir,"clean.png"))
            tu.save_image((corrupt_img+1)/2, os.path.join(mlmcoptions.eval_dir,"corrupt.png"))
            tu.save_image((x1+1)/2, os.path.join(mlmcoptions.eval_dir,"x1.png"))
            with open(os.path.join(mlmcoptions.eval_dir, "label.pt"), "wb") as fout:
                torch.save(y,fout)
            with open(os.path.join(mlmcoptions.eval_dir, "mask.pt"), "wb") as fout:
                torch.save(mask,fout)

        #corrupt_img, x1, mask, cond, y=corrupt_img[0], x1[0], mask[0], cond[0], y[0]
            runner.Giles_plot(opt.acc,ckpt_opt, corrupt_img.to('cpu'), mask=mask, cond=cond)
        if it==imgnum:
            break
    del runner
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",           type=int,  default=0)
    parser.add_argument("--n-gpu-per-node", type=int,  default=1,           help="number of gpu on each node")
    parser.add_argument("--master-address", type=str,  default='localhost', help="address for master")
    parser.add_argument("--node-rank",      type=int,  default=0,           help="the index of node")
    parser.add_argument("--num-proc-node",  type=int,  default=1,           help="The number of nodes in multi node env")

    # data
    parser.add_argument("--image-size",     type=int,  default=256)
    parser.add_argument("--dataset-dir",    type=Path, default="/dataset",  help="path to LMDB dataset")

    # sample
    parser.add_argument("--batch-size",     type=int,  default=20)
    parser.add_argument("--ckpt",           type=str,  default=None,        help="the checkpoint name from which we wish to sample")
    parser.add_argument("--clip-denoise",   action="store_true",  default=False,          help="clamp predicted image to [-1,1] at each")
    parser.add_argument("--use-fp16",       action="store_true",            help="use fp16 network weight for faster sampling")
    
    #mlmc
    parser.add_argument("--M",     type=int,  default=2)
    parser.add_argument("--N0",           type=int,  default=100,        help="initial MLMC samples")
    parser.add_argument("--Lmin",   type=int, default=3,          help="minimum level")
    parser.add_argument("--Lmax",       type=int, default=8,          help="maximum level")
    parser.add_argument("--accsplit",       type=float, default=np.sqrt(.5),          help="bias-variance splitting")
    parser.add_argument("--acc",       type=float, nargs='+', help="accuracies for MLMC")
    parser.add_argument("--payoff",       type=str, default='second_moment',          help="payoff for MLMC")
    parser.add_argument("--savedir",       type=str, default=None,          help="use existing file")
    arg = parser.parse_args()
   
    opt = edict(
        device="cuda",
    )
    opt.update(vars(arg))

    # one-time download: ADM checkpoint
    download_ckpt("data/")

    main(opt)
