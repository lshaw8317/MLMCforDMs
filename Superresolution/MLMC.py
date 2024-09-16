import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import numpy as np
from collections import OrderedDict

if __name__ == "__main__":
    MCSAMPLES=int(1e6)
    MCL=9
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_16_128.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['val'],
                        help='Run val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-acc', '--accuracy', type=float, nargs='+',default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-payoff', '--payoff', type=str, choices =['mean','second_moment','proj_mean'], default='second_moment')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')
    parser.add_argument('-mode','--mode', type=str, choices=['MLMC','MC'],default='MLMC')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    # logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        import wandb
        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None

    # dataset phase == 'val'
    phase=opt['phase']
    assert phase=='val'
    dataset_opt=opt['datasets'][phase]
    val_set = Data.create_dataset(dataset_opt, phase)
    val_loader = Data.create_dataloader(val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    temp_path=os.path.join(opt['datasets'][phase]['dataroot'],'mask_128')
    for maskfile in os.listdir(temp_path):
        #Fudge, should only be one mask
        with open(os.path.join(temp_path,maskfile),'rb') as f:
            mask=torch.load(f)
            opt['datasets']['MASK']=1.*mask #convert to float
    diffusion = Model.create_model(opt)
    diffusion.eval_dir=diffusion.eval_dir
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    logger.info('Initial Model Finished')
    
    logger.info('Begin Model Evaluation.')
    avg_psnr = 0.0
    avg_ssim = 0.0
    idx = 0
    result_path = diffusion.eval_dir 
    os.makedirs(result_path, exist_ok=True)
    if args.mode=='MLMC':
        #Modify noise schedule to correspond to MLMC max L in diffusion
        MLMCsteps=(diffusion.M**diffusion.Lmax)
        diffusion.set_new_noise_schedule(
            opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'],MLMCsteps=MLMCsteps)
        acc=args.accuracy
        for _,  val_data in enumerate(val_loader):
            #val_data automatically has batch size 1 for phase=val
            idx += 1
            diffusion.feed_data(val_data) #loads in self.data['SR'] which is accessed by self.mlmc
            diffusion.Giles_plot(acc)
            visuals=OrderedDict()
            visuals['INF'] = diffusion.data['SR'].detach().float().cpu()
            visuals['HR'] = diffusion.data['HR'].detach().float().cpu()
            if 'LR' in diffusion.data:
                visuals['LR'] = diffusion.data['LR'].detach().float().cpu()
    
            hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
            lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
            fake_img = Metrics.tensor2img(visuals['INF'])  # uint8
    
            Metrics.save_img(
                hr_img, '{}/hr.png'.format(result_path))
            Metrics.save_img(
                lr_img, '{}/lr.png'.format(result_path))
            Metrics.save_img(
                fake_img, '{}/inf.png'.format(result_path))
            if idx>0:
                break
    else:#args.mode=='MC'
        #Modify noise schedule to correspond to desired MC L
        opt['model']['beta_schedule'][opt['phase']]['n_timestep']=diffusion.M**MCL
        diffusion.set_new_noise_schedule(
            opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])

        for _,  val_data in enumerate(val_loader):
            #val_data automatically has batch size 1 for phase=val
            idx += 1
            diffusion.feed_data(val_data) #loads in self.data['SR'] which is accessed by self.mlmc
            diffusion.mc(MCSAMPLES,MCL)
            visuals=OrderedDict()
            visuals['INF'] = diffusion.data['SR'].detach().float().cpu()
            visuals['HR'] = diffusion.data['HR'].detach().float().cpu()
            if 'LR' in diffusion.data:
                visuals['LR'] = diffusion.data['LR'].detach().float().cpu()
    
            hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
            lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
            fake_img = Metrics.tensor2img(visuals['INF'])  # uint8
    
            Metrics.save_img(
                hr_img, '{}/hr.png'.format(result_path))
            Metrics.save_img(
                lr_img, '{}/lr.png'.format(result_path))
            Metrics.save_img(
                fake_img, '{}/inf.png'.format(result_path))
            if idx>0:
                break

