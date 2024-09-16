# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import numpy as np
import pickle

import torch
import torch.nn.functional as F
from torch.optim import AdamW, lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP

from torch_ema import ExponentialMovingAverage
import torchvision.utils as tu
import torchmetrics

import distributed_util as dist_util
from evaluation import build_resnet50

from . import util
from .network import Image256Net
from .diffusion import Diffusion

from ipdb import set_trace as debug


def mom2norm(sqsums,MASK):
    #sqsums should have shape L,C,H,W
    s=MASK.sum()
    #if len(s)!=4:
     #   raise Exception('shape of sqsums likely not LHCW')
    return torch.sum(torch.flatten(sqsums*MASK[None,...], start_dim=1, end_dim=-1),dim=-1)/s

def imagenorm(img,MASK):
    s=MASK.sum()
    #if len(s)==1: #fix for when img is single dimensional (batch_size,) -> (batch_size,1)
      ##  img=img[:,None]
    n=torch.linalg.norm(torch.flatten(img*MASK[None,...], start_dim=1, end_dim=-1),dim=-1) #flattens non-batch dims and calculates norm
    n/=np.sqrt(s)
    return n

def make_beta_schedule(n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
    # return np.linspace(linear_start, linear_end, n_timestep)
    betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return betas.numpy()

class MLMCRunner(object):
    def __init__(self, opt, log,mlmcoptions=None, save_opt=True):
        super(MLMCRunner,self).__init__()

        # Save opt.
        if save_opt:
            opt_pkl_path = opt.ckpt_path / "options.pkl"
            with open(opt_pkl_path, "wb") as f:
                pickle.dump(opt, f)
            log.info("Saved options pickle to {}!".format(opt_pkl_path))

        betas = make_beta_schedule(n_timestep=opt.interval, linear_end=opt.beta_max / opt.interval)
        betas = np.concatenate([betas[:opt.interval//2], np.flip(betas[:opt.interval//2])])
        self.diffusion = Diffusion(betas, opt.device)
        log.info(f"[Diffusion] Built I2SB diffusion: steps={len(betas)}!")

        noise_levels = torch.linspace(opt.t0, opt.T, opt.interval, device=opt.device) * opt.interval
        self.net = Image256Net(log, noise_levels=noise_levels, use_fp16=opt.use_fp16, cond=opt.cond_x1)
        self.ema = ExponentialMovingAverage(self.net.parameters(), decay=opt.ema)
    
        if opt.load:
            checkpoint = torch.load(opt.load, map_location="cpu")
            self.net.load_state_dict(checkpoint['net'])
            log.info(f"[Net] Loaded network ckpt: {opt.load}!")
            self.ema.load_state_dict(checkpoint["ema"])
            log.info(f"[Ema] Loaded ema ckpt: {opt.load}!")
        
        #MLMC params
        self.M=mlmcoptions.M
        self.Lmax=mlmcoptions.Lmax
        self.Lmin=mlmcoptions.Lmin
        self.mlmc_batch_size=mlmcoptions.batch_size
        self.accsplit=mlmcoptions.accsplit
        self.N0=mlmcoptions.N0
        self.eval_dir=mlmcoptions.eval_dir

        print('Identity mask')
        MASK=torch.ones((3,256,256))
        if 'inpaint-center' in opt.corrupt:
            print('inpainting mask')
            MASK=torch.zeros((3,256,256))
            MASK[:,64:192,64:192]=1.

        self.mom2norm = lambda img : mom2norm(img,MASK)
        self.imagenorm =lambda img : imagenorm(img,MASK)
        if mlmcoptions.payoff=='mean':
            self.payoff=lambda x: torch.clip(x,-1.,1.) #[-1,1]
        elif mlmcoptions.payoff=='second_moment':
            self.payoff=lambda x: torch.clip(x,-1.,1.)**2 #[-1,1]
        else:
            print('mlmcoptions payoff arg not recognised, Setting to mean payoff by default')
            self.payoff=lambda x: torch.clip(x,-1.,1.) #[-1,1]
        
        self.ema.store()
        self.ema.copy_to()
        self.net.to(opt.device)
        self.ema.to(opt.device)
        self.net.eval()
        self.net.diffusion_model = torch.nn.DataParallel(self.net.diffusion_model)
        self.device=opt.device

        self.log = log

    def pred_x0_fn(self,xt, step,cond):
        out = self.net(xt, step, cond=cond)
        std_fwd = self.diffusion.get_std_fwd(step, xdim=xt.shape[1:])
        pred_x0 = xt - std_fwd * out
        return pred_x0

    @torch.no_grad()
    def mlmcsample(self, finesteps, M, corrupt_img,bs,cond=None, mask=None, ot_ode=False, log_steps=None, verbose=True):
        filler=tuple([1 for i in range(len(corrupt_img.shape[1:]))])
        corrupt_img=corrupt_img.detach().repeat(bs,*filler).to(self.device)
        if mask is None:
            xf = corrupt_img.clone().detach().to(self.device)
        else:
            mask=mask.to(self.device)
            xf = (1. - mask) * corrupt_img + mask *torch.randn_like(corrupt_img).to(self.device)
        xc = xf.clone().detach().to(self.device)
        dWc=torch.zeros_like(xc).to(self.device)
        assert len(finesteps)<len(self.diffusion.betas)
        finesteps = finesteps[::-1]
        fine_pair = zip(finesteps[1:], finesteps[:-1])
        counter=0
        coarse_step=finesteps[0]
        for prev_step, step in fine_pair:
            dWf=torch.randn_like(xf).to(self.device)
            xf=self.diffusion.mysampler_fun(self.pred_x0_fn,xf,step,prev_step,dWf,ot_ode,mask,corrupt_img,cond)
            dWc+=dWf*torch.sqrt(torch.tensor([1./M]).to(self.device))
            counter+=1
            if counter==M:
                xc=self.diffusion.mysampler_fun(self.pred_x0_fn,xc,coarse_step,prev_step,dWc,ot_ode,mask,corrupt_img,cond)
                dWc*=0.
                counter=0
                coarse_step=prev_step
        return xf,xc
    
    @torch.no_grad()
    def mlmclooper(self, Nl, l ,M, opt, corrupt_img, mask=None, cond=None, clip_denoise=False, log_count=0, verbose=True):
        # create discrete time steps that split [0, INTERVAL] into NFE sub-intervals.
        # e.g., if NFE=2 & INTERVAL=1000, then STEPS=[0, 500, 999] and 2 network
        # evaluations will be invoked, first from 999 to 500, then from 500 to 0.
        nfe = int(M**l)
        assert 0 < nfe < opt.interval == len(self.diffusion.betas)
        steps = util.space_indices(opt.interval, nfe+1)
        torch.cuda.empty_cache()

        assert cond==None
        # # create log steps
        # log_count = min(len(steps)-1, log_count)
        # log_steps = [steps[i] for i in util.space_indices(len(steps)-1, log_count)]
        # assert log_steps[0] == 0
        # self.log.info(f"[MLMC loop Sampling] steps={opt.interval}, {nfe=}, {log_steps=}!")
              

        num_sampling_rounds = Nl // self.mlmc_batch_size + 1
        numrem=Nl % self.mlmc_batch_size
        for r in range(num_sampling_rounds):
            bs=numrem if r==num_sampling_rounds-1 else self.mlmc_batch_size
            if bs==0:
                break
            with torch.no_grad():
                Xf, Xc = self.mlmcsample(
                    steps, M, corrupt_img, bs, mask=mask, ot_ode=opt.ot_ode, 
                    log_steps=None, verbose=verbose)
            fine_payoff=self.payoff(Xf)
            coarse_payoff=self.payoff(Xc)
            if r==0:
                sums=torch.zeros((3,*fine_payoff.shape[1:])) #skip batch_size
                sqsums=torch.zeros((4,*fine_payoff.shape[1:]))
            sumXf=torch.sum(fine_payoff,axis=0).to('cpu') #sum over batch size
            sumXf2=torch.sum(fine_payoff**2,axis=0).to('cpu')
            if l==self.Lmin:
                sqsums+=torch.stack([sumXf2,sumXf2,torch.zeros_like(sumXf2),torch.zeros_like(sumXf2)])
                sums+=torch.stack([sumXf,sumXf,torch.zeros_like(sumXf)])
            elif l<self.Lmin:
                raise ValueError("l must be at least Lmin")
            else:
                dX_l=fine_payoff-coarse_payoff #Image difference
                sumdX_l=torch.sum(dX_l,axis=0).to('cpu') #sum over batch size
                sumdX_l2=torch.sum(dX_l**2,axis=0).to('cpu')
                sumXc=torch.sum(coarse_payoff,axis=0).to('cpu')
                sumXc2=torch.sum(coarse_payoff**2,axis=0).to('cpu')
                sumXcXf=torch.sum(coarse_payoff*fine_payoff,axis=0).to('cpu')
                sums+=torch.stack([sumdX_l,sumXf,sumXc])
                sqsums+=torch.stack([sumdX_l2,sumXf2,sumXc2,sumXcXf])
        # Directory to save samples. Just to save an example sample for debugging
        if l>self.Lmin:
            this_sample_dir = os.path.join(self.eval_dir, f"level_{l}")
            if not os.path.exists(this_sample_dir):
                os.mkdir(this_sample_dir)
            samples_f=np.clip(.5*(Xf+1.).permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
            samples_c=np.clip(.5*(Xc+1.).permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
            with open(os.path.join(this_sample_dir, "samples_f.npz"), "wb") as fout:
                np.savez_compressed(fout, samplesf=samples_f)
            with open(os.path.join(this_sample_dir, "samples_c.npz"), "wb") as fout:
                np.savez_compressed(fout, samplesc=samples_c)
                                    
        return sums,sqsums 
    
    
    def Giles_plot(self,acc,opt, corrupt_img, mask, cond):
        torch.cuda.empty_cache() 
        #Set mlmc params
        M=self.M
        N0=self.N0
        Lmax=self.Lmax
        eval_dir = self.eval_dir
        Nsamples=1000
        Lmin=self.Lmin
        
        # Directory to save means and norms                                                                                               
        this_sample_dir = os.path.join(eval_dir, f"VarMean_M_{M}_Nsamples_{Nsamples}")
        tpayoffshape=self.payoff(torch.randn(*corrupt_img.shape)).shape[1:]
        sums=torch.zeros((1,3,*tpayoffshape))
        sqsums=torch.zeros((1,4,*tpayoffshape))

        if not os.path.exists(this_sample_dir):
            #Variance and mean samples
            sums=torch.zeros((Lmax+1,*sums.shape[1:]))
            sqsums=torch.zeros((Lmax+1,*sqsums.shape[1:]))
            os.mkdir(this_sample_dir)
            print(f'Proceeding to calculate variance and means with {Nsamples} estimator samples')
            for i,l in enumerate(range(Lmin,Lmax+1)):
                print(f'l={l}')
                sums[i],sqsums[i] = self.mlmclooper(Nsamples, l ,M, opt, corrupt_img, mask, cond, clip_denoise=opt.clip_denoise, log_count=0, verbose=False)
            
            
                # Write samples to disk or Google Cloud Storage
                with open(os.path.join(this_sample_dir, "averages.pt"), "wb") as fout:
                    torch.save(sums/Nsamples,fout)
                with open(os.path.join(this_sample_dir, "sqaverages.pt"), "wb") as fout:
                    torch.save(sqsums/Nsamples,fout)
                with open(os.path.join(this_sample_dir, "avgcost.pt"), "wb") as fout:
                    torch.save(torch.cat((torch.tensor([1]),(1+1./M)*M**torch.arange(1,Lmax+1))),fout)
            
            means_dp=self.imagenorm(sums[:,0])/Nsamples
            V_dp=self.mom2norm(sqsums[:,0])/Nsamples-means_dp**2  
            means_p=self.imagenorm(sums[:,1])/Nsamples
            V_p=self.mom2norm(sqsums[:,1])/Nsamples-means_p**2  
            #Estimate orders of weak (alpha from means) and strong (beta from variance) convergence using LR
            Lmincond=V_dp[1:]<(np.sqrt(M)-1.)**2*V_p[1:]/(1+M) #index of optimal lmin
            cutoff=np.argmax(Lmincond[1:]*Lmincond[:-1])
            means_p=means_p[cutoff:]
            V_p=V_p[cutoff:]
            means_dp=means_dp[cutoff:]
            V_dp=V_dp[cutoff:]
            
            X=np.ones((Lmax-cutoff,2))
            X[:,0]=np.arange(1.,Lmax-cutoff+1)
            a = np.linalg.lstsq(X,np.log(means_dp[1:]),rcond=None)[0]
            alpha = -a[0]/np.log(M)
            Y0=np.exp(a[1])
            b = np.linalg.lstsq(X,np.log(V_dp[1:]),rcond=None)[0]
            beta = -b[0]/np.log(M) 

            print(f'Estimated alpha={alpha}\n Estimated beta={beta}\n')
            with open(os.path.join(this_sample_dir, "info_text.txt"),'w') as f:
                f.write(f'MLMC params: Nsamples={Nsamples}, M={M}, accsplit={self.accsplit}.\n')
                f.write(f'Estimated alpha={alpha}. Estimated beta={beta}. Estimated Lmin={cutoff}.')
            with open(os.path.join(this_sample_dir, "alphabetagamma.pt"), "wb") as fout:
                torch.save(torch.tensor([alpha,beta,cutoff]),fout)
        try:          
            with open(os.path.join(this_sample_dir, "alphabetagamma.pt"),'rb') as f:
                temp=torch.load(f)
                alpha=temp[0].item()
                beta=temp[1].item()
                Lmin=int(temp[-1])
        except:
            pass
        print(alpha,beta,Lmin)
        #Do the calculations and simulations for num levels and complexity plot
        for i in range(len(acc)):
            e=acc[i]
            print(f'Performing mlmc for accuracy={e}')
            sums,sqsums,N=self.mlmc(e,Lmin,alpha_0=alpha,beta_0=beta,opt=opt, corrupt_img=corrupt_img, mask=mask, cond=cond) #sums=[dX,Xf,Xc], sqsums=[||dX||^2,||Xf||^2,||Xc||^2]
            L=len(N)-1+Lmin
            means_p=self.imagenorm(sums[:,1])/N #Norm of mean of fine discretisations
            V_p=torch.clip(self.mom2norm(sqsums[:,1])/N-means_p**2,min=0)

            #cost
            cost_mlmc=torch.sum(N*(M**np.arange(Lmin,L+1)+np.hstack((0,M**np.arange(Lmin,L))))) #cost is number of NFE
            cost_mc=V_p[-1]*(self.M**L)/(e*self.accsplit)**2
            
            
            # Directory to save means, norms and N
            dividerN=N.clone() #add axes to N to broadcast correctly on division
            for i in range(len(sums.shape[1:])):
                dividerN.unsqueeze_(-1)
            this_sample_dir = os.path.join(eval_dir, 'Experiment',f"M_{M}_accuracy_{e}")
            if not os.path.exists(os.path.join(eval_dir, 'Experiment')):
                os.mkdir(os.path.join(eval_dir, 'Experiment'))
                
            if not os.path.exists(this_sample_dir):
                os.mkdir(this_sample_dir)        
            with open(os.path.join(this_sample_dir, "averages.pt"), "wb") as fout:
                torch.save(sums/dividerN,fout)
            with open(os.path.join(this_sample_dir, "sqaverages.pt"), "wb") as fout:
                torch.save(sqsums/dividerN,fout) #sums has shape (L,4,C,H,W)
            with open(os.path.join(this_sample_dir, "N.pt"), "wb") as fout:
                torch.save(N,fout)
            with open(os.path.join(this_sample_dir, "costs.npz"), "wb") as fout:
               np.savez_compressed(fout,costmlmc=np.array(cost_mlmc),costmc=np.array(cost_mc))

            meanimg=torch.sum(sums[:,0]/dividerN[:,0,...],axis=0)#cut off one dummy axis
            meanimg=torch.clip(.5*meanimg+.5,0.,1.).permute(1,2,0).cpu().numpy()
            # Write samples to disk or Google Cloud Storage
            with open(os.path.join(this_sample_dir, "meanpayoff.npz"), "wb") as fout:
                np.savez_compressed(fout, meanpayoff=meanimg)

        return None

    def mlmc(self,accuracy,Lmin,alpha_0,beta_0, opt, corrupt_img, mask, cond, clip_denoise=False, log_count=0, verbose=False):
        accsplit=self.accsplit
        #Orders of convergence
        alpha=max(0,alpha_0)
        beta=max(0,beta_0)
        M=self.M
        N0=self.N0
        Lmax=self.Lmax
        L=Lmin+1

        mylen=L+1-Lmin
        V=torch.zeros(mylen) #Initialise variance vector of each levels' variance
        N=torch.zeros(mylen) #Initialise num. samples vector of each levels' num. samples
        dN=N0*torch.ones(mylen) #Initialise additional samples for this iteration vector for each level
        sqrt_cost=torch.sqrt(M**torch.arange(Lmin,L+1.)+torch.hstack((torch.tensor([0.]),M**torch.arange(Lmin,1.*L))))
        it0_ind=False
        while (torch.sum(dN)>0): #Loop until no additional samples asked for
            mylen=L+1-Lmin
            for i,l in enumerate(torch.arange(Lmin,L+1)):
                num=dN[i]
                if num>0: #If asked for additional samples...
                    tempsums,tempsqsums=self.mlmclooper(int(num), l ,M, opt, corrupt_img, mask, cond, clip_denoise, log_count, verbose) #Call function which gives sums
                    if not it0_ind:
                        sums=torch.zeros((mylen,*tempsums.shape)) #Initialise sums array of unnormed [dX,Xf,Xc], each column is a level
                        sqsums=torch.zeros((mylen,*tempsqsums.shape)) #Initialise sqsums array of normed [dX^2,Xf^2,Xc^2,XcXf], each column is a level
                        it0_ind=True
                    sqsums[i,...]+=tempsqsums
                    sums[i,...]+=tempsums
                    
            N+=dN #Increment samples taken counter for each level
            Yl=(self.imagenorm(sums[:,0])/N).to(torch.float32)
            cf=1
            while True:
                V=torch.clip(self.mom2norm(sqsums[:,0])/N-(Yl)**2,min=0.).to(torch.float32) #Calculate variance based on updated samples                                
                ##Fix to deal with zero variance or mean by linear extrapolation                                                                                         
                V[2:]=torch.maximum(V[2:],.5*V[1:-1]*M**(-beta))
                cf_=torch.sqrt((1+V/(N*Yl**2)))
                Y_corrected=(self.imagenorm(sums[:,0])/N).to(torch.float32)/cf_ #correct for bias                                                                        
                Yl=Y_corrected
                if torch.max(cf_-cf)<.01:
                    break
                cf=cf_
            
            #Estimate order of weak convergence using LR
            #Yl=(M^alpha-1)khl^alpha=(M^alpha-1)k(TM^-l)^alpha=((M^alpha-1)kT^alpha)M^(-l*alpha)
            #=>log(Yl)=log(k(M^alpha-1)T^alpha)-alpha*l*log(M)
            X=torch.ones((mylen-1,2))
            X[:,0]=torch.arange(1,mylen)
            a = torch.linalg.lstsq(X,torch.log(Yl[1:]))[0]
            alpha_ = max(-a[0]/np.log(M),0.)
            b = torch.linalg.lstsq(X,torch.log(V[1:]))[0]
            beta_= -b[0]/np.log(M)
            if alpha_0==-1:
                alpha=alpha_
            if beta_0==-1:
                beta=beta_
                
            sqrt_V=torch.sqrt(V)
            Nl_new=torch.ceil(2*((accuracy)**-2)*torch.sum(sqrt_V*sqrt_cost)*(sqrt_V/sqrt_cost)) #Estimate optimal number of samples/level
            dN=torch.clip(Nl_new-N,min=0) #Number of additional samples
            print(f'Asking for {dN} new samples for l=[{Lmin,L}]')
            if torch.sum(dN > 0.01*N).item() == 0: #Almost converged
                test=max(Yl[-2]/(M**alpha),Yl[-1]) if L-Lmin>1 else Yl[-1]
                print(f'sqrt_variance={(2*V/N).sum().sqrt()}, bias={np.sqrt(2)*test/(M**alpha-1.)}.')
                if test>(M**alpha-1)*accuracy*np.sqrt(.5): 
                    L+=1
                    print(f'Increased L to {L}')
                    if (L>Lmax):
                        print('Asked for an L greater than maximum allowed Lmax. Ending MLMC algorithm.')
                        break
                    #Add extra entries for the new level and estimate sums with N0 samples 
                    V=torch.cat((V,V[-1]*M**(-beta)*torch.ones(1)), dim=0)
                    sqrt_V=torch.sqrt(V)
                    newcost=torch.sqrt(torch.tensor([M**L+M**((L-1.))]))
                    sqrt_cost=torch.cat((sqrt_cost,newcost),dim=0)
                    Nl_new=torch.ceil(((accsplit*accuracy)**-2)*torch.sum(sqrt_V*sqrt_cost)*(sqrt_V/sqrt_cost)) #Estimate optimal number of sample
                    N=torch.cat((N,torch.tensor([0])),dim=0)
                    dN=torch.clip(Nl_new-N,min=0) #Number of additional samples
                    print(f'With new L, estimate of {dN} new samples for l=[{Lmin,L}]')
                    sums=torch.cat((sums,torch.zeros((1,*sums[0].shape))),dim=0)
                    sqsums=torch.cat((sqsums,torch.zeros((1,*sqsums[0].shape))),dim=0)
                    
        print(f'Estimated alpha = {alpha_}')
        print(f'Estimated beta = {beta_}')
        return sums,sqsums,N
