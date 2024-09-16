# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 08:51:39 2023

@author: lshaw
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.ticker as ticker
from matplotlib.legend import Legend
from PIL import Image
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts} \usepackage{amsmath}',
    'legend.fontsize':5
})
plt.rcParams['figure.dpi'] = 1000
plt.rcParams['savefig.dpi'] = 1000
M=2
Nsamples=1000

def PSNR(eps):
    #assuming [-1,1]
    return np.round(eps,4)
    # return eps
    
def imagenorm(img,MASK):
    s=MASK.sum()
    n=torch.linalg.norm(torch.flatten(img*MASK[None,...], start_dim=1, end_dim=-1),dim=-1) #flattens non-batch dims and calculates norm
    n/=np.sqrt(s)
    return n

def mom2norm(sqsums,MASK):
    s=MASK.sum()
    return torch.sum(torch.flatten(sqsums*MASK[None,...], start_dim=1, end_dim=-1),dim=-1)/s


# Set plotting params
fig,_=plt.subplots(1,4,figsize=(8,2.5))
# fig,_=plt.subplots(2,2,figsize=(4,4))
# markersize=(fig.get_size_inches()[0])
markersize=4
axis_list=fig.axes
# expdir='MLMCforI2SB/results/inpaint-center/_0_second_moment'
expdir='Image-Super-Resolution-via-Iterative-Refinement/results/sr_sr3_16_128_second_momenttest'
# expdir = 'score_sde_pytorch/exp/eval/cifar10DenoisingSecondmoment_0.8'

switcher= 'Superresolution' #expdir.split('_')[-1]
label=switcher +' Second Moment'


#Do the calculations and simulations for num levels and complexity plot
files = [os.path.join(expdir,'Experiment',f) for f in os.listdir(expdir+'/Experiment') if f.startswith('M_2')]
acc=np.zeros(len(files))
realvar=np.zeros(len(files))
realbias=np.zeros(len(files))

cost_mlmc=np.zeros(len(files))
cost_mc=np.zeros(len(files))
# plt.rc('axes', titlesize=2)     # fontsize of the axes title
larr=np.array([int(f.split('_')[-1]) for f in os.listdir(expdir) if f.startswith('level')])
if switcher.lower()=='denoising':
    with open(os.path.join(expdir,'Denoising_0.8/x0.pt'),'rb') as f:
        x0=.5*(torch.load(f)+1.)
    with open(os.path.join(expdir,'Denoising_0.8/v.pt'),'rb') as f:
        v=torch.load(f)
    MASK=torch.ones_like(x0)
    fig2,ax=plt.subplots(1,len(files)+1)
    plt.figure(fig2)
    ax[-1].imshow(x0.permute(1,2,0))
    plt.rc('axes', titlesize=8)     # fontsize of the axes title
    plt.rc('axes', labelsize=8)     # fontsize of the axes title

    ax[-1].set_title('Reference')
    ax[-1].set_axis_off()
    Lmin=3
    
if switcher.lower()=='superresolution':
    # with open(os.path.join('Image-Super-Resolution-via-Iterative-Refinement/results/sr_sr3_16_128_MC','PCA_0.pt'),'rb') as f:
    #     pca0=torch.load(f)
    # def imagenorm(img,MASK):
    #     n=torch.sum(torch.flatten(img[:,:,50:70,35:95],start_dim=1,end_dim=-1)*pca0[None,...],dim=-1)
    #     return np.abs(n)

    # def mom2norm(sqsums,MASK):
    #     return sqsums

    with open(os.path.join(expdir,'hr.png'),'rb') as f:
        x0 = np.array(Image.open(f))
    # with open('dataset/celebahq_16_128/mask_128/Mask00031.pt','rb') as f:
    #     MASK = torch.load(f)
        
    MASK=torch.ones_like(torch.tensor(x0.transpose(2,0,1)))
    fig2,ax=plt.subplots(1,len(files)+1)
    plt.figure(fig2)
    ax[-1].imshow(x0)
    plt.rc('axes', titlesize=8)     # fontsize of the axes title
    plt.rc('axes', labelsize=8)     # fontsize of the axes title

    ax[-1].set_title('Reference')
    ax[-1].set_axis_off()
    Lmin=4

if switcher.lower()=='inpainting':
    with open(os.path.join(expdir,'clean.png'),'rb') as f:
        x0 = np.array(Image.open(f))
    MASK=torch.zeros_like(torch.tensor(x0.transpose(2,0,1)))
    imsize=MASK.shape[-1]
    MASK[:,imsize//4:3*imsize//4,imsize//4:3*imsize//4]=1.
    fig2,ax=plt.subplots(1,len(files)+1)
    plt.figure(fig2)
    ax[-1].imshow(x0)
    plt.rc('axes', titlesize=8)     # fontsize of the axes title
    plt.rc('axes', labelsize=8)     # fontsize of the axes title

    ax[-1].set_title('Reference')
    ax[-1].set_axis_off()
    Lmin=3

lseq=[]
markers=['s','d','x','P','*','p','+','.','o']+['s','d','x','P','*','p','+','.','o']
for i,f in enumerate(files):
    e=float(f.split('_')[-1])
    acc[i]=e
    # Load saved data
    with open(os.path.join(f, "averages.pt"), "rb") as fout:
        avgs=torch.load(fout)
    with open(os.path.join(f, "N.pt"), "rb") as fout:
        N=torch.load(fout)
    with open(os.path.join(f, "sqaverages.pt"), "rb") as fout:
        sqavgs=torch.load(fout)
    with np.load(os.path.join(f,'meanpayoff.npz'),allow_pickle=True) as data:
        meanimg=data['meanpayoff']
    L=len(N)-1 
    lseq+=[L]
    if i==0:
        this_sample_dir = os.path.join(expdir,f"VarMean_M_{M}_Nsamples_{Nsamples}")
        with open(os.path.join(this_sample_dir, "averages.pt"), "rb") as fout:
            avgs=torch.load(fout)
        with open(os.path.join(this_sample_dir, "sqaverages.pt"), "rb") as fout:
            sqavgs=torch.load(fout)
        Lmax=len(avgs)-1
        means_p=imagenorm(avgs[:,1],MASK)
        V_p=mom2norm(sqavgs[:,1],MASK)-means_p**2 
        means_dp=imagenorm(avgs[:,0],MASK)
        V_dp=mom2norm(sqavgs[:,0],MASK)-means_dp**2  
        means_dp=imagenorm(avgs[:,0],MASK)
        X=np.ones((Lmax-Lmin,2))
        X[:,0]=np.arange(1.,Lmax-Lmin+1)
        a = np.linalg.lstsq(X,np.log(means_dp[Lmin+1:]),rcond=None)[0]
        alpha =-a[0]/np.log(M)
        eps0=np.exp(a[1])/(M**alpha-1.)*np.sqrt(2)
        
        Lmincond=V_dp[1:]<(np.sqrt(M)-1.)**2*V_p[-1]/(1+M) #index of optimal lmin                                                     
        cutoff=np.argmax(Lmincond[1:]*Lmincond[:-1])
        
        b = np.linalg.lstsq(X,np.log(V_dp[Lmin+1:]),rcond=None)[0]
        beta = -b[0]/np.log(M)
        V0=np.exp(b[1])
        
        if switcher.lower()=='inpainting':
            Lmax=9
            means_dp=torch.cat((means_dp,means_dp[-1:]/M**alpha))
            V_p=torch.cat((V_p,V_p[-1:]))
            V_dp=torch.cat((V_dp,V_dp[-1:]/M**beta))
            means_p=torch.cat((means_p,means_p[-1:]))

        VMC=V_p[Lmin+1:]
        tempL=torch.arange(1,Lmax-Lmin+1,dtype=torch.float32)
        theory_epsilon=(means_dp[Lmin+1:]*np.sqrt(2)/(M**alpha-1.))
        # theory_epsilon=eps0*M**(-alpha*tempL)
        sV0=np.sqrt(V_p[Lmin])

        cumsum2=np.cumsum(np.sqrt(V_dp[Lmin+1:]*M**(tempL)*(1.+1./M)))
        factor=(sV0+cumsum2)**2
        theory_ratio=M**(-tempL)*factor/(VMC)

        theory_samps= lambda eps, l: 2/(eps**2)*(sV0+cumsum2[l-Lmin])*np.hstack((sV0,np.sqrt(V_dp[Lmin+1:]/(M**(tempL)*(1.+1./M)))))[:l-Lmin+1]
        
    Vdp=V_dp[Lmin:Lmin+len(N)].clone()
    Vdp[0]=V_p[Lmin]
    mdp=means_dp[Lmin:Lmin+len(N)].clone()
    mdp[0]=means_p[Lmin]
    Vp=V_p[Lmin:Lmin+len(N)].clone()
    # with open(os.path.join(f, "averages.pt"), "rb") as fout:
    #     avgs=torch.load(fout)
    # with open(os.path.join(f, "N.pt"), "rb") as fout:
    #     N=torch.load(fout)
    # with open(os.path.join(f, "sqaverages.pt"), "rb") as fout:
    #     sqavgs=torch.load(fout)
    # mp=imagenorm(avgs[:,1],MASK)
    # Vp=mom2norm(sqavgs[:,1],MASK)-mp**2 
    # mdp_=imagenorm(avgs[:,0],MASK)
    # print(mdp[1],mdp_[1])
    # Vdp=mom2norm(sqavgs[:,0],MASK)-mdp**2  
    realvar[i]=torch.sum(Vdp/N)
    realbias[i]=((mdp[-1]/(M**alpha-1.))**2)
    
    with np.load(os.path.join(f, "costs.npz")) as fout:
        cost_mlmc[i] = fout['costmlmc']
        cost_mc[i]= fout['costmc']
        
    axis_list[2].semilogy(Lmin+np.arange(L+1),N,'k-',marker=markers[i],label='$'+f'{PSNR(e)}'+'$',markersize=markersize,
                   markerfacecolor='k',markeredgecolor='k', markeredgewidth=1)
    # axis_list[2].semilogy(Lmin+np.arange(L+1),theory_samps(e,L+Lmin),'k--',marker=markers[i],markersize=markersize,
    #                 markerfacecolor='k',markeredgecolor='k', markeredgewidth=1)
    
    if True:
        plt.figure(fig2)
        plt.rc('axes', titlesize=8)     # fontsize of the axes title
        plt.rc('axes', labelsize=8)     # fontsize of the axes title

        ax[i].imshow(meanimg)
        ax[i].set_title('$'+str(PSNR(np.sqrt(realvar[i]+realbias[i])))+f'({PSNR(e)})$')
        ax[i].set_axis_off()

#Plot variances
axis_list[0].semilogy(range(Lmax+1),V_p,'k--',label='$\mathbb{V}[f({\widehat X}_\ell)]$',
                  marker='None',markersize=markersize,markerfacecolor="None",markeredgecolor='k', markeredgewidth=1,base=2)
axis_list[0].semilogy(range(1,Lmax+1),V_dp[1:],'k-',label='$V_{\ell}$',
                  marker='None', markersize=markersize, markerfacecolor="None", markeredgecolor='k',base=2,
                  markeredgewidth=1)
axis_list[0].set_ylim([V_dp[1:].min()/2,2*V_p.max()])

axis_list[0].semilogy(range(3,Lmax+1),4*V_dp[3]*M**(-beta*torch.arange(0,Lmax-2)),'k:',label=f'Ref. line $\\beta={round(beta,2)}$',base=2)
#Plot means
axis_list[1].semilogy(range(Lmax+1),means_p,'k--',label='$\|\mathbb{E}[f({\widehat X}_\ell)]\|$',
                  marker='None', markersize=markersize, markerfacecolor="None",markeredgecolor='k',base=2,
                  markeredgewidth=1)
axis_list[1].semilogy(range(1,Lmax+1),means_dp[1:],'k-',label='$\|Y_{\ell}\|$',
                  marker='None',markersize=markersize,markerfacecolor="None",markeredgecolor='k', markeredgewidth=1,base=2)
axis_list[1].semilogy(range(2,Lmax+1),4*means_dp[3]*M**(-alpha*torch.arange(0,Lmax-1)),'k:',label=f'Ref. line $\\alpha={round(alpha,2)}$',base=2)

    
#Label variance plot
axis_list[0].set_xlabel('$\ell$')
axis_list[0].set_ylabel(f'var')
axis_list[0].legend(framealpha=0.6, frameon=True)
axis_list[0].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
#Add estimated beta
s='$\\beta$ = {}'.format(round(beta,2))
# t = axis_list[0].annotate(s, (.6,.5), xycoords='axes fraction',fontsize=markersize,bbox=dict(ec='None',facecolor='None',lw=2))

#Label means plot
axis_list[1].set_xlabel('$\ell$')
axis_list[1].set_ylabel(f'mean')
axis_list[1].legend(framealpha=0.6, frameon=True)
axis_list[1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
#Add estimated alpha
s='$\\alpha$ = {}'.format(round(alpha,2))
# t = axis_list[1].annotate(s, (.4,.5), xycoords='axes fraction',fontsize=markersize,bbox=dict(ec='None',facecolor='None',lw=2))


plt.figure(fig2)
fig2.tight_layout(pad=.1)
plt.savefig(os.path.join(expdir,f'MeanImages_{switcher}.pdf'),bbox_inches='tight',format='pdf')

#Label number of levels plot
axis_list[2].set_xlabel('$\ell$')
axis_list[2].set_ylabel('$N_{\ell}$')
xa=axis_list[2].xaxis
xa.set_major_locator(ticker.MaxNLocator(integer=True))
(lines,labels)=axis_list[2].get_legend_handles_labels()
ncol=1
leg = Legend(axis_list[2], lines, labels, ncol=ncol, title='$\\varepsilon$',
             frameon=True, framealpha=0.6)
# leg._legend_box.align = "centr"
axis_list[2].add_artist(leg)
    
#Label and plot complexity plot
indices=np.flip(np.argsort(acc))
sortcost_mc=cost_mc[indices]
sortcost_mlmc=cost_mlmc[indices]
# sortacc=np.sqrt(realvar+realbias)#[indices] 
sortacc=acc[indices]

# asymptotic=sortacc**(beta/alpha)
# c0=(sortcost_mlmc/sortcost_mc)[0]/asymptotic[0]

mcacc=np.log(np.array(acc))
mccost=np.log(np.array(cost_mc))
coeffs=np.polynomial.polynomial.Polynomial.fit(mccost, mcacc, 2,domain=[min(mcacc),max(mcacc)])
fitted=np.exp(np.vander(np.log(cost_mlmc),3,increasing=True)@(coeffs.convert().coef))

# sV0=np.sqrt(V_p[Lmin])
# Ls=Lmin+np.array(lseq)[indices]
# start=min(Ls)
# end=max(Ls)
# cumsum2=np.cumsum(np.sqrt(V_dp[start:end+1]*M**(Ls)*(1.+1./M)))
# s=torch.sum(np.sqrt(V_dp[Lmin+1:start]*M**(np.arange(Lmin+1,start))*(1.+1./M)))
# factor=(sV0*M**(Lmin/2)+s+cumsum2)**2
# fitted=np.array(sortacc)*np.sqrt(sortcost_mlmc)/np.sqrt(2*V_p[Ls]*M**Ls)
fitted=np.array(sortacc)*np.sqrt(sortcost_mlmc)/np.sqrt(sortacc**2*sortcost_mc)

axis_list[3].loglog(sortcost_mlmc,fitted**2,'k-',marker='x',markersize=markersize,
             markerfacecolor="None",markeredgecolor='k', markeredgewidth=1,base=10)

# axis_list[3].loglog(theory_mlmc,theory_epsilon,'k-',marker='x',markersize=markersize,
#                 markerfacecolor="None",markeredgecolor='k', markeredgewidth=1,base=10)
# axis_list[3].loglog(sortcost_mc,sortacc,'k-',marker='None',markersize=markersize,
#              markerfacecolor="None",markeredgecolor='b-', markeredgewidth=1,base=2,label='MC')
# axis_list[3].loglog(theory_mc,theory_epsilon,'b-',marker='x',markersize=markersize,
#              markerfacecolor="None",markeredgecolor='b', markeredgewidth=1,base=10)

# axis_list[3].loglog(PSNR(sortacc),c0*asymptotic,'k--',marker='s',markersize=markersize,
#                 markerfacecolor="None",markeredgecolor='k', markeredgewidth=1,label='Asymptotic',base=2)
axis_list[3].set_ylabel('$\\varepsilon^2_{\\text{MLMC}}/\\varepsilon^2_{\\text{MC}}$')
axis_list[3].set_xlabel('Total NFEs')
axis_list[3].legend()
if switcher.lower()=='inpainting':
    axis_list[3].set_yticks([0.1,.2,0.3,0.4,0.5, 0.6])
    axis_list[3].set_xticks([10**4, 10**5, 10**6])
elif switcher.lower()=='denoising':
    axis_list[3].set_yticks([0.15,.2,0.3,0.4,0.6,0.8])
    axis_list[3].set_xticks([10**3,10**4, 10**5, 10**6])
else: #superresolution
    axis_list[3].set_yticks([0.1, 0.2,0.3,0.4,0.6,0.8,1])
    axis_list[3].set_xticks([10**4,10**5,10**6,10**7])
axis_list[3].get_yaxis().set_major_formatter(ticker.ScalarFormatter())


# axis_list[3].legend(frameon=True,framealpha=0.6)
# axis_list[3].set_ylim([2**-10,2**-6])
# axis_list[3].set_ylim([2.**-2.5,2.**.5])
#Add title and space out subplots
# fig.suptitle(label)
fig.tight_layout(rect=[0, 0.01, 1, 0.99],h_pad=1,w_pad=1,pad=.7)

fig.savefig(os.path.join(expdir,f'GilesPlot_{switcher}.pdf'), format='pdf', bbox_inches='tight')

files = np.array([os.path.join(expdir,f) for f in os.listdir(expdir) if f.startswith('level') and not f.endswith('pdf')])
orderer=np.array([int(f.split('_')[-1]) for f in files])
indices=np.argsort(orderer)
files=files[indices]
nrow = 2
ncol = len(files)

fig = plt.figure(figsize=(ncol+1, nrow+1)) 

from matplotlib import gridspec
gs = gridspec.GridSpec(nrow, ncol,
         wspace=0.0, hspace=0.0, 
         top=1.-0.5/(nrow+1), bottom=0.5/(nrow+1), 
         left=0.5/(ncol+1), right=1-0.5/(ncol+1))
plt.rc('axes', titlesize=15)     # fontsize of the axes title
plt.rc('axes', labelsize=15)     # fontsize of the axes title

for i,f in enumerate(files):
    l=int(f.split('_')[-1])
    stf='steps' if i==0 else ''
    stc='step' if i==0 else ''
    with np.load(os.path.join(f,'samples_f.npz')) as data:
        num=data['samplesf'].shape[0]
        r=np.random.randint(low=0,high=num)
        ax= plt.subplot(gs[0,i])
        ax.imshow(data['samplesf'][r])
        ax.set_title(f'{M**l} '+stf)
        ax.tick_params(
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False, 
            left=False,# ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off
        ax.set_yticklabels(())

    with np.load(os.path.join(f,'samples_c.npz')) as data:
        ax= plt.subplot(gs[1,i])
        # ax.set_title(f'{M**(l-1)} '+stc)
        ax.imshow(data['samplesc'][r])
        ax.tick_params(
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,
            left=False,# ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off
        ax.set_yticklabels(())

ax= plt.subplot(gs[1,0])
ax.set_ylabel('coarse')
ax = plt.subplot(gs[0,0])
ax.set_ylabel('fine')
plt.savefig(os.path.join(expdir,f'SampleLevels_{switcher}.pdf'),format='pdf',bbox_inches='tight')
        