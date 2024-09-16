import logging
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import core.metrics as Metrics
import os
import model.networks as networks
from .base_model import BaseModel
logger = logging.getLogger('base')

def mom2norm(sqsums,MASK):
    #sqsums should have shape L,C,H,W
    s=MASK.sum()
    return torch.sum(torch.flatten(sqsums*MASK[None,...], start_dim=1, end_dim=-1),dim=-1)/s

def imagenorm(img,MASK):
    s=MASK.sum()
    n=torch.linalg.norm(torch.flatten(img*MASK[None,...], start_dim=1, end_dim=-1),dim=-1) #flattens non-batch dims and calculates norm
    n/=np.sqrt(s)
    return n

class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)
        
        self.M=2
        self.Lmax=11
        self.Lmin=4
        self.mlmc_batch_size=80
        self.accsplit=np.sqrt(.5)
        self.N0=10
        self.eval_dir=opt['path']['experiments_root']
        self.image_size=self.opt['model']['diffusion']['image_size']
        self.channels=self.opt['model']['diffusion']['channels']
        try:
            self.MASK=(opt['datasets']['MASK']).type(torch.float32)
        except:
            print('No MASK selected for MLMC. Defaulting to identity')
        print('No MASK selected for MLMC. Defaulting to identity')
        self.MASK=torch.ones((self.channels,self.image_size,self.image_size))
            
        self.squaresummer = lambda p,axis: torch.sum(p**2,axis=axis)
        self.summer = lambda p,axis: torch.sum(p,axis=axis)
        self.imagenorm = lambda img: imagenorm(img,self.MASK)
        self.mom2norm =lambda img: mom2norm(img,self.MASK)
        if opt['payoff']=='mean':
            self.Lmin=4
            self.alpha0=.69
            self.beta0=1.05
            print("mean payoff selected.")
            self.payoff = lambda samples: torch.clip(samples,max=1.,min=-1.) #default to identity payoff
        elif opt['payoff']=='second_moment':
            print("second_moment payoff selected.")
            self.payoff = lambda samples: torch.clip(samples,max=1.,min=-1.)**2 #variance/second moment payoff
        elif opt['payoff']=='proj_mean':
            self.Lmin=3
            self.alpha0=.85
            self.beta0=1.7
            with open(os.path.join(os.getcwd(),'results/sr_sr3_16_128_MC','PCA_0.pt'),'rb') as f:
                pca0=torch.load(f).to('cuda')
            self.payoff = lambda samples: torch.clip(samples,max=1.,min=-1.)
            self.squaresummer = lambda p,axis:torch.sum(torch.sum(torch.flatten(p[:,:,50:70,35:95],start_dim=1,end_dim=-1)*pca0[None,...],dim=-1)**2,axis=axis)
           # self.summer= lambda p,axis:torch.sum(torch.sum(torch.flatten(p[:,:,50:70,35:95],start_dim=1,end_dim=-1)*pca0[None,...],dim=-1),axis=axis)
            self.imagenorm = lambda img :torch.abs(torch.sum(torch.flatten(img[:,:,50:70,35:95],start_dim=1,end_dim=-1)*pca0.to(img.device)[None,...],dim=-1))
            self.mom2norm = lambda img : img
        else:
            print("opt['payoff'] not recognised. Defaulting to mean calculation.")
            self.payoff = lambda samples: torch.clip(samples,max=1.,min=-1.) #default to identity payoff
        kwargs={'M':self.M,'Lmax':self.Lmax,'Lmin':self.Lmin,
                'mlmc_batch_size':self.mlmc_batch_size,'N0':self.N0,
                'eval_dir':self.eval_dir,'payoff':self.payoff}
        # define network and load pretrained model
        self.netG = self.set_device(networks.define_G(opt,**kwargs))
        #self.netG = networks.define_G(opt,**kwargs)
        self.schedule_phase = None

        # set loss and load resume state
        self.set_loss()
        self.set_new_noise_schedule(
            opt['model']['beta_schedule']['train'], schedule_phase='train')
        if self.opt['phase'] == 'train':
            self.netG.train()
            # find the parameters to optimize
            if opt['model']['finetune_norm']:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                optim_params = list(self.netG.parameters())

            self.optG = torch.optim.Adam(
                optim_params, lr=opt['train']["optimizer"]["lr"])
            self.log_dict = OrderedDict()
        self.load_network()
        # self.print_network()

    def feed_data(self, data):
        self.data = self.set_device(data)

    def optimize_parameters(self):
        self.optG.zero_grad()
        l_pix = self.netG(self.data)
        # need to average in multi-gpu
        b, c, h, w = self.data['HR'].shape
        l_pix = l_pix.sum()/int(b*c*h*w)
        l_pix.backward()
        self.optG.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()

    def test(self, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.super_resolution(
                    self.data['SR'], continous)
            else:
                self.SR = self.netG.super_resolution(
                    self.data['SR'], continous)
        self.netG.train()
    
    def mc(self, Nl, MCL,continous=False):
        self.netG.eval()
        eval_dir=self.eval_dir
        this_sample_dir = os.path.join(eval_dir,'MCsamples')
        if not os.path.exists(this_sample_dir):
            os.mkdir(this_sample_dir)
        M=self.M
        l=MCL
        with open(os.path.join(this_sample_dir, "info_text.txt"),'w') as f:
            f.write(f'MC params:L={l}, Nsamples={Nl}, M={M}.')
        num_sampling_rounds = Nl // self.mlmc_batch_size + 1
        print(num_sampling_rounds)
        numrem=Nl % self.mlmc_batch_size
        for r in range(num_sampling_rounds):
            bs=numrem if r==num_sampling_rounds-1 else self.mlmc_batch_size
            with torch.no_grad():
                if isinstance(self.netG, nn.DataParallel):
                    Xf= self.netG.module.mcsample(self.data['SR'], bs, continous)
                else:
                    Xf = self.netG.mcsample(self.data['SR'], bs, continous)
    
            # Directory to save samples.
            with open(os.path.join(this_sample_dir, f"samples_{self.opt['gpu_ids'][0]}_{r}.npz"), "wb") as fout:
                np.savez_compressed(fout, samples=Xf.cpu().numpy())
        
        self.netG.train()
        return None

    
    def Giles_plot(self,acc):
        self.netG.eval()
        #self.netG.denoise_fn.eval()
        #self.netG.denoise_fn.to(self.device)
        #self.netG.denoise_fn=torch.nn.DataParallel(self.netG.denoise_fn)
        #Set mlmc params
        M=self.M
        N0=self.N0
        Lmax=self.Lmax
        eval_dir = self.eval_dir
        Nsamples=1000
        condition_x=self.data['SR'].to(self.device)
        Lmin=self.Lmin
        
        # Directory to save means and norms                                                                                               
        this_sample_dir = os.path.join(eval_dir, f"VarMean_L_{Lmax}_Nsamples_{Nsamples}")
        if not os.path.exists(this_sample_dir):
            #Variance and mean samples
            sums,sqsums=self.mlmclooper(condition_x,l=1,Nl=1,Lmin=0) #dummy run to get sum shapes 
            sums=torch.zeros((Lmax+1,*sums.shape))
            sqsums=torch.zeros((Lmax+1,*sqsums.shape))
            os.mkdir(this_sample_dir)
            print(f'Proceeding to calculate variance and means with {Nsamples} estimator samples')
            for i,l in enumerate(range(Lmax+1)):
                print(f'l={l}')
                sums[i],sqsums[i] = self.mlmclooper(condition_x,Nsamples,l)

                
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
            Lmincond=V_dp[1:]<(np.sqrt(M)-1.)**2*V_p[-1]/(1+M) #index of optimal lmin
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
            with open(os.path.join(this_sample_dir, "mlmc_info.txt"),'w') as f:
                f.write(f'MLMC params: N0={N0}, Lmax={Lmax}, Lmin={cutoff}, Nsamples={Nsamples}, M={M}.\n')
                f.write(f'Estimated alpha={alpha}\n Estimated beta={beta}')
            with open(os.path.join(this_sample_dir, "alphabeta.pt"), "wb") as fout:
                torch.save(torch.tensor([alpha,beta,cutoff]),fout)
        try:        
            with open(os.path.join(this_sample_dir, "alphabeta.pt"),'rb') as f:
                temp=torch.load(f)
                alpha=float(temp[0].item())
                beta=float(temp[1].item())
                Lmin=int(temp[2].item())
        except:
            alpha=self.alpha0
            beta=self.beta0
            Lmin=self.Lmin
        print(f'alpha={alpha},beta={beta}')
        #Do the calculations and simulations for num levels and complexity plot
        for i in range(len(acc)):
            e=acc[i]
            print(f'Performing mlmc for accuracy={e}')
            sums,sqsums,N=self.mlmc(e,condition_x,alpha_0=alpha,beta_0=beta) #sums=[dX,Xf,Xc], sqsums=[||dX||^2,||Xf||^2,||Xc||^2]
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
            dividerN2=N.clone() #add axes to N to broadcast correctly on division                                                        
            for i in range(len(sqsums.shape[1:])):
                dividerN2.unsqueeze_(-1)
                
            this_sample_dir = os.path.join(eval_dir, 'Experiment',f"M_{M}_accuracy_{e}")
            if not os.path.exists(os.path.join(eval_dir, 'Experiment')):
                os.mkdir(os.path.join(eval_dir, 'Experiment'))
            
            if not os.path.exists(this_sample_dir):
                os.mkdir(this_sample_dir)        
            with open(os.path.join(this_sample_dir, "averages.pt"), "wb") as fout:
                torch.save(sums/dividerN,fout)
            with open(os.path.join(this_sample_dir, "sqaverages.pt"), "wb") as fout:
                torch.save(sqsums/dividerN2,fout) #sums has shape (L,4,C,H,W) if img (L,4,2048) if activations
            with open(os.path.join(this_sample_dir, "N.pt"), "wb") as fout:
                torch.save(N,fout)
            with open(os.path.join(this_sample_dir, "costs.npz"), "wb") as fout:
               np.savez_compressed(fout,costmlmc=np.array(cost_mlmc),costmc=np.array(cost_mc))

            meanimg=torch.sum(sums[:,0]/dividerN[:,0,...],axis=0)#cut off one dummy axis
            meanimg=Metrics.tensor2img(meanimg,min_max=(-1.,1)) #default min_max=(-1., 1.)
            # Write samples to disk or Google Cloud Storage
            with open(os.path.join(this_sample_dir, "meanpayoff.npz"), "wb") as fout:
                np.savez_compressed(fout, meanpayoff=meanimg)
        self.netG.train()

        return None

    def mlmc(self,accuracy,x_in,alpha_0=-1,beta_0=-1):
        accsplit=self.accsplit
        #Orders of convergence
        alpha=max(0,alpha_0)
        beta=max(0,beta_0)
        M=self.M
        N0=self.N0
        Lmax=self.Lmax
        Lmin=self.Lmin
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
                    tempsums,tempsqsums=self.mlmclooper(condition_x=x_in,Nl=int(num),l=l,Lmin=Lmin) #Call function which gives sums
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
            X[:,0]=torch.arange(1.,mylen)
            a = torch.linalg.lstsq(X,torch.log(Yl[1:]))[0]
            alpha_ = max(-a[0]/np.log(M),0.)
            b = torch.linalg.lstsq(X,torch.log(V[1:]))[0]
            beta_= -b[0]/np.log(M)
            if alpha_0==-1:
                alpha=alpha_
            if beta_0==-1:
                beta=beta_
                
            sqrt_V=torch.sqrt(V)
            Nl_new=torch.ceil(((accsplit*accuracy)**-2)*torch.sum(sqrt_V*sqrt_cost)*(sqrt_V/sqrt_cost)) #Estimate optimal number of samples/level
            dN=torch.clip(Nl_new-N,min=0) #Number of additional samples
            print(f'Asking for {dN} new samples for l=[{Lmin,L}]')
            print(f'sqrt_var = {torch.sum(2*V/N).sqrt()}, bias = {np.sqrt(2)*(Yl[-1])/(M**alpha-1.)}')
            if torch.sum(dN > 0.01*N).item() == 0: #Almost converged
                test=max(Yl[-2]/(M**alpha),Yl[-1]) if L-Lmin>1 else Yl[-1]
                if test>(M**alpha-1)*accuracy*np.sqrt(1-accsplit**2):
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
    
    def mlmclooper(self,condition_x,Nl,l,Lmin=0):
        eval_dir=self.eval_dir
        num_sampling_rounds = Nl // self.mlmc_batch_size + 1
        numrem=Nl % self.mlmc_batch_size
        for r in range(num_sampling_rounds):
            bs=numrem if r==num_sampling_rounds-1 else self.mlmc_batch_size
            if bs==0:
                break
            with torch.no_grad():
                if isinstance(self.netG, nn.DataParallel):
                    Xf,Xc=self.netG.module.mlmcsample(condition_x,bs,l) #should automatically use cuda
                else:
                    Xf,Xc=self.netG.mlmcsample(condition_x,bs,l) #should automatically use cuda
            fine_payoff=self.payoff(Xf)
            coarse_payoff=self.payoff(Xc)
            sumXf=self.summer(fine_payoff,axis=0).to('cpu') #sum over batch size                                          
            sumXf2=self.squaresummer(fine_payoff,axis=0).to('cpu')
            sumXf3=torch.sum(self.imagenorm(fine_payoff.to('cpu'))**3,axis=0).to('cpu')*torch.ones_like(sumXf2)
            sumXf4=torch.sum(self.imagenorm(fine_payoff.to('cpu'))**4,axis=0).to('cpu')*torch.ones_like(sumXf2)
            if r==0:
                sums=torch.zeros((3,*sumXf.shape)) #skip batch_size
                sqsums=torch.zeros((4,*sumXf2.shape))
            if l==Lmin:
                sqsums+=torch.stack([sumXf2,sumXf2,sumXf3,sumXf4])
                sums+=torch.stack([sumXf,sumXf,torch.zeros_like(sumXf)])
            elif l<Lmin:
                raise ValueError("l must be at least Lmin")
            else:
                dX_l=fine_payoff-coarse_payoff #Image difference
                sumdX_l=self.summer(dX_l,axis=0).to('cpu') #sum over batch size
                sumdX_l2=self.squaresummer(dX_l,axis=0).to('cpu')
                sumXc=self.summer(coarse_payoff,axis=0).to('cpu')
                sumXc2=self.squaresummer(coarse_payoff,axis=0).to('cpu')
                sumdX_l3=torch.sum(self.imagenorm(dX_l.to('cpu'))**3,axis=0).to('cpu')*torch.ones_like(sumdX_l2)
                sumdX_l4=torch.sum(self.imagenorm(dX_l.to('cpu'))**4,axis=0).to('cpu')*torch.ones_like(sumdX_l2)
                sums+=torch.stack([sumdX_l,sumXf,sumXc])
                sqsums+=torch.stack([sumdX_l2,sumXf2,sumdX_l3,sumdX_l4])
    
        # Directory to save samples. Just to save an example sample for debugging
        if l>Lmin:
            this_sample_dir = os.path.join(eval_dir, f"level_{l}")
            if not os.path.exists(this_sample_dir):
                os.mkdir(this_sample_dir)
                samples_f=Metrics.tensor2img(Xf[0])[None,...]
                samples_c=Metrics.tensor2img(Xc[0])[None,...]
                for i in range(1,min(Xf.shape[0],20)):
                    samples_f=np.concatenate([samples_f,Metrics.tensor2img(Xf[i])[None,...]],axis=0)
                    samples_c=np.concatenate([samples_c,Metrics.tensor2img(Xc[i])[None,...]],axis=0)  
                with open(os.path.join(this_sample_dir, "samples_f.npz"), "wb") as fout:
                    np.savez_compressed(fout, samplesf=samples_f)
                with open(os.path.join(this_sample_dir, "samples_c.npz"), "wb") as fout:
                    np.savez_compressed(fout, samplesc=samples_c)
                                
        return sums,sqsums 
    
    
    def sample(self, batch_size=1, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.sample(batch_size, continous)
            else:
                self.SR = self.netG.sample(batch_size, continous)
        self.netG.train()

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train',MLMCsteps=None):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(
                    schedule_opt, self.device,MLMCsteps=MLMCsteps)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device,MLMCsteps=MLMCsteps)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_LR=True, sample=False):
        out_dict = OrderedDict()
        if sample:
            out_dict['SAM'] = self.SR.detach().float().cpu()
        else:
            out_dict['SR'] = self.SR.detach().float().cpu()
            out_dict['INF'] = self.data['SR'].detach().float().cpu()
            out_dict['HR'] = self.data['HR'].detach().float().cpu()
            if need_LR and 'LR' in self.data:
                out_dict['LR'] = self.data['LR'].detach().float().cpu()
            else:
                out_dict['LR'] = out_dict['INF']
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def save_network(self, epoch, iter_step):
        gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_gen.pth'.format(iter_step, epoch))
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        # gen
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step,
                     'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optG.state_dict()
        torch.save(opt_state, opt_path)

        logger.info(
            'Saved model in [{:s}] ...'.format(gen_path))

    def load_network(self):
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)
            # gen
            network = self.netG
            if isinstance(self.netG, nn.DataParallel):
                network = network.module
            network.load_state_dict(torch.load(
                gen_path), strict=(not self.opt['model']['finetune_norm']))
            # network.load_state_dict(torch.load(
            #     gen_path), strict=False)
            if self.opt['phase'] == 'train':
                # optimizer
                opt = torch.load(opt_path)
                self.optG.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']
