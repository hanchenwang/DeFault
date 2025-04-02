import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import time
import pandas as pd

class FWIForward(nn.Module):
    def __init__(self, v_denorm_func, s_norm_func, sample_ratio, device, dataset='25', mode=1):
        super(FWIForward, self).__init__()
        self.v_denorm_func = v_denorm_func
        self.s_norm_func = s_norm_func
        self.device = device
        self.sample_ratio = sample_ratio
        self.mode = mode
        if dataset == 'mam_low' or dataset == 'mam_high':
            self.args = {
                'nbc': 100,
                'dx': 10, 
                'nt': 1000,
                'dt': 1e-3,
                'f': 20,
                'sx': np.linspace(0, 69, num=5) * 10,
                'sz': 10,
                'gx': np.linspace(0, 69, num=70) * 10,
                'gz': 10
            }
        elif dataset == 'salt':
            self.args = {
                'nbc': 60,
                'dx': 10, 
                'nt': 2000,
                'dt': 1e-3,
                'f': 12,
                'sx': np.arange(0., 2700., 300.),
                'sz': 10,
                'gx': np.linspace(0, 300, num=301) * 10,
                'gz': 10
            }
        elif dataset == 'salt_down':
            self.args = {
                'nbc': 40,
                'dx': 10, 
                'nt': 600,
                'dt': 1e-3,
                'f': 20,
                'sx': np.linspace(0, 59, num=5) * 10,
                'sz': 10,
                'gx': np.linspace(0, 59, num=60) * 10,
                'gz': 10
            }
        elif dataset == 'mam_original':
            self.args = {
                'nbc': 100,
                'dx': 5, 
                'nt': 5000,
                'dt': 2e-4,
                'f': 15,
                'sx': np.linspace(0, 199, num=10) * 5,
                'sz': 10,
                'gx': np.linspace(0, 199, num=200) * 5,
                'gz': 10
            }
        elif dataset == 'SJTL':
            self.args = {
                'nbc': 120,
                'dx': 10, 
                'nt': 1000,
                'dt': 1e-3,
                'f': 15,
                'sx': np.linspace(0, 69, num=5) * 10,
                'sz': 10,
                'gx': np.linspace(0, 69, num=70) * 10,
                'gz': 10
            }
        else:
            self.args = {
                'nbc': 30,
                'dx': 10, 
                'nt': 1000,
                'dt': 8e-4,
                'f': 25,
                'sx': np.linspace(0, 99, num=10) * 10,
                'sz': 10,
                'gx': np.linspace(0, 99, num=100) * 10,
                # 'gx': np.linspace(0, 69, num=70) * 15,
                'gz': 10
            }
        self.pad = nn.ReplicationPad2d(self.args['nbc'])

    #source func
    def ricker(self, f, dt, nt):
        nw = 2.2/f/dt
        nw = 2*np.floor(nw/2)+1
        nc = np.floor(nw/2)
        k = np.arange(nw)
        
        alpha = (nc-k)*f*dt*np.pi
        beta = alpha ** 2
        w0 = (1-beta*2)*np.exp(-beta)
        w = np.zeros(nt)
        w[:len(w0)] = w0
        return w
    
    # absorbing boundary condition
    def get_Abc(self, vp, nbc, dx):
        dimrange = 1.0*torch.unsqueeze(torch.arange(nbc, device=self.device), dim=-1)
        damp = torch.zeros_like(vp, device=self.device, requires_grad=False) #
        
        velmin,_ = torch.min(vp.view(vp.shape[0],-1), dim=-1, keepdim=False)

        nzbc, nxbc = vp.shape[2], vp.shape[3]
        nz = nzbc-2*nbc
        nx = nxbc-2*nbc
        a = (nbc-1)*dx
        
        kappa = 3.0 * velmin * np.log(1e7) / (2.0 * a)
        kappa = torch.unsqueeze(kappa,dim=0)
        kappa = torch.repeat_interleave(kappa, nbc, dim=0)
        
        damp1d = kappa * (dimrange*dx/a) ** 2
        damp1d = damp1d.permute(1,0).unsqueeze(1)
        
        damp[:,:,:nbc, :] = torch.repeat_interleave(torch.flip(damp1d,dims=[-1]).unsqueeze(-1), vp.shape[-1], dim=-1) 
        damp[:,:,-nbc:,:] = torch.repeat_interleave(damp1d.unsqueeze(-1), vp.shape[-1], dim=-1) 
        damp[:,:,:, :nbc] = torch.repeat_interleave(torch.flip(damp1d,dims=[-1]).unsqueeze(-2), vp.shape[-2], dim=-2) 
        damp[:,:,:,-nbc:] = torch.repeat_interleave(damp1d.unsqueeze(-2), vp.shape[-2], dim=-2) 
        return damp

    # adjust source/receiver location
    def adj_sr(self, sx,sz,gx,gz,dx,nbc):
        isx = np.around(sx/dx)+nbc
        isz = np.around(sz/dx)+nbc
        
        igx = np.around(gx/dx)+nbc
        igz = np.around(gz/dx)+nbc
        return isx.astype('int'),int(isz),igx.astype('int'),int(igz)


    def FWM(self, v, nbc=120, dx=10, nt=1000, dt=1e-3, f=15,
            sx=np.linspace(0, 69, num = 5) * 10, sz=10,
            gx=np.linspace(0, 69, num = 70) * 10, gz=10):
        '''
        # constant setting for forward modeling
        grids = 100
        # time interval
        nt = 1000
        # grid
        dx = 15
        # bc
        nbc = 30
        # grid t
        dt = 1e-3
        # src positions
        sz = 10
        sx = np.linspace(0, grids-1, num = 10)*dx
        ns = len(sx)
        # receivers positions
        gx = np.linspace(0, grids-1, num = grids)*dx
        gz = 10
        '''
        # print(v.shape)
        src = self.ricker(f, dt, nt)
        alpha = (v*dt/dx) ** 2

        abc = self.get_Abc(v, nbc, dx)
        kappa = abc*dt

        c1 = -2.5
        c2 = 4.0/3.0
        c3 = -1.0/12.0 

        temp1 = 2+2*c1*alpha-kappa
        temp2 = 1-kappa
        beta_dt = (v*dt) ** 2
        
        ns = len(sx)
#        print('ns:',ns)
        isx,isz,igx,igz = self.adj_sr(sx,sz,gx,gz,dx,nbc)
        seis = []
        p1 = torch.zeros((v.shape[0], ns, v.shape[2], v.shape[3]), device=self.device, requires_grad=True)
        p0 = torch.zeros((v.shape[0], ns, v.shape[2], v.shape[3]), device=self.device, requires_grad=True)
        p  = torch.zeros((v.shape[0], ns, v.shape[2], v.shape[3]), device=self.device, requires_grad=True)
        # start_time = time.time()
        for i in range(nt):
            # if i % 1000 == 0:
            #     log_time = time.time() - start_time
            #     print('Step: ', i)
            #     print('Time: ', str(datetime.timedelta(seconds=int(log_time))))
            p = (temp1*p1 - temp2*p0 + alpha * 
                (c2*(torch.roll(p1, 1, dims = -2) + torch.roll(p1, -1, dims = -2) + torch.roll(p1, 1, dims = -1)+ torch.roll(p1, -1, dims = -1))
                +c3*(torch.roll(p1, 2, dims = -2) + torch.roll(p1, -2, dims = -2) + torch.roll(p1, 2, dims = -1)+ torch.roll(p1, -2, dims = -1))
                ))
            for loc in range(ns):
                p[:,loc,isz,isx[loc]] = p[:,loc,isz,isx[loc]] + beta_dt[:,0,isz,isx[loc]] * src[i]
            if i % self.sample_ratio == 0:
#            if i % 10 == 0:
                seis.append(torch.unsqueeze(p[:, :, [igz]*len(igx), igx], dim=2))
            p0=p1
            p1=p
        return torch.cat(seis, dim=2)

    # forbidden implementation
    def FWM2(self, v, nbc=15, dx=30, nt=500, dt=2e-3, f=15,
        sx=np.linspace(0, 49, num = 5) * 15, sz=10,
        gx=np.linspace(0, 49, num = 50) * 15, gz=10):

        src = self.ricker(f, dt, nt)
        alpha = (v*dt/dx) ** 2

        abc = self.get_Abc(v, nbc, dx)
        kappa = abc*dt

        c1 = -2.5
        c2 = 4.0/3.0
        c3 = -1.0/12.0 

        temp1 = 2+2*c1*alpha-kappa
        temp2 = 1-kappa
        beta_dt = (v*dt) ** 2
        
        ns = len(sx)
        isx,isz,igx,igz = self.adj_sr(sx,sz,gx,gz,dx,nbc)

        def update(p0, p1):
            p = (temp1 * p1 - temp2 * p0 + alpha * 
                (c2 * (torch.roll(p1, 1, dims=-2) 
                    + torch.roll(p1, -1, dims=-2) 
                    + torch.roll(p1, 1, dims=-1) 
                    + torch.roll(p1, -1, dims=-1)) +
                c3 * (torch.roll(p1, 2, dims=-2) 
                    + torch.roll(p1, -2, dims=-2) 
                    + torch.roll(p1, 2, dims=-1) 
                    + torch.roll(p1, -2, dims=-1))
                ))
            for loc in range(ns):
                p[:,loc,isz,isx[loc]] = p[:,loc,isz,isx[loc]] + beta_dt[:,0,isz,isx[loc]] * src[i]
            p0 = p1
            p1 = p
            return p, p0, p1

        seis = torch.zeros((v.shape[0], ns, nt // self.sample_ratio, v.shape[-1]-2*nbc), device=self.device, requires_grad=False)
        p1 = torch.zeros((v.shape[0], ns, v.shape[2], v.shape[3]), device=self.device, requires_grad=False)
        p0 = torch.zeros((v.shape[0], ns, v.shape[2], v.shape[3]), device=self.device, requires_grad=False)

        for i in range(nt):
            p, p0, p1 = update(p0, p1)
            if i % self.sample_ratio == 0:
                seis[:, :, i // self.sample_ratio, :] = p[:, :, [igz]*len(igx), igx]
        return seis

    def forward(self, v):
        v_denorm = self.v_denorm_func(v)
        v_pad = self.pad(v_denorm)
        # v_pad = self.pad(v)
        if self.mode == 1:
            s = self.FWM(v_pad, **self.args)
        else:
            s = self.FWM2(v_pad, **self.args)
        # return s
        s_norm = self.s_norm_func(s)
#        np.save('/vast/home/hwang/Desktop/repo/UPFWI_modified/src/models_cycle/sj_t3/6000_train/snorm.npy',s_norm)
        return s_norm
