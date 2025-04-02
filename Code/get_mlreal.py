import numpy as np
import mlreal_functions as mlr
import torch
from scipy.ndimage import gaussian_filter
from scipy.signal import hilbert, chirp
import random

window = np.zeros((500,1,1000,31))
for i in range(31):
    if i<=2:
        window[:,:,300+i*10:600+i*10,i] = 1
        window[:,:,:,i] = gaussian_filter(window[:,:,:,i],sigma=20)
    elif i>2:
        window[:,:,500+i*10:,i] = 1
        window[:,:,:,i] = gaussian_filter(window[:,:,:,i],sigma=20)

np.save('./window.npy',np.float32(window))



for i in range(0,7):
    print(i)
    fd = np.load('./field_data_fk_filtered_all_500_'+str(i)+'_avg_mask.npy')

#    syn = np.load('../field_syn_data_bandpass_trace_norm_500_1_1000_31/syn_catalog_500_'+str(i)+'.npy')
    syn = np.load('../field_syn_data_bandpass_trace_norm_500_1_1000_31/syn_data_500_'+str(i)+'.npy')

    print('fd/syn:',fd.max(),fd.min(),syn.max(),syn.min())
###### env #####
    syn_ref = mlr.get_ref_cor(torch.Tensor(syn),0)
    syn_ref = syn_ref.numpy() # size(500,1,1000,31)

#    syn_ref_reverse = np.zeros((500,1,1000,31))
#    syn_ref_reverse[:,0,:400,:] = syn_ref[:,0,600:,:] 
#    syn_ref_reverse[:,0,400:,:] = syn_ref[:,0,:600,:] 


    for iclip in range(500):
        for itrace in range(31):
            syn_ref[iclip,0,:,itrace] = syn_ref[iclip,0,:,itrace] / (np.abs(syn_ref[iclip,0,:,itrace]).max()+1e-4)
    print('syn ref:',syn_ref.max(),syn_ref.min())
    envelop = np.abs(hilbert(syn_ref.transpose(2,1,0,3)).transpose(2,1,0,3))
    #envelop_filter = np.zeros((500,1,1000,31))
    for iclip in range(500):
        for itrace in range(31):
            for itime in range(1000):
                if envelop[iclip,0,itime,itrace] < 0.1:
                    envelop[iclip,0,itime,itrace] = 0
                if envelop[iclip,0,itime,itrace] >= 0.1:
                    envelop[iclip,0,itime,itrace] = 1
    
    envelop = envelop*window
    envelop_s = np.zeros((500,1,1000,31))
    tmp = np.zeros((1,1,1000,1))
    for idata in range(500):
        for itrace in range(31): 
            tmp = envelop[idata,0,:,itrace]
            envelop_s[idata,0,:,itrace] = gaussian_filter(tmp,sigma=15)

#gaussian_filter(envelop.transpose(2,1,0,3),sigma=2)
#    print(envelop_s.shape)
#    envelop_s = envelop_s.transpose(2,1,0,3)
#    print(envelop_s.shape)
#    envelop = envelop*window

#    np.save('./test_field_'+str(i)+'_avg_mask_env.npy',np.float32(fd*envelop_s))
#    np.save('./test_env'+str(i)+'.npy',np.float32(envelop_s))
#    np.save('./test_syn_ref_reverse'+str(i)+'.npy',np.float32(syn_ref))
#    print(i,'before mlr:',fd.max(),fd.min(),syn.max(),syn.min())
    fd_conv, syn_conv = mlr.get_mlreal(torch.Tensor(fd*envelop_s), torch.Tensor(syn_ref*envelop_s), normalize=True)
    print(i,'after mlr:',fd_conv.numpy().max(),fd_conv.numpy().min(),syn_conv.numpy().max(),syn_conv.numpy().min())
#    syn_conv_reverse = torch.Tensor(np.zeros((500,1,1000,31)))
#    syn_conv_reverse[:,0,:400,:] = syn_conv[:,0,600:,:] 
#    syn_conv_reverse[:,0,400:,:] = syn_conv[:,0,:600,:] 

#    fd_conv = fd_conv * window
#    syn_conv_reverse = syn_conv_reverse * window
    random_map = np.random.rand(500,1,1000,31)
    random_map = (random_map - 0.5) * 0.001
    fd_conv = fd_conv * envelop_s + random_map
    syn_conv = syn_conv * envelop_s + random_map
    for idata in range(500):
        for itrace in range(31):
            fd_conv[idata,0,:,itrace] = fd_conv[idata,0,:,itrace] / (torch.max(torch.abs(fd_conv[idata,0,:,itrace]))+1e-4)
            syn_conv[idata,0,:,itrace] = syn_conv[idata,0,:,itrace] / (torch.max(torch.abs(syn_conv[idata,0,:,itrace]))+1e-4)

    print(i,'after mlr and filter and norm:',fd_conv.numpy().max(),fd_conv.numpy().min(),syn_conv.numpy().max(),syn_conv.numpy().min())
    #(500,1,1000,31)

#    np.save('./field_data_fk_filtered_all_mlr_500_'+str(i)+'_avg_mask.npy',np.float32(fd_conv.numpy()))
#    np.save('./syn_catalog_fk_filtered_all_mlr_500_'+str(i)+'_avg_mask_normal.npy',np.float32(syn_conv.numpy()))
    np.save('./syn_data_fk_filtered_all_mlr_500_'+str(i+28)+'_avg_mask_normal.npy',np.float32(syn_conv.numpy()))


