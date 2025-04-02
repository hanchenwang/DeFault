import numpy as np
import torch
import scipy
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def normalize_by_str_mean(in_data):
    in_std = in_data.std()
    in_mean = in_data.mean()
    return (in_data-in_mean)/in_std

def normalize_by_min_max(in_data):
    min = in_data.min()
    max = in_data.max()
    return ((in_data-min)/(max-min)-0.5)*2

def get_ref_cor(in_data,tr_no):
    
#    in_data_np = in_data.cpu().detach().numpy()
#    ref_trace=np.expand_dims(in_data_np[:,:,:,tr_no],axis=-1)
    ref_trace = in_data[:,:,:,tr_no].unsqueeze(dim=3)
    batch_size = in_data.shape[0]
    trace_size = in_data.shape[3]
    time_size = in_data.shape[2]
#    dup_ref = torch.Tensor(torch.repeat(torch.repeat(ref_trace, trace_size, axis=-1),batch_size,axis=-1))
    dup_ref = ref_trace.repeat(1,1,1,trace_size)
#dup_ref = dup_ref.view(time_size,1,trace_size,batch_size).transpose(3,1,0,2)

    data_fft = torch.fft.rfft(in_data,axis=2)
    dup_ref_fft = torch.fft.rfft(dup_ref,axis=2)
    tmp = torch.fft.irfft(data_fft * torch.conj(dup_ref_fft),axis=2)
    out_data = torch.zeros(batch_size,1,time_size,trace_size)
    out_data[:,:,:int(time_size/2),:] = tmp[:,:,int(time_size/2):,:]
    out_data[:,:,int(time_size/2):,:] = tmp[:,:,:int(time_size/2),:]

    return out_data

def get_auto_cor(in_data):
    

    data_fft = torch.fft.rfft(in_data,axis=2)
    tmp = torch.fft.irfft(data_fft * torch.conj(data_fft),axis=2)
    batch_size = in_data.shape[0]
    trace_size = in_data.shape[3]
    time_size = in_data.shape[2]

    out_data = torch.zeros(batch_size,1,time_size,trace_size)
    out_data[:,:,:int(time_size/2),:] = tmp[:,:,int(time_size/2):,:]
    out_data[:,:,int(time_size/2):,:] = tmp[:,:,:int(time_size/2),:]

    return out_data

def get_conv(ref_cor, auto_cor):
    
    #out_data = torch.zeros(batch_size,1,time_size,trace_size)
    ref_cor_fft = torch.fft.rfft(ref_cor,axis=2).to('cuda')
    auto_cor_fft = torch.fft.rfft(auto_cor,axis=2).to('cuda')
    batch_size = ref_cor.shape[0]
    trace_size = ref_cor.shape[3]
    time_size = ref_cor.shape[2]

    out_data = torch.zeros(batch_size,1,time_size,trace_size)

    tmp = torch.fft.irfft(ref_cor_fft * auto_cor_fft,axis=2)
    out_data[:,:,:int(time_size/2),:] = tmp[:,:,int(time_size/2):,:]
    out_data[:,:,int(time_size/2):,:] = tmp[:,:,:int(time_size/2),:]

    return out_data

def get_mlreal(fd, sd, normalize=False):

    sd_ref_cor = sd#get_ref_cor(sd,0)
    if normalize:
        sd_ref_cor = normalize_by_std_mean(sd_ref_cor)
    fd_ref_cor = fd#get_ref_cor(fd,0)
    if normalize:
        fd_ref_cor = normalize_by_std_mean(fd_ref_cor)

    fd_ac = get_auto_cor(fd)
    if normalize:    
        fd_ac = normalize_by_std_mean(fd_ac)
    sd_ac = get_auto_cor(sd)
    if normalize:    
        sd_ac = normalize_by_std_mean(sd_ac)

    sd_conv = get_conv(sd_ref_cor,fd_ac)
    if normalize:
        sd_conv = normalize_by_std_mean(sd_conv)
    fd_conv = get_conv(fd_ref_cor,sd_ac)
    if normalize:    
        fd_conv = normalize_by_std_mean(fd_conv)

    return fd_conv,sd_conv

#def get_mlreal_from_data(fd, sd):







