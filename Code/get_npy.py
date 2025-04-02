import numpy as np

'''
from scipy.ndimage import gaussian_filter

window = np.zeros((500,1,1000,31))
for i in range(31):
    if i<=2:
        window[:,:,300+i*10:600+i*10,i] = 1
        window[:,:,:,i] = gaussian_filter(window[:,:,:,i],sigma=50)
    elif i>2:
        window[:,:,400+i*10:700+i*10,i] = 1
        window[:,:,:,i] = gaussian_filter(window[:,:,:,i],sigma=50)
'''

for i in range(7):
    with open ('clip_all_filtered_'+str(i)+'_500_1000_31.rsf@','rb') as f1:
        tmp = np.fromfile(f1,np.single).reshape(1,31,1000,500).transpose(3,0,2,1)#(500,1,1000,31)

    np.save('./field_data_fk_filtered_all_500_'+str(i)+'_avg_mask.npy',np.float32(tmp))
    

