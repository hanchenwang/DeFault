import numpy as np 
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import random
import math

fd = np.load('/projects/piml_inversion/hwang/Illinois/src_chicoma/data/heat_maps/field_label_heat_map_random_intz_500_0.npy')
syn = np.load('/projects/piml_inversion/hwang/Illinois/src_chicoma/data/heat_maps/syn_label_heat_map_random_intz_500_0.npy')
print(fd.shape)

plt.figure()
plt.subplot(1,2,1)
plt.imshow(fd[1,:,:,30],cmap='gray')
plt.colorbar()
plt.subplot(1,2,2)
plt.imshow(syn[1,:,:,30],cmap='gray')
plt.colorbar()
plt.show()









exit()
def gaussian_xyz(x,y,z,lx,ly,lz,sigma):
    A = 1/(sigma*math.sqrt(2*(math.pi)))
    B = -(math.pow(x-lx,2)+math.pow(y-ly,2)+math.pow(z-lz,2))/(2*math.pow(sigma,2))
    return A*math.pow(math.e,B)


###### parameters #######
dx = 0.05
dy = 0.05
dz = 0.01
kx = 0.26
lx = 3.43
ky = -1.73
ly = 2.22
kz = 1.80
lz = 2.35
nx = int(round((lx-kx)/dx,0))
ny = int(round((ly-ky)/dy,0))
nz = int(round((lz-kz)/dz,0))
print('map size:',nx,ny,nz)





### read field labels ####
for i in range(7,7):

    all_heat_maps = np.zeros((500,nx,ny,nz))
    # (500,3)
    labels = np.load('/projects/piml_inversion/hwang/Illinois/src_chicoma/data/field_syn_data_bandpass_trace_norm_500_1_1000_31/field_label_500_'+str(i)+'.npy')
#    print('i:',i,labels.shape,'x range:',labels[:,0].max(),labels[:,0].min(),'y range:',labels[:,1].max(),labels[:,1].min(),'z range:',labels[:,2].max(),labels[:,2].min())

    for idata in range(500):
#        xx = int(round((labels[idata,0]-kx)/dx,0))
#        yy = int(round((labels[idata,1]-ky)/dx,0))
#        zz = int(round((labels[idata,2]-kz)/dx,0))
#        tmp_heat_map = np.zeros((nx,ny,nz))
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    all_heat_maps[idata,ix,iy,iz]=gaussian_xyz(ix*dx+kx,iy*dy+ky,iz*dz+kz,labels[idata,0],labels[idata,1],labels[idata,2],0.5)
#        tmp_heat_map[xx,yy] = 1
#        tmp_heat_map[xx,yy,zz] = gaussian_filter(tmp_heat_map[xx,yy,zz],sigma=3)
#        all_heat_maps[idata,:,:] = tmp_heat_map#gaussian_filter(tmp_heat_map,sigma=50)
        random_map = 2*0.008*(np.random.rand(nx,ny,nz)-0.5)
        all_heat_maps[idata,:,:] = all_heat_maps[idata,:,:] + random_map
#    all_heat_maps = all_heat_maps/(all_heat_maps.max())
    np.save('./field_label_heat_map_random_intz_500_'+str(i)+'.npy',np.float32(all_heat_maps))
    print('done field label ',i,'/6','range:',all_heat_maps.max(),all_heat_maps.min())
#plt.figure()
#plt.subplot(1,2,1)
#plt.imshow(all_heat_maps[0,:,:,5],cmap='gray')
#plt.colorbar()
#plt.subplot(1,2,2)
#plt.imshow(all_heat_maps[1,:,:,5],cmap='gray')
#plt.colorbar()
#plt.show()

#exit()
### read syn labels ####
for i in range(23,32):

    all_heat_maps = np.zeros((500,nx,ny,nz))
    # (500,3)
    labels = np.load('/projects/piml_inversion/hwang/Illinois/src_chicoma/data/field_syn_data_bandpass_trace_norm_500_1_1000_31/syn_label_500_'+str(i)+'.npy')
#    print('i:',i,labels.shape,'x range:',labels[:,0].max(),labels[:,0].min(),'y range:',labels[:,1].max(),labels[:,1].min(),'z range:',labels[:,2].max(),labels[:,2].min())

    for idata in range(500):
#        xx = int(round((labels[idata,0]-kx)/dx,0))
#        yy = int(round((labels[idata,1]-ky)/dx,0))
#        zz = int(round((labels[idata,2]-kz)/dx,0))
#        tmp_heat_map = np.zeros((nx,ny))
#        tmp_heat_map[xx,yy] = 1
#        tmp_heat_map[xx,yy,zz] = gaussian_filter(tmp_heat_map[xx,yy,zz],sigma=3)
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    all_heat_maps[idata,ix,iy,iz]=gaussian_xyz(ix*dx+kx,iy*dy+ky,iz*dz+kz,labels[idata,0],labels[idata,1],labels[idata,2],0.5)

        random_map = 2*0.008*(np.random.rand(nx,ny,nz)-0.5)
        all_heat_maps[idata,:,:] = all_heat_maps[idata,:,:] + random_map

    np.save('./syn_label_heat_map_XY_random_intz_500_'+str(i)+'.npy',np.float32(all_heat_maps))
    print('done syn label ',i,'/31','range:',all_heat_maps.max(),all_heat_maps.min())

