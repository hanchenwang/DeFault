import numpy as np
import matplotlib.pyplot as plt
import transforms as T
import random
import scipy
from glob import glob
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from PIL import Image

def second_largest(numbers):
    count = 0
    m1=m2=float('-inf')
    for x in numbers:
        count+=1
        #print(x,m2)
        if x>m2:
            if x>=m1:
                m1,m2=x,m1
            else:
                m2=x
    return m2 if count>=2 else None

def upsampling(data,factor,o):
    return scipy.ndimage.zoom(data, factor, order=o)



receiver = np.load('./receiver_geometry.npy')
label = np.load('./locl_upsample.npy')
pred = np.load('./locp_upsample.npy')
# shape(3335,3) # dim1 = 3 -> x, y, z of all receiver locations in km 
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
#xx = int(round((labels[idata,0]-kx)/dx,0))
#yy = int(round((labels[idata,1]-ky)/dx,0))
#zz = int(round((labels[idata,2]-kz)/dx,0))

count = np.linspace(1,3500,num=3500)
#print('pred shape:',pred.shape,pred.max(),pred.min())#(700, 317, 395)
#print('label shape:',label.shape)#(700, 317, 395)
label_min = -0.008#-1.72
label_max = 0.806#3.42#2.83

locp = np.load('./locp_upsample.npy')

#fig, ax = plt.subplots()
rect1 = patches.Rectangle((0, 0.01),2.24,3.38,linewidth=2,edgecolor='black',facecolor='none')
rect2 = patches.Rectangle((0, -2.35),2.24,2.35,linewidth=2,edgecolor='black',facecolor='none')
rect3 = patches.Rectangle((0.01, -2.35),3.39,2.35,linewidth=2,edgecolor='black',facecolor='none')
count = np.linspace(1,3500,num=3500)
plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.scatter(label[:,1],label[:,0],marker='o',s=8)
plt.scatter(pred[:,1],pred[:,0],marker='^',c=count,cmap='viridis',s=16)
plt.scatter(receiver[2:31,1],receiver[2:31,0],c='red',s=20)
plt.scatter(receiver[:2,1],receiver[:2,0],c='black',marker='^',s=20)
#plt.gca().add_patch(rect1)
plt.title('X-Y view')
plt.xlabel('Easting (km)')
plt.ylabel('Northing (km)')
plt.xlim([-1.73, 2.25])
plt.ylim([0, 3.4])
#plt.colorbar()
plt.subplot(1,3,2)
plt.scatter(label[:,1],-label[:,2],marker='o',s=8)
plt.scatter(pred[:,1],-pred[:,2],marker='^',c=count,cmap='viridis',s=16)
plt.scatter(receiver[2:31,1],-receiver[2:31,2],c='red',s=20)
plt.scatter(receiver[:2,1],-receiver[:2,2],c='black',marker='^',s=20)
#plt.gca().add_patch(rect2)
plt.title('X-Z view')
plt.xlabel('Easting (km)')
plt.ylabel('Depth (km)')
plt.xlim([-1.73, 2.25])
plt.ylim([-2.35, 0])
#plt.colorbar()
plt.subplot(1,3,3)
plt.scatter(label[:,0],-label[:,2],marker='o',s=8)
plt.scatter(pred[:,0],-pred[:,2],marker='^',c=count,cmap='viridis',s=16)
plt.scatter(receiver[2:31,0],-receiver[2:31,2],c='red',s=20)
plt.scatter(receiver[:2,0],-receiver[:2,2],c='black',marker='^',s=20)
#plt.gca().add_patch(rect3)
plt.title('Y-Z view')
plt.xlabel('Northing (km)')
plt.ylabel('Depth (km)')
plt.xlim([0, 3.4])
plt.ylim([-2.35, 0])
#plt.colorbar()
#plt.show()
plt.savefig('./2D_syn_train_fd_pred_max_only_pred_upsample_max.png')
plt.close()



rect1 = patches.Rectangle((0, 0.01),2.24,3.38,linewidth=2,edgecolor='black',facecolor='none')
rect2 = patches.Rectangle((0, -2.35),2.24,2.35,linewidth=2,edgecolor='black',facecolor='none')
rect3 = patches.Rectangle((0.01, -2.35),3.39,2.35,linewidth=2,edgecolor='black',facecolor='none')
plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.scatter(label[:,1],label[:,0],marker='o',s=8)
plt.scatter(0.75*pred[:,1]+0.25*pred[:,4],(0.75*pred[:,0]+0.25*pred[:,3]),marker='^',c=count,cmap='viridis',s=16)
plt.scatter(receiver[2:31,1],receiver[2:31,0],c='red',s=20)
plt.scatter(receiver[:2,1],receiver[:2,0],c='black',marker='^',s=20)
#plt.gca().add_patch(rect1)
plt.title('X-Y view')
plt.xlabel('Easting (km)')
plt.ylabel('Northing (km)')
plt.xlim([-1.73, 2.25])
plt.ylim([0, 3.4])
#plt.colorbar()
plt.subplot(1,3,2)
plt.scatter(label[:,1],-label[:,2],marker='o',s=8)
plt.scatter(0.75*pred[:,1]+0.25*pred[:,4],-(0.75*pred[:,2]+0.25*pred[:,5]),marker='^',c=count,cmap='viridis',s=16)
plt.scatter(receiver[2:31,1],-receiver[2:31,2],c='red',s=20)
plt.scatter(receiver[:2,1],-receiver[:2,2],c='black',marker='^',s=20)
#plt.gca().add_patch(rect2)
plt.title('X-Z view')
plt.xlabel('Easting (km)')
plt.ylabel('Depth (km)')
plt.xlim([-1.73, 2.25])
plt.ylim([-2.35, 0])
#plt.colorbar()
plt.subplot(1,3,3)
plt.scatter(label[:,0],-label[:,2],marker='o',s=8)
plt.scatter(0.75*pred[:,0]+0.25*pred[:,3],-(0.75*pred[:,2]+0.25*pred[:,5]),marker='^',c=count,cmap='viridis',s=16)
plt.scatter(receiver[2:31,0],-receiver[2:31,2],c='red',s=20)
plt.scatter(receiver[:2,0],-receiver[:2,2],c='black',marker='^',s=20)
#plt.gca().add_patch(rect3)
plt.title('Y-Z view')
plt.xlabel('Northing (km)')
plt.ylabel('Depth (km)')
plt.xlim([0, 3.4])
plt.ylim([-2.35, 0])
#plt.colorbar()
#plt.show()
plt.savefig('./2D_syn_train_fd_pred_max_only_pred_upsample_int.png')




#####3#########3#####3####3#####################3##

'''
import numpy as np
from scipy.special import softmax

# Generate a 3D coordinate grid
grid_x, grid_y, grid_z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz))

# Function to calculate soft-argmax
def soft_argmax(pred):
    # Apply softmax to get probabilities
    prob = softmax(pred.flatten())

    # Compute expectation
    expected_x = np.sum(grid_x.flatten() * prob)
    expected_y = np.sum(grid_y.flatten() * prob)
    expected_z = np.sum(grid_z.flatten() * prob)

    return [expected_x, expected_y, expected_z]

# Apply soft-argmax to each predicted heatmap
for i in range(num_heatmaps):  # num_heatmaps is the total number of heatmaps
    pred_i = np.load(f'pred_{i}.npy')  # Assuming your heatmaps are saved as .npy files
    coordinates_i = soft_argmax(pred_i)
    print(f'The coordinates for heatmap {i} are: {coordinates_i}')

'''


