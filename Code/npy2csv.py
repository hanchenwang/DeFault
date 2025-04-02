import numpy as np

a=np.load('/Users/hwang/Desktop/locl_upsample.npy')

b=np.zeros((3500,3))

#for i in range(3500):
#    b[i,0]=a[i,0]*0.75+a[i,3]*0.25
#    b[i,1]=a[i,1]*0.75+a[i,4]*0.25
#    b[i,2]=-(a[i,2]*0.75+a[i,5]*0.25)

np.savetxt("label_events_upsample_run10.csv", a, delimiter=",")
