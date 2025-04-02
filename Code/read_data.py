import numpy as np
import segyio
from scipy import signal
import matplotlib.pyplot as plt
from seg2_files.seg2load import seg2_load
from obspy import read
#from obspy.seg2.seg2 import readSEG2
from obspy.core import Stream
from SEG2Py import *

tr = np.zeros((4000,29))
print('try to read')


st = read("/Users/hwang/Desktop/illinois/ibdp_located_microseismic_event_data/IBDP_Located_Microseismic_Event_Data/IBDP_Downhole_Geophone_Data/IBDP_Raw_Event_Data/20160404.08034044.sg2",format="SEG2")
print(st[0].data.shape)
print(st[0].stats.seg2)

save_index=0
for i in [10,13,16,17,19,22,25,28,31,34,37,40,43,46,49,52,55,58,61,64,67,70,73,76,79,82,85,88,91]:
    tr[:,save_index]=st[i].data
    save_index+=1

plt.figure()
#plt.imshow(tr,cmap='gray',aspect=0.02 )
plt.plot(tr)
#plt.colorbar()
plt.show()



'''
tr.append(st[94-1])
new_stream = Stream(traces=tr)
new_stream.merge(method=1, fill_value='interpolate')
print('merge success')
#print('new_stream:',new_stream.shape)
for i in range(0,94):
    print('trace no.=',i,st[i].stats.seg2['RECEIVER_LOCATION'])
    print('trace no.=',i,st[i].stats.seg2['SAMPLE_INTERVAL'])

    print('trace no.=',i,st[i].stats.seg2)

#plotSEG2(st, gain = 3, shading = True, clip = True, record_end = 100)

#st = readSEG2("/Users/hwang/Desktop/illinois/ibdp_located_microseismic_event_data/IBDP_Located_Microseismic_Event_Data/IBDP_Downhole_Geophone_Data/IBDP_Raw_Event_Data/20180609.10355405.sg2", format="SEG2")
#tr.append(st[nch-1])
#new_stream = Stream(traces=tr)
#new_stream.merge(method=1, fill_value='interpolate')


'''



'''
#st = read("/Users/hwang/Desktop/illinois/ibdp_located_microseismic_event_data/IBDP_Located_Microseismic_Event_Data/IBDP_Downhole_Geophone_Data/IBDP_Raw_Event_Data/20180609.10355405.sg2", format="SEG2")

st = readSEG2("/Users/hwang/Desktop/illinois/ibdp_located_microseismic_event_data/IBDP_Located_Microseismic_Event_Data/IBDP_Downhole_Geophone_Data/IBDP_Raw_Event_Data/20180609.10355405.sg2", format="SEG2")


seis_data, seis_header = seg2_load('/Users/hwang/Desktop/illinois/ibdp_located_microseismic_event_data/IBDP_Located_Microseismic_Event_Data/IBDP_Downhole_Geophone_Data/IBDP_Raw_Event_Data/20180609.10355405.sg2')

#for trace in st.trace:
#    print(trace.shape)
#print(st)
#print(st.__str__(extended=True))
#traces, header = seg2_load('/Users/hwang/Desktop/illinois/ibdp_located_microseismic_event_data/IBDP_Located_Microseismic_Event_Data/IBDP_Downhole_Geophone_Data/IBDP_Raw_Event_Data/20180702.00391289.sg2')
#print(st.shape)
#print(traces.shape)
#print('header:',header)

#np.save('./stkvel_interval_at_surface_2501_301_2901.npy',traces.reshape((2501,301,2901)))

##np.save('/home/hwang/Desktop/illinois/ibdp_active_seismic/IBDP_4D_Seismic_Volume/Baseline_volume_2011/Final_PSTM_migration_vel_2011/t2000_20_82_bl_migvel_segy_rms_at_surface_2501_2901_301.npy',mig_vel2)
#plt.figure(figsize=(40,120))
#plt.imshow(mig_vel[:,:5000]/f2m,cmap='jet')
#plt.colorbar()
#plt.show()

#corr = signal.correlate(sig_noise, np.ones(128), mode='same') / 128





#/home/hwang/Desktop/illinois/ibdp_active_seismic/IBDP_4D_Seismic_Volume/Monitor_volume_2015/2015_Full_Survey/t2000_20_102_bl_Yr2015full_ovt_pstm_enhanced_stk_segy
'''

