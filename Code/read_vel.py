import numpy as np
import segyio
from scipy import signal
import matplotlib.pyplot as plt

'''
for i in range(1500,1600):
	b=461471+1
	print('factoring by ',i,':',b/i)


exit()
'''
mig_vel = np.zeros((2501,873201))

with segyio.open('./ibdp_active_seismic/IBDP_4D_Seismic_Volume/Baseline_volume_2011/Final_PSTM_stacking_vel_2011/t2000_20_82_bl_stkvel_segy_interval_at_surface',ignore_geometry=True) as f1:#[15000,4320]-[60s,1440*3traces]
    read_index=0
    for trace in f1.trace:#4320 traces in total [15000,1]
        print('trace ',read_index, 'shape:',trace.shape)
        mig_vel[:,read_index] = trace
        read_index += 1
        
    np.save('./stkvel_interval_at_surface_2501_301_2901.npy',mig_vel.reshape((2501,301,2901)))

    for dx in f1.header:

        print(dx)

f2m = 3.28084
mig_vel = np.load('stkvel_interval_at_surface_2501_301_2901.npy')#('/home/hwang/Desktop/illinois/ibdp_active_seismic/IBDP_4D_Seismic_Volume/Baseline_volume_2011/Final_PSTM_migration_vel_2011/t2000_20_82_bl_migvel_segy_rms_at_surface.npy')
mig_vel2 = mig_vel.reshape((2501,301,2901))
##np.save('/home/hwang/Desktop/illinois/ibdp_active_seismic/IBDP_4D_Seismic_Volume/Baseline_volume_2011/Final_PSTM_migration_vel_2011/t2000_20_82_bl_migvel_segy_rms_at_surface_2501_2901_301.npy',mig_vel2)



plt.figure(figsize=(40,120))
plt.imshow(mig_vel2[:,151,:]/f2m,cmap='jet')
plt.colorbar()
plt.show()

#corr = signal.correlate(sig_noise, np.ones(128), mode='same') / 128





#/home/hwang/Desktop/illinois/ibdp_active_seismic/IBDP_4D_Seismic_Volume/Monitor_volume_2015/2015_Full_Survey/t2000_20_102_bl_Yr2015full_ovt_pstm_enhanced_stk_segy
