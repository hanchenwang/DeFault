from rsf.proj import *
import math
import numpy as np

#Flow('spike_right',None,'spike d1=2 n1=501 d2=0.015625 n2=64 o1=0 o2=0 nps=2 mag=[0,1] k2=[0,33] l2=[33,64] | put o2=-0.5')

for i in range(3,7):
    Flow('filter_all_norm'+str(i),'field_data_filter_all_norm_500_'+str(i),'window f1=32 | put o1=1 d2=0.0005 n2=1000 o2=0 d3=1 n3=500 o3=1 d1=1 n1=31 | transp plane=13 ')#field_data_norm_envelope_filtered_500_
    Flow('syn'+str(i),'syn_catalog_500_'+str(i),'window f1=32 | put o1=1 d2=0.0005 n2=1000 o2=0 d3=1 n3=500 o3=1 d1=1 n1=31 | transp plane=13 ')

#for i in range(3,7):

    Flow('clip_all_tmp_'+str(i)+'_0','filter_all_norm'+str(i),'window n1=1 f1=0 ')

    Flow('clip_all_mask_real_'+str(i)+'_0','filter_all_norm'+str(i),'window n1=1 f1=0 |  fft1  | fft3 | real | math output="0" ')
    Flow('clip_all_mask_imag_'+str(i)+'_0','filter_all_norm'+str(i),'window n1=1 f1=0  |  fft1  | fft3 | imag | math output="0" ')
    for j in range(500):

        # find catalog masks
        Flow('clip_syn_'+str(i)+'_'+str(j),'syn'+str(i),'window n1=1 f1=%d ' %(j))
        Flow('clip_syn_'+str(i)+'_'+str(j)+'_fft','clip_syn_'+str(i)+'_'+str(j),' fft1  | fft3 ')#| math output="input*exp(I*%g*x2*x1)"% (2*2*math.pi)
        Flow('clip_syn_'+str(i)+'_'+str(j)+'_mask_real','clip_syn_'+str(i)+'_'+str(j)+'_fft','real | scale axis=123')
        Flow('clip_syn_'+str(i)+'_'+str(j)+'_mask_imag','clip_syn_'+str(i)+'_'+str(j)+'_fft','imag | scale axis=123')
        Flow('clip_all_mask_real_'+str(i)+'_'+str(j+1),['clip_all_mask_real_'+str(i)+'_'+str(j),'clip_syn_'+str(i)+'_'+str(j)+'_mask_real'],'math a=${SOURCES[1]} output="input+a"')        
        Flow('clip_all_mask_imag_'+str(i)+'_'+str(j+1),['clip_all_mask_imag_'+str(i)+'_'+str(j),'clip_syn_'+str(i)+'_'+str(j)+'_mask_imag'],'math a=${SOURCES[1]} output="input+a"')        

        Flow('rm-clip_syn_'+str(i)+'_'+str(j)+'_fft','clip_syn_'+str(i)+'_'+str(j)+'_fft','sfrm ${SOURCES[0]}',stdout=-1)
        Flow('rm-clip_syn_'+str(i)+'_'+str(j),'clip_syn_'+str(i)+'_'+str(j),'sfrm ${SOURCES[0]}',stdout=-1)
        Flow('rm-clip_syn_'+str(i)+'_'+str(j)+'_mask_real','clip_syn_'+str(i)+'_'+str(j)+'_mask_real','sfrm ${SOURCES[0]}',stdout=-1)
        Flow('rm-clip_syn_'+str(i)+'_'+str(j)+'_mask_imag','clip_syn_'+str(i)+'_'+str(j)+'_mask_imag','sfrm ${SOURCES[0]}',stdout=-1)
        Flow('rm-clip_all_mask_real_'+str(i)+'_'+str(j),'clip_all_mask_real_'+str(i)+'_'+str(j),'sfrm ${SOURCES[0]}',stdout=-1)
        Flow('rm-clip_all_mask_imag_'+str(i)+'_'+str(j),'clip_all_mask_imag_'+str(i)+'_'+str(j),'sfrm ${SOURCES[0]}',stdout=-1)

    Flow('clip_all_mask_real_'+str(i),'clip_all_mask_real_'+str(i)+'_500','scale axis=123 ')
    Flow('clip_all_mask_imag_'+str(i),'clip_all_mask_imag_'+str(i)+'_500','scale axis=123 ')
    Flow('rm-clip_all_mask_real_'+str(i)+'_500','clip_all_mask_real_'+str(i)+'_500','sfrm ${SOURCES[0]}',stdout=-1)
    Flow('rm-clip_all_mask_imag_'+str(i)+'_500','clip_all_mask_imag_'+str(i)+'_500','sfrm ${SOURCES[0]}',stdout=-1)

    for j in range(500):
        # pick 1 slice from data
        Flow('clip_'+str(i)+'_'+str(j),'filter_all_norm'+str(i),'window n1=1 f1=%d ' %(j))
        # fft both dim
        Flow('clip_'+str(i)+'_'+str(j)+'_fft','clip_'+str(i)+'_'+str(j),' fft1  | fft3 ')#| math output="input*exp(I*%g*x2*x1)"% (2*2*math.pi)

#        # dipfilter
#        Flow('clip_'+str(i)+'_'+str(j)+'_fft_dip','clip_'+str(i)+'_'+str(j)+'_fft','dipfilter v1=150 v2=151 v3=9999 v4=10000 pass=y ' )
#        # real part after dipfilter
#        Flow('clip_'+str(i)+'_'+str(j)+'_fft_dip_fkreal',['clip_'+str(i)+'_'+str(j)+'_fft_dip','spike_right_2','spike_all'],'real | math a=${SOURCES[1]} b=${SOURCES[2]} output="input*(1-a)*(1-b)" ')
#        # imag part after dipfilter
#        Flow('clip_'+str(i)+'_'+str(j)+'_fft_dip_fkimag',['clip_'+str(i)+'_'+str(j)+'_fft_dip','spike_right_2','spike_all'],'imag | math a=${SOURCES[1]} b=${SOURCES[2]} output="input*(1-a)*(1-b)" ')

        # real part after dipfilter
        Flow('clip_'+str(i)+'_'+str(j)+'_fft_dip_fkreal',['clip_'+str(i)+'_'+str(j)+'_fft','clip_all_mask_real_'+str(i)],'real | math a=${SOURCES[1]} output="input*a" ')
        # imag part after dipfilter
        Flow('clip_'+str(i)+'_'+str(j)+'_fft_dip_fkimag',['clip_'+str(i)+'_'+str(j)+'_fft','clip_all_mask_imag_'+str(i)],'imag | math a=${SOURCES[1]} output="input*a" ')

        # ifft to time-space domain
        Flow('clip_'+str(i)+'_'+str(j)+'_fft_dip_fk_ifft',['clip_'+str(i)+'_'+str(j)+'_fft_dip_fkreal','clip_'+str(i)+'_'+str(j)+'_fft_dip_fkimag'],'cmplx ${SOURCES[1]} | fft3 inv=y | fft1 inv=y | scale axis=12' )
        # cat all data
        Flow('clip_all_tmp_'+str(i)+'_'+str(j+1),['clip_all_tmp_'+str(i)+'_'+str(j),'clip_'+str(i)+'_'+str(j)+'_fft_dip_fk_ifft'],'cat axis=3 ${SOURCES[1]} ')
        
        # remove unused data
        Flow('rm-clip_'+str(i)+'_'+str(j),'clip_'+str(i)+'_'+str(j),'sfrm ${SOURCES[0]}',stdout=-1)
        Flow('rm-clip_'+str(i)+'_'+str(j)+'_fft','clip_'+str(i)+'_'+str(j)+'_fft','sfrm ${SOURCES[0]}',stdout=-1)
#        Flow('rm-clip_'+str(i)+'_'+str(j)+'_fft_dip','clip_'+str(i)+'_'+str(j)+'_fft_dip','sfrm ${SOURCES[0]}',stdout=-1)
        Flow('rm-clip_'+str(i)+'_'+str(j)+'_fft_dip_fkreal','clip_'+str(i)+'_'+str(j)+'_fft_dip_fkreal','sfrm ${SOURCES[0]}',stdout=-1)
        Flow('rm-clip_'+str(i)+'_'+str(j)+'_fft_dip_fkimag','clip_'+str(i)+'_'+str(j)+'_fft_dip_fkimag','sfrm ${SOURCES[0]}',stdout=-1)
        Flow('rm-clip_'+str(i)+'_'+str(j)+'_fft_dip_fk_ifft','clip_'+str(i)+'_'+str(j)+'_fft_dip_fk_ifft','sfrm ${SOURCES[0]}',stdout=-1)
        Flow('rm-clip_all_tmp_'+str(i)+'_'+str(j),'clip_all_tmp_'+str(i)+'_'+str(j),'sfrm ${SOURCES[0]}',stdout=-1)



    Flow('clip_all_filtered_'+str(i)+'_500_1000_31','clip_all_tmp_'+str(i)+'_500','window f3=1 n3=500 | put n1=1000 n2=31 n3=500 d2=1 d1=0.0005 d3=1 o1=0 o2=1 o3=1 | bandpass fhi=70 nphi=20 | transp plane=13 | transp plane=23 ')
    Flow('rm-clip_all_tmp_'+str(i)+'_500','clip_all_tmp_'+str(i)+'_500','sfrm ${SOURCES[0]}',stdout=-1)
#for i in range(0,7):
#    with open ('clip_all_filtered_'+str(i)+'_500_1000_31.rsf@','rb') as f1:
#        tmp = np.fromfile(f1,np.single).reshape(1,31,1000,500).transpose(3,0,2,1)
#    np.save('./field_data_fk_filtered_all_500_'+str(i)+'.npy',np.float32(tmp))
    
