User Manual FY23 Milestone 4.3.3:
An ML-based tool for fault/fracture identification

Progress:
We developed a novel deep learning method, DeFault, specifically designed for IBDP passive seismic source relocation and fault delineating for passive seismic monitoring projects at Decatur, Illinois area. By leveraging data domain-adaptation, our method allows us to train a neural network with labeled synthetic data and apply it directly to field data. The passive seismic sources are automatically clustered based on their recording time and spatial locations, and subsequently, faults and fractures are delineated accordingly.

Resulting ML-Based Tools Developed for SMART:
DeFault algorithm has three consecutive steps that will be delivered and uploaded to EDX:
	MLReal data processing algorithm: 
Input detected raw seismic waveform -> output MLReal data for deep-learning relocation inference
	Deep-learning relocation algorithm: 
Input MLReal data -> output seismic event source location
	Fault/fracture identification algorithm:
Input relocated source locations -> output fault planes and fractures

Resulting ML-Based Products, Publications, etc. for SMART
	Wang, H., Chen, Y., Alkhalifah, T., Chen, T., Lin, Y., and Alumbaugh, D., “DeFault: DEep-learning-based FAULT Delineation Using the IBDP Passive Seismic Data at the Decatur CO2 Storage Site,” submitted to the Earth and Space Science, Nov. 20, 2023.
	Wang, H, Chen, Y, Alkhalifah, T, Lin, Y. “Fault MLReal: A fault delineation study for the Decatur CO2 field data using neural network predicted passive seismic locations,” InThird International Meeting for Applied Geoscience & Energy. Society of Exploration Geophysicists and American Association of Petroleum Geologists. 2023 Aug 1 (pp. 386-390), presented at IMAGE 2023, Houston, TX, USA, Aug. 2023



Task 4.3.3 Overview:
 
Figure 1IBDP monitoring system geometry
We utilize the seismic monitoring data from CCS-1 and GM-1 borehole receivers to relocate the sources and therefore delineate the faults and fractures induced by CO2 injection at IBDP. 
 
Figure 2 Microseismic event migration over time.


DeFault Algorithm Overview:
 
Figure 3 DeFault algorithm scratch.
The acoustic wave equation in 3D for a medium with constant density is typically represented as follows: 
\frac{\partial^2p}{\partialt^2}=c^2\nabla^2p
Here, 𝑝 represents the pressure field, 𝑐 is the speed of sound in the medium, \frac{\partial^2p}{\partial t^2} is the second time derivative of pressure, and \nabla^2p (the Laplacian of 𝑝) represents the spatial second derivatives of the pressure. 
Passive seismic sources 𝑠, which are triggered by CO2 injection, generates the seismic waves, which travel through the subsurface medium such as layer of rocks or water.

The workflow can be divided into four main steps:
	Signal processing and enhancement: We first apply a series of filters to the raw field seismic recorded waveforms to remove noise and enhance effective signals, specifically, to improve the signal-to-noise ratio (SNR).
	Deep-learning-based event location: A large synthetic training set is generated, and combined with the processed field data to be fed into an MLReal domain adaptation encoder-decoder neural network training stage.
	K-means clustering: The processed field data pass through the MLReal steps and is fed into the trained encoder-decoder network to predict microseismic event locations. These predictions within a certain time period are then clustered with a K-means algorithm, which automatically separates the events spatially. A least square distance match is then applied to find the corresponding fault plane in each cluster.
	Dropout uncertainty analysis: Dropout uncertainty analysis involves modifying a neural network by inserting dropout layers, which randomly ignore a subset of neurons during training. This technique often prevents overfitting, and thus, promotes a model that can generalize better to unseen data. By applying dropout during testing (not just training), we can measure the variability in the network's predictions, offering a practical approach to gauge the model's confidence in its predictions. Essentially, this method allows us to assess the reliability of the neural network's outputs, which is crucial for understanding the potential range of outcomes and making informed decisions based on the model's predictions. We modify and retrain the network architecture by adding dropout layers with dropout rate ( p = 0.2 ) after each convolutional layer except the last one, and use it to perform uncertainty analysis.
Signal processing and enhancement:
Input detected raw seismic waveform -> output MLReal data for deep-learning relocation inference

# Users will need to install Madagascar functions prior to the commands. Download and install from https://www.reproducibility.org/wiki/Main_Page
# User will need to install basic Python libraries, including PyTorch, CUDA, torchvision, numpy, scipy, scikit-learn, pandas, matplotlib, geopandas, seaborn.

We have totally simulated 15000 seismic sources, using the SLB active seismic survey interval Vp model, a Ricker wavelet with 30Hz peak frequency as the source function, to build our training set.
Signal processing includes 6 main steps:
	Lowpass Filter -> 2. Notch bandpass filters -> 3. Trace-wise amplitude normalization ->
	F-k domain dipping filter -> 5. Time domain anomalous noise purge -> 6. F-k domain envelop filtering
These steps are specifically designed for IBDP passive borehole monitoring data. 
Then the processed data will go through a MLReal step for data distribution merge.

To get the following figures, run: 
	scons -f SConstruct.py 
	python get_npy.py
	python get_mlreal.py

 
Figure 4 Raw, filtered, and MLReal data.
Deep-learning-based Event Relocation:
Input MLReal data -> output seismic event source location
 
Figure 5 Relocation deep-learning network architecture.

You will need to 
	Run gen_heat_map.py to generate the heatmaps for training. 
	Revise the train_notch_filter_avg_mask_env_syn_normal_only_random_intz_heat_map.txt 
and train_notch_filter_avg_mask_env_fd_random_intz_heat_map.txt files. Redirect the inside directories to the above processed data and heatmaps. 
	Run training with: sh job_train_milestone.sh on Nvidia gpu clusters.
To test the training:
python -u test_demo.py -o models -n mlreal_IBDP -s demo -m print_labels --up_mode nearest -ds illinois_heat_map_3D -b 1000 -eb 301 -nb 1 -j 1 --lr 1e-3 --tensorboard -g1v 0.5 -g2v 0.5 --k 1 -t test_notch_filter_avg_mask_env_fd_random_intz_heat_map_dup_p1.txt -v test_notch_filter_avg_mask_env_fd_random_intz_heat_map_dup_p1.txt --resume ./heat3d_demo/checkpoint.pth
Then, run plot.py to get the upsampled heatmaps and the corresponding relocated coordinates.

To check the results:
import numpy as np
labelsf = np.load('labelsf.npy')
print('labelsf size:', labelsf.shape, '= (sample,[X,Y,Z])')
python show_event_locations.py
from IPython.display import Image
from IPython.display import display
a=Image('Figure_run10_3D_faults/3D_view_1440_orange_locations.png', width = 600, height = 600)
b=Image('Figure_run10_3D_faults/3D_XY_view_1440_orange_locations.png', width = 600, height = 600)
c=Image('Figure_run10_3D_faults/3D_XZ_view_1440_orange_locations.png', width = 600, height = 600)
d=Image('Figure_run10_3D_faults/3D_YZ_view_1440_orange_locations.png', width = 600, height = 600)
display(a,b,c,d)
This is the file that contains the relocated IBDP event coordinates.
Ready for fault/fracture delineation.
K-means clustering:
Main file: kmean_3D_by_period.py and kmean_3D_by_period_iterative.py

Estimating faults through relocated event locations:

Input seismic event locations -> output delineated faults/fractures
[𝑋,𝑌,𝑍]→𝐹𝑎𝑢𝑙𝑡𝐿𝑖𝑛𝑒𝑠

python kmean_3D_by_period_iterative.py
from IPython.display import Image
from IPython.display import display
a=Image('Figure_run10_3D_faults_itr/3D_view_1440_orange.png', width = 600, height = 600)
b=Image('Figure_run10_3D_faults_itr/3D_XY_view_1440_orange.png', width = 600, height = 600)
c=Image('Figure_run10_3D_faults_itr/3D_XZ_view_1440_orange.png', width = 600, height = 600)
d=Image('Figure_run10_3D_faults_itr/3D_YZ_view_1440_orange.png', width = 600, height = 600)
display(a,b,c,d)
Input seismic event locations -> output delineated faults/fractures
python kmean_3D_by_period.py
from IPython.display import Image
from IPython.display import display
a=Image('Figure_run10_3D_faults/3D_view_1440_orange.png', width = 600, height = 600)
b=Image('Figure_run10_3D_faults/3D_XY_view_1440_orange.png', width = 600, height = 600)
c=Image('Figure_run10_3D_faults/3D_XZ_view_1440_orange.png', width = 600, height = 600)
d=Image('Figure_run10_3D_faults/3D_YZ_view_1440_orange.png', width = 600, height = 600)
display(a,b,c,d)



