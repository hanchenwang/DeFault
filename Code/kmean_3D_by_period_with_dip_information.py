import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from datetime import datetime
import warnings
import seaborn as sns
from datetime import timedelta
from datetime import datetime
import imageio
import os
from PIL import Image, ImageSequence
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
import math

# plot configure
draw_uq=False
draw_rolling_gif=False # rolling view 3D figure
save_all_periods=False # Has to be True for draw_gif
draw_gif=False # plume migration by predicted events
draw_catalog=True
draw_pred=True
draw_fault_plane=True
draw_fault_line=True
draw_arrow=False
draw_receiver=True
dipping_annotation = True
dipping_anno_num = 5

cur_anno_num = 0
period_length = 30  # Change this to your desired period length in days

if period_length == 30:
    # Filter points that are too far from the centroid of the cluster
    threshold_distance = 0.15 #km
    fault_line_length_threshold = 1 #km
elif period_length == 1440:
    # Filter points that are too far from the centroid of the cluster
    threshold_distance = 0.35#0.15 #km
    fault_line_length_threshold = 2 #km

labels=np.zeros((3500,3))
label = np.load('./labelsf.npy')
label = (label/2+0.5)*(3.42+1.61)-1.61
labels[:,0]= label[:,1]
labels[:,1]= label[:,0]
labels[:,2]= -label[:,2]
#np.savetxt("catalog_labels.csv", labels, delimiter=",")
print('Long',labels[:,0].min(),labels[:,0].max())
print('Lat',labels[:,1].min(),labels[:,1].max())
print('Depth',labels[:,2].min(),labels[:,2].max())


c_min = 5 # minimum clusters in a time period
def create_time_period_group(date, period_length):
    """
    This function returns the starting date of the time period in which 'date' falls.
    'period_length' is the length of the time period in days.
    """
    n_periods = date.toordinal() // period_length
    period_start_date = datetime.fromordinal(n_periods * period_length)
    return period_start_date.date()

def calculate_dipping_angle_and_azimuth(m1, m2):
    # Normal vector to the plane
    normal_vector = np.array([-m1, -m2, 1])
    normal_vector /= np.linalg.norm(normal_vector)  # Normalize the vector

    # Calculate dipping angle
    dipping_angle = np.arccos(normal_vector[2])  # Angle with the vertical
    dipping_angle = np.degrees(dipping_angle)  # Convert to degrees

    # Calculate azimuth
    azimuth = np.arctan2(normal_vector[1], normal_vector[0])  # Angle in XY plane
    azimuth = np.degrees(azimuth)  # Convert to degrees
    if azimuth < 0:
        azimuth += 360  # Ensure the azimuth is between 0 and 360 degrees

    return dipping_angle, azimuth
    
# Ignore all warning messages
warnings.filterwarnings("ignore")

# Read the CSV file into a pandas DataFrame
event_data = pd.read_csv('relocated_events_upsampling_int_date_time_fit_coordinates_run8.csv')

# Convert the 'date' column to datetime format
event_data['date'] = pd.to_datetime(event_data['date'])
event_data['period'] = event_data['date'].apply(create_time_period_group, args=(period_length,))
# Extract month and year from the 'date' column
event_data['month'] = event_data['date'].dt.month
event_data['year'] = event_data['date'].dt.year

# Set the spatial range for figures
latitude_range = [0, 3.4]  # km
longitude_range = [-1.73, 2.25]  # km
depth_range = [-2.5, -1.6]  # km

# Group events by month and perform K-means clustering for each month
grouped = event_data.groupby(['year', 'period'])

# Create a figure to plot all the clusters and lines
fig_all = plt.figure(figsize=(15, 15))
ax_all = fig_all.add_subplot(111, projection='3d')

previous_planes = []  # Store the planes of previous months
# Load receiver geometry from the npy file
with open('./receiver.rsf@','r') as f1:
    receiver_geometry = np.fromfile(f1,np.single()).reshape(31,3)
f1.close()
receiver_geometry[:,0], receiver_geometry[:,1] = receiver_geometry[:,1], receiver_geometry[:,0]

receiver_geometry[:,2]=-receiver_geometry[:,2]

predictions = np.load('./all_predictions.npy')[:,:3,:]
mean = np.mean(predictions, axis=2)
std = np.std(predictions, axis=2)
variance = np.var(predictions, axis=2)
print(mean.shape)
##########
##########
##########
##########
##########
##########
##########
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111, projection='3d')

if period_length == 30:
    color_all = plt.cm.jet(np.linspace(0, 1, 325))#m310/q147/h71/y34
elif period_length == 90:
    color_all = plt.cm.jet(np.linspace(0, 1, 147))#m310/q147/h71/y34
elif period_length == 180:
    color_all = plt.cm.jet(np.linspace(0, 1, 94))#m310/q147/h71/y34
elif period_length == 360:
    color_all = plt.cm.jet(np.linspace(0, 1, 71))#m310/q147/h71/y34
elif period_length == 720:
    color_all = plt.cm.jet(np.linspace(0, 1, 30))#m310/q147/h71/y34
elif period_length == 1440:
    color_all = plt.cm.jet(np.linspace(0, 1, 20))#m310/q147/h71/y34
elif period_length == 2880:
    color_all = plt.cm.jet(np.linspace(0, 1, 10))#m310/q147/h71/y34

    
cstart = 0
ccount = 0

#os.makedirs(image_path, exist_ok=True)  # Create the directory if it doesn't exist

image_files = []  # List to store the filenames of the images
roll_image_files = []  # List to store the filenames of the images
# List to store corner points of each fault plane
fault_plane_corners = []

for name, group in grouped:
    year, period = name
    period_str = period.strftime('%Y-%m-%d')

    # Print year and month
    print(f"Processing year: {year}, period: {period_str}")
    # Create a new figure for each month's clusters

    # Perform K-means clustering if the number of events is larger than the number of clusters
    if len(group) >= 5:
        # Perform K-means clustering
        k = max(len(group) // 100, c_min)  # Number of clusters based on event count (minimum 5)
        kmeans = KMeans(n_clusters=k, random_state=0)
        group['Cluster'] = kmeans.fit_predict(group[['Longitude', 'Latitude', 'Depth']])

        colors = color_all[cstart : cstart + k, :]  # Colors for each cluster
        cstart += k
        ccount += k
        for cluster, color in zip(range(k), colors):
            cluster_points = group[group['Cluster'] == cluster]

            # Calculate the distance of each point to the centroid of the cluster
            distances = np.sqrt(
                (cluster_points['Longitude'] - cluster_points['Longitude'].mean()) ** 2 +
                (cluster_points['Latitude'] - cluster_points['Latitude'].mean()) ** 2 +
                (cluster_points['Depth'] - cluster_points['Depth'].mean()) ** 2
            )

            # Filter points that are too far from the centroid of the cluster
            #threshold_distance = 0.15  # km
            filtered_points = cluster_points[distances <= threshold_distance]

            # Fit a plane to the filtered points using least squares method
            A = np.vstack([filtered_points['Longitude'], filtered_points['Latitude'], np.ones(len(filtered_points))]).T
            m1, m2, c = np.linalg.lstsq(A, filtered_points['Depth'], rcond=None)[0]

            # Check for intersection with previous planes
#            for plane in previous_planes:
                # Calculate intersection line
                # This can be complex in 3D space and depends on the specifics of your application.
                # You might need to implement a method to calculate this.
                # For now, we'll leave it as a placeholder
#                pass

            # Store the current plane
            previous_planes.append((m1, m2, c))
            
            if dipping_annotation:
                dipping_angle, azimuth = calculate_dipping_angle_and_azimuth(m1, m2)
            # Find the bounds of the cluster's spatial range
            cluster_lon_min = filtered_points['Longitude'].min()
            cluster_lon_max = filtered_points['Longitude'].max()
            cluster_lat_min = filtered_points['Latitude'].min()
            cluster_lat_max = filtered_points['Latitude'].max()

            # Define a grid of x, y values for the plane
            x = np.linspace(cluster_lon_min, cluster_lon_max, 50)
            y = np.linspace(cluster_lat_min, cluster_lat_max, 50)
            X, Y = np.meshgrid(x, y)
            Z = m1 * X + m2 * Y + c
            #x = np.linspace(longitude_range[0], longitude_range[1], 50)
            #y = np.linspace(latitude_range[0], latitude_range[1], 50)
            #X, Y = np.meshgrid(x, y)
            #Z = m1 * X + m2 * Y + c

            # Plot the plane
            if draw_fault_plane:
                ax.plot_surface(X, Y, Z, color=color, alpha=0.5)  # using the cluster color with some transparency
            # Plot data points for the cluster with recording time
            if draw_pred:
                ax.scatter(cluster_points['Longitude'], cluster_points['Latitude'], cluster_points['Depth'], c=[color], label=f'Cluster {cluster} - {period_str}', s=5,alpha=1)#'red',[color],cluster_points,filtered_points
#            ax.plot(filtered_points['Longitude'], filtered_points['Depth'], 'r+', zdir='y', zs=1.5)
#            ax.plot(filtered_points['Latitude'], filtered_points['Depth'], 'g+', zdir='x', zs=-0.5)
#            ax.plot(filtered_points['Longitude'], filtered_points['Latitude'], 'k+', zdir='z', zs=-1.5)
            
            
            # Calculate the arrow direction based on recording time
            start_time = filtered_points['time'].min()
            end_time = filtered_points['time'].max()
            arrow_direction = np.sign(end_time - start_time)

            # Calculate arrow length as a fraction of the spatial range
            arrow_length = 0.3 * max(longitude_range[1] - longitude_range[0],
                                      latitude_range[1] - latitude_range[0],
                                      depth_range[1] - depth_range[0])

            # Calculate arrow coordinates
            arrow_start_x = filtered_points['Longitude'].min()
            arrow_start_y = filtered_points['Latitude'].min()
            arrow_start_z = m1 * arrow_start_x + m2 * arrow_start_y + c
            arrow_end_x = filtered_points['Longitude'].max()
            arrow_end_y = filtered_points['Latitude'].max()
            arrow_end_z = m1 * arrow_end_x + m2 * arrow_end_y + c
            # Determine receiver coordinates
            receiver_x, receiver_y, receiver_z = receiver_geometry[0, 0], receiver_geometry[0, 1], receiver_geometry[0, 2]

            # Calculate the distances of arrow start and end points to the first receiver
            dist_start_to_receiver = np.sqrt((arrow_start_x - receiver_x)**2 + (arrow_start_y - receiver_y)**2 + (arrow_start_z - receiver_z)**2)
            dist_end_to_receiver = np.sqrt((arrow_end_x - receiver_x)**2 + (arrow_end_y - receiver_y)**2 + (arrow_end_z - receiver_z)**2)

            # If the start distance is larger than the end distance, flip the arrow
            if dist_start_to_receiver > dist_end_to_receiver:
                arrow_start_x, arrow_end_x = arrow_end_x, arrow_start_x
                arrow_start_y, arrow_end_y = arrow_end_y, arrow_start_y
                arrow_start_z, arrow_end_z = arrow_end_z, arrow_start_z

            # Plot the arrow indicating the recording time direction
            if draw_arrow:
                ax.quiver(
                    arrow_start_x,
                    arrow_start_y,
                    arrow_start_z,
                    arrow_end_x - arrow_start_x,
                    arrow_end_y - arrow_start_y,
                    arrow_end_z - arrow_start_z,
                    color='green',
                    length=arrow_length*2,
                    linewidth=1,
                    alpha=0.8,
    #                linestyle='dashed',
                    edgecolor='black',
                    facecolor='black'
    #                headwidth=1,
    #                headlength=1
                )
            if draw_fault_line:
                fault_line_length = ((arrow_start_x - arrow_end_x)**2 + (arrow_start_y - arrow_end_y)**2 + (arrow_start_z - arrow_end_z)**2)**0.5
                if fault_line_length <= fault_line_length_threshold:
                    ax.plot([arrow_start_x, arrow_end_x],
                            [arrow_start_y, arrow_end_y],
                            [arrow_start_z, arrow_end_z], color='orange',linewidth=3)
                    # Annotation position (midpoint of the fault line)
                    if dipping_annotation and dipping_anno_num >= cur_anno_num:
                        
                        mid_x = (arrow_start_x + arrow_end_x) / 2
                        mid_y = (arrow_start_y + arrow_end_y) / 2
                        mid_z = (arrow_start_z + arrow_end_z) / 2

                        # Annotation text
                        annotation_text = f"Dip Angle: {dipping_angle:.2f}°\nAzimuth: {azimuth:.2f}°\nLength: {fault_line_length:.2f} km"

                        # Annotate the plot with dipping angle, azimuth, and trace length
                        
                        ax.text(mid_x, mid_y, mid_z, annotation_text, color='red')
                        cur_anno_num += 1
                        
                # Calculate corner points
            cluster_lon_min = filtered_points['Longitude'].min()
            cluster_lon_max = filtered_points['Longitude'].max()
            cluster_lat_min = filtered_points['Latitude'].min()
            cluster_lat_max = filtered_points['Latitude'].max()

            corners = [
                (cluster_lon_min, cluster_lat_min, m1 * cluster_lon_min + m2 * cluster_lat_min + c),
                (cluster_lon_min, cluster_lat_max, m1 * cluster_lon_min + m2 * cluster_lat_max + c),
                (cluster_lon_max, cluster_lat_min, m1 * cluster_lon_max + m2 * cluster_lat_min + c),
                (cluster_lon_max, cluster_lat_max, m1 * cluster_lon_max + m2 * cluster_lat_max + c)
            ]

            fault_plane_corners.append(corners)

        # Convert the list of corners to a DataFrame
        corners_df = pd.DataFrame([item for sublist in fault_plane_corners for item in sublist], columns=['Longitude', 'Latitude', 'Depth'])

        # Save to CSV
        corners_df.to_csv('fault_plane_corners_short_term_30_days.csv', index=False)
#    else:
        # Not enough events to perform K-means clustering, so just plot the events
        #ax.scatter(group['Longitude'], group['Latitude'], group['Depth'], label=f'Events - {period_str}')


# Set the spatial range for the all-months plot
        # Plot receiver stations with different color
    if draw_receiver:
        ax.scatter(receiver_geometry[2:,0], receiver_geometry[2:,1], receiver_geometry[2:,2], c='red', label=f'GM1', s=20, marker='^')
        ax.scatter(receiver_geometry[:2,0], receiver_geometry[:2,1], receiver_geometry[:2,2], c='black', label='CCS1', s=20, marker='D')
    ax.set_xlim(longitude_range)
    ax.set_ylim(latitude_range)
    ax.set_zlim(depth_range)

# Set the labels for the axes
    ax.set_xlabel('Longitude (km)', fontsize=16)
    ax.set_ylabel('Latitude (km)', fontsize=16)
    ax.set_zlabel('Depth (km)', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=10)

# Set the title for the plot
#    ax.set_title(f'Events and Clusters - All {period_str}')

# Show the legend for the plot
#ax_all.legend()

# Save the all-months plot as a PNG file
#    fig.savefig(f'Clusters_All_by_{period_str}.png')
    if save_all_periods:
        img_filename = f'Figure_example_outcomes/all_cluster_{period_length}_{period_str}.png'
        plt.savefig(img_filename)  # Save the current figure as a png image
    #plt.show()
    #plt.close()

        image_files.append(img_filename)  # Append the filename to the list

# Factor for 90% cut-off
factor = np.sqrt(6.251)
#color_all = plt.cm.jet(np.linspace(0, 1, 100))
# Go through all events
if draw_uq:
    for i in range(0,3500,20):
        # Extract mean and std for each dimension
        mean_x, mean_y, mean_z = mean[i]
        std_x, std_y, std_z = std[i] * 0.15  # Adjust for 90% cut-off
        
        # Create a grid for the ellipsoid
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        y = std_x * np.outer(np.cos(u), np.sin(v)) + mean_x
        x = std_y * np.outer(np.sin(u), np.sin(v)) + mean_y
        z = -(std_z * np.outer(np.ones(np.size(u)), np.cos(v)) + mean_z)
        
        # Plot the ellipsoid
        ax.plot_surface(x, y, z, color='b', alpha=0.3)  # alpha controls the transparency
        print('drawing UQ:',i)
if draw_catalog:
    ax.scatter(labels[:,0], labels[:,1], labels[:,2], c='blue', label=f'DD', s=3, marker='.')
    print('drawing catalog')

if draw_rolling_gif:
    print('drawing rolling gif')

    for azimuth in range(-90,90,1):
        print('azimuth:',azimuth)

        ax.view_init(azim=azimuth)
        roll_img_filename = f'Figure_example_outcomes/rolling_view/roll_3D_XZ_view_{period_length}_{azimuth}.png'
        plt.savefig(roll_img_filename)
        roll_image_files.append(roll_img_filename)  # Append the filename to the list
#if draw_rolling_gif:
# Create a gif from the png images
    images = [Image.open(filename) for filename in roll_image_files]  # Read the images into memory

    # Append images with the same base
    im = images[0]
    im.save('./Figure_example_outcomes/rolling_view/roll_all_clusters_'+str(period_length)+'.gif', save_all=True, append_images=images[1:], optimize=False, duration=100, loop=0)

plt.savefig(f'./Figure_example_outcomes/3D_view_{period_length}_orange.png')

# XY
ax.view_init(elev=90, azim=-90)
plt.savefig(f'./Figure_example_outcomes/3D_XY_view_{period_length}_orange.png')
# XZ
ax.view_init(elev=0, azim=-90)
plt.savefig(f'./Figure_example_outcomes/3D_XZ_view_{period_length}_orange.png')
# YZ
ax.view_init(elev=0, azim=0)
plt.savefig(f'./Figure_example_outcomes/3D_YZ_view_{period_length}_orange.png')
#plt.show()
#plt.savefig(f'./Figure_run8_3D_faults/3D_XY_view_{period_length}.png')


if draw_gif:
# Create a gif from the png images
    images = [Image.open(filename) for filename in image_files]  # Read the images into memory

    # Append images with the same base
    im = images[0]
    im.save('./Figure_example_outcomes/all_clusters_'+str(period_length)+'.gif', save_all=True, append_images=images[1:], optimize=False, duration=100, loop=0)

# Show the all-months plot
print(ccount)

