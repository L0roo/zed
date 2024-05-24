import sys
import pyzed.sl as sl
import argparse
import os
import shutil
import numpy as np
import time
import struct

# Set SVO path for playback
input_path = "data/HD1080_SN34783283_15-12-38.svo"
output_base = "data/bin/multi4"
downsampling = 50 # each x frame is used

max_depth = 2.5 # in m
scale = 3 # scale up a bit bc of white screen on visualisation, big mystery
perc = 0.02 # percentage of points to keep,  higher percentage, might need higher scaling (is applied before max_depth reduction for speed)
#perc of 0.02 gives approx 30000 points
unit = 1000 # 1000 for mm, 1 for m, via .py gives in mm, gui gives m
read_color = False




deg_x = np.deg2rad(0)
deg_y = np.deg2rad(0)
deg_z = np.deg2rad(0)
rot_mat_x = np.array([[1, 0, 0], [0, np.cos(deg_x), -np.sin(deg_x)], [0, np.sin(deg_x), np.cos(deg_x)]])
rot_mat_y = np.array([[np.cos(deg_y), 0, np.sin(deg_y)], [0, 1, 0], [-np.sin(deg_y), 0, np.cos(deg_y)]])
rot_mat_z = np.array([[np.cos(deg_z), -np.sin(deg_z), 0], [np.sin(deg_z), np.cos(deg_z), 0], [0, 0, 1]])

size_list = []

zed = sl.Camera()

def pcd2bin(pcd_data, output_path):
    num_points = np.shape(pcd_data)[0]
    points = np.zeros([num_points, 6], dtype=np.float32)
    points[: , :3] = pcd_data
    points[:, :3] = points[:, :3] * (scale / unit)



    #subsample
    mask = np.random.choice([True, False], size=len(points), p=[perc, 1-perc])

    if read_color:
        #read color doesnt work yet
        for i in range(num_points):
            if mask[i]:
                pass
                #char_array = struct.unpack('BBBB', pcd_data.pc_data['rgb'][i])
                #points[i,3:] = char_array[:3]
                #print("(R,G,B,A) = {}".format(char_array))
    else:
        points_color = np.random.randint(size= (num_points, 3),low=0, high=255)
        points[:,3:] = points_color


    points = points[mask]


    mask2 = points[:,2] < max_depth*scale
    points = points[mask2]

    #rotate points


    for i in range(len(points)):
        points[i,:3]= np.matmul(rot_mat_x,points[i,:3])
        points[i,:3]= np.matmul(rot_mat_y,points[i,:3])
        points[i,:3]= np.matmul(rot_mat_z,points[i,:3])



    points = points.astype(np.float32)
    print(np.shape(points))
    size_list.append(np.shape(points)[0])
    with open(output_path, 'wb') as f:
        f.write(points.tobytes())




if os.path.exists(output_base):
    # Remove the directory and its contents
    shutil.rmtree(output_base)
    print("removed existing directory")
# Create the directory
os.makedirs(output_base)
print("created directory")
init_parameters = sl.InitParameters()
init_parameters.set_from_svo_file(input_path)

# Open the ZED
zed = sl.Camera()
err = zed.open(init_parameters)



point_cloud = sl.Mat()
i=0
while i<10000:
  if zed.grab() == sl.ERROR_CODE.SUCCESS:
    if i % downsampling == 0:
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
        output_path = output_base + "/" + str(int(i/downsampling)) + ".bin"
        point_cloud_np = point_cloud.get_data()[:, :, :3]
        point_cloud_np = point_cloud_np.reshape((-1, 3))
        pcd2bin(point_cloud_np,output_path)
        #point_cloud.write(output_path)
        print("Created file "+ str(int(i/downsampling)))
    # Get frame count
    svo_position = zed.get_svo_position()
    i+=1
  elif zed.grab() == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
    print("SVO end has been reached. "+str(int(i/downsampling)+1) +" pcd files generated")
    break
zed.close()