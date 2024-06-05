import sys
import pyzed.sl as sl
import argparse
import os
import shutil
import numpy as np
import time
import struct
import open3d as o3d

# Set SVO path for playback
input_path = "data/small_object_svo/HD2K_small2.svo2"
output_base = "data/bin/small2"

#make sure parameters are the same
downsampling = 5 # each x frame is used
max_frames = 50

max_depth = 100.6 # in m (runs only on z coordinate) set above 50 to disable
max_dist = 1.5
scale = 4.0 # scale up a bit gives higher pred scores
ppc = 40000 #points per cloud, makes perc obsolete
filter_scale = 1.5 #need to initialy sample more points to compensate for points filtered out
read_color = True
rotate = True
outlier_removal = True
depth_filter = True


hrot_matrix_1=np.array([[-0.2472, 0.7015, -0.6684, 0.8397],
 [-0.9609, -0.0887, 0.2623, -0.2136],
 [0.1247, 0.7071, 0.6960, -0.7072],
 [0.0000, 0.0000, 0.0000, 1.0000]])

hrot_matrix = np.array([[0.8580, -0.1040, 0.5030, -0.5229],
 [0.5100, 0.2892, -0.8101, 0.8699],
 [-0.0612, 0.9516, 0.3012, -0.6123],
 [0.0000, 0.0000, 0.0000, 1.0000]])



np.random.seed(427)

unit = 1000 # 1000 for mm, 1 for m, via .py gives in mm, gui gives m (use 1000)

size_list = []

def filter_points(point_cloud):
    points = point_cloud.get_data()[:, :, :4]
    points = points.reshape((-1, 4))



    # depth/dist filter
    st_depth = time.time()
    if depth_filter:
        if max_depth < 50:
            mask2 = points[:,2] < max_depth*unit
            points = points[mask2]
        if max_dist < 50:
            mask3 = np.linalg.norm(points[:,:3], axis=1) < max_dist*unit
            points = points[mask3]
    et_depth = time.time()


    # subsample
    st_sub = time.time()
    perc = ppc*filter_scale / len(points)
    if perc > 1: perc = 1
    mask = np.random.choice([True, False], size=len(points), p=[perc, 1-perc])
    points = points[mask]
    et_sub = time.time()

    if outlier_removal:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:,:3])
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=40,std_ratio=1.5)
        points = points[ind]








    # rotate with matrix
    st_quat = time.time()
    if rotate:
        ones = np.ones((points.shape[0], 1))
        homogeneous_points = np.hstack((points[:,:3], ones))
        hom_transformed = (hrot_matrix @ homogeneous_points.T).T
        points[:,:3] = hom_transformed[:, :3]
    et_quat = time.time()

    # rotate points
    # for i in range(len(points)):
    #    points[i,:3]= np.matmul(rot_mat_x,points[i,:3])
    #    points[i,:3]= np.matmul(rot_mat_y,points[i,:3])
    #    points[i,:3]= np.matmul(rot_mat_z,points[i,:3])






    # scale





    st_color = time.time()
    if read_color:
        col_list = []
        for i in range(len(points)):
            col_list.append(struct.unpack('BBBB', points[i, 3])[:3])
        points = np.hstack((points[:, :3], np.array(col_list)))

    else:
        points_color = np.random.randint(size=(len(points), 3), low=0, high=255)
        points = np.hstack((points[:, :3], points_color))
    et_color = time.time()

    # scale points
    points[:, :3] = points[:, :3] * (scale / unit)

    points = points.astype(np.float32)
    print(np.shape(points))
    size_list.append(np.shape(points)[0])



    return points



if os.path.exists(output_base):
    # Remove the directory and its contents
    shutil.rmtree(output_base)
    print("removed existing directory")
# Create the directory
os.makedirs(output_base)
print("created directory")
init_parameters = sl.InitParameters()
init_parameters.set_from_svo_file(input_path)
init_parameters.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS

# Open the ZED
zed = sl.Camera()
err = zed.open(init_parameters)

point_cloud = sl.Mat()
i=0
st = time.time()
while i<10000:
  if zed.grab() == sl.ERROR_CODE.SUCCESS:
    if i % downsampling == 0:
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
        output_path = output_base + "/" + str(int(i/downsampling)) + ".bin"
        points = filter_points(point_cloud)
        size_list.append(np.shape(points)[0])
        with open(output_path, 'wb') as f:
            f.write(points.tobytes())
        print("Created file "+ str(int(i/downsampling)))
    # Get frame count
    i+=1
  elif zed.grab() == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
    print("SVO end has been reached. "+str(int(i/downsampling)+1) +" pcd files generated")
    et = time.time()
    time_dif = et - st
    time_pf = round(time_dif / (int(i/downsampling)+1), 4)
    print("Time per file: " + str(time_pf) + " s")
    break
zed.close()
print("Average points per cloud: "+ str(round(np.mean(size_list),0)))
