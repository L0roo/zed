import numpy as np
from pypcd import pypcd
import struct

input_file = 'test2.pcd'
output_path = "data/bin/"+input_file[:-3] + 'bin'

max_depth = 1.2 # in m
scale = 5 # scale up a bit bc of white screen on visualisation, big mystery
perc = 0.01 # percentage of points to keep,  higher percentage, might need higher scaling
unit = 1000 # 1000 for m, 1 for mm, via .py gives in mm, gui gives m

deg_x = np.deg2rad(0)
deg_y = np.deg2rad(0)
deg_z = np.deg2rad(0)

pcd_data = pypcd.PointCloud.from_path("data/pcd/"+input_file)
points = np.zeros([pcd_data.width, 6], dtype=np.float32)
points[:, 0] = pcd_data.pc_data['x'].copy()
points[:, 1] = pcd_data.pc_data['y'].copy()
points[:, 2] = pcd_data.pc_data['z'].copy()
points[:,:3] = points[:,:3] *(scale / unit)# scaling makes the visualization work somehow, also has a big influence on bboxes



#random color, reduce dim above from 6 to 3
#points_color = np.random.randint(size= (pcd_data.width, 3),low=0, high=255)
#points_color = points_color.astype(np.uint8)
#points_color = points_color + 2570

#points = np.hstack((points,points_color))




#subsample

mask = np.random.choice([True, False], size=len(points), p=[perc, 1-perc])



for i in range(pcd_data.width):
    if mask[i]:
        char_array = struct.unpack('BBBB', pcd_data.pc_data['rgb'][i])
        points[i,3:] = char_array[:3]
        #print("(R,G,B,A) = {}".format(char_array))


points = points[mask]


mask2 = points[:,2] < max_depth*scale
points = points[mask2]

#rotate points

rot_mat_x = np.array([[1, 0, 0],[0, np.cos(deg_x), -np.sin(deg_x)],[0, np.sin(deg_x), np.cos(deg_x)]])
rot_mat_y = np.array([[np.cos(deg_y), 0, np.sin(deg_y)],[0, 1, 0],[-np.sin(deg_y), 0, np.cos(deg_y)]])
rot_mat_z = np.array([[np.cos(deg_z), -np.sin(deg_z), 0],[np.sin(deg_z), np.cos(deg_z), 0],[0, 0, 1]])
for i in range(len(points)):
    points[i,:3]= np.matmul(rot_mat_x,points[i,:3])
    points[i,:3]= np.matmul(rot_mat_y,points[i,:3])
    points[i,:3]= np.matmul(rot_mat_z,points[i,:3])



points = points.astype(np.float32)
print(points)
print(np.shape(points))
with open(output_path, 'wb') as f:
    f.write(points.tobytes())