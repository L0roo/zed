import numpy as np
from pypcd import pypcd

input_file = 'test1.pcd'
output_path = "data/bin/"+input_file[:-3] + 'bin'

pcd_data = pypcd.PointCloud.from_path("data/pcd/"+input_file)
points = np.zeros([pcd_data.width, 3], dtype=np.float32)
points[:, 0] = pcd_data.pc_data['x'].copy()
points[:, 1] = pcd_data.pc_data['y'].copy()
points[:, 2] = pcd_data.pc_data['z'].copy()
#points[:, 3] = pcd_data.pc_data['intensity'].copy().astype(np.float32)
#mean = np.mean(points[:,:3], axis=0)
#std_dev = np.std(points[:,:3], axis=0)
points[:,:3] = points[:,:3] / 1000 # do if .pcd via python not ZED depth viewer (scale)



points_color = np.random.randint(size= (pcd_data.width, 3),low=0, high=255)
#points_color = points_color.astype(np.uint8)
#points_color = points_color + 2570

points = np.hstack((points,points_color))

# Normalize each column
#points[:,:3] = (points[:,:3] - mean) / std_dev


#subsample
perc = 0.01
mask = np.random.choice([True, False], size=len(points), p=[perc, 1-perc])
points = points[mask]

points = points.astype(np.float32)
print(points)
print(np.shape(points))
with open(output_path, 'wb') as f:
    f.write(points.tobytes())