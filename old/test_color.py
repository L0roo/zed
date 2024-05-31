import struct
import numpy as np
from pypcd import pypcd

input_file = 'test1.pcd'

pcd_data = pypcd.PointCloud.from_path("data/pcd/"+input_file)
points = np.zeros([pcd_data.width, 3], dtype=np.float32)

perc = 0.01
mask = np.random.choice([True, False], size=len(points), p=[perc, 1-perc])
points = points[mask]



for i in range(pcd_data.width):
    if mask[i]:
        char_array = struct.unpack('BBBB', pcd_data.pc_data['rgb'][i])
        print("(R,G,B,A) = {}".format(char_array)+" i:"+str(i))