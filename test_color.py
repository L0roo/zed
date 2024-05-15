import struct
import numpy as np
from pypcd import pypcd

input_file = 'test1.pcd'

pcd_data = pypcd.PointCloud.from_path("data/pcd/"+input_file)


#err, point_cloud_value = point_cloud.get_data(x,y)    # Get your point cloud value at pixel (x,y)
#packed = struck.pack('f', point_cloud_value[3])
for i in range(pcd_data.width):
    char_array = struct.unpack('BBBB', pcd_data.pc_data['rgb'][i])
    print("(R,G,B,A) = {}".format(char_array))