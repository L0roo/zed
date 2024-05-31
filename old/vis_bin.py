import os
import numpy as np
import struct
import open3d

def read_bin_velodyne(path):
    pc_list=[]
    with open(path,'rb') as f:
        content=f.read()
        pc_iter=struct.iter_unpack('ffffff',content)
        a = True
        for idx,point in enumerate(pc_iter):
            pc_list.append([point[0],point[1],point[2]])
            if a:
                print(point)
                a = False
    print(np.shape(np.asarray(pc_list,dtype=np.float32)))
    #print(np.asarray(pc_list,dtype=np.float32))
    return np.asarray(pc_list,dtype=np.float32)



def main2():
    pcd = open3d.open3d.geometry.PointCloud()
    path = "../data/bin/test_gui.bin"
    path2 = "scene0000_00.bin"
    example = read_bin_velodyne(path)
    # From numpy to Open3D
    pcd.points = open3d.open3d.utility.Vector3dVector(example)
    open3d.open3d.visualization.draw_geometries([pcd])

if __name__=="__main__":
    main2()