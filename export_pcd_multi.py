import sys
import pyzed.sl as sl
import argparse
import os
import shutil


zed = sl.Camera()

# Set SVO path for playback
input_path = "data/HD1080_SN34783283_15-12-38.svo"
output_base = "data/pcd/multi3"
downsampling = 5 # each x frame is used

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
        output_path = output_base + "/" + str(int(i/downsampling)) + ".pcd"
        point_cloud.write(output_path)
        print("Created file "+ str(int(i/downsampling)))
    # Get frame count
    svo_position = zed.get_svo_position()
    i+=1
  elif zed.grab() == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
    print("SVO end has been reached. "+str(int(i/downsampling)+1) +" pcd files generated")
    break
zed.close()