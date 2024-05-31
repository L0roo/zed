# Create a ZED camera object
import sys
import pyzed.sl as sl
import argparse
import os


zed = sl.Camera()

# Set SVO path for playback
input_path = "../data/HD1080_SN34783283_15-12-38.svo"
init_parameters = sl.InitParameters()
init_parameters.set_from_svo_file(input_path)

# Open the ZED
zed = sl.Camera()
err = zed.open(init_parameters)


point_cloud = sl.Mat()
i=0
while i<100:
  if zed.grab() == sl.ERROR_CODE.SUCCESS:
    # Read side by side frames stored in the SVO
    zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
    if i==12:
      point_cloud.write("data/pcd/test2.pcd")
    # Get frame count
    svo_position = zed.get_svo_position()
    i+=1
  elif zed.grab() == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
    print("SVO end has been reached. Looping back to first frame")
    break
zed.close()