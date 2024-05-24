# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os
from argparse import ArgumentParser
import numpy as np
import time
import sys
import pyzed.sl as sl
import argparse
import shutil
import struct

from mmengine.logging import print_log

from mmdet3d.apis import LidarDet3DInferencer

# needs to be run in mmdetection env

'''
To run this file: 
mmdetection3d environment with minkowski engine
download chosen model, set path accordingly in example below
adjust parameters in this file
example:
python demo_svo.py "data/HD1080_SN34783283_15-12-38.svo" /home/pdz/PythonProjects/mmdetection3d/mmdetection3d/projects/TR3D/configs/tr3d_1xb16_scannet-3d-18class.py /home/pdz/PythonProjects/mmdetection3d/mmdetection3d/pdz/tr3d_1xb16_scannet-3d-18class.pth --pred-score-thr=0.09 --wait-time=0.0 --show

important: if no prediction score is above the threshold, no image will be displayed

'''



downsampling = 5 # each x frame is used

max_depth = 2.0 # in m (runs only on z coordinate)
scale = 3 # scale up a bit gives higher pred scores
ppc = 10000 #points per cloud, makes perc obsolete
unit = 1000 # 1000 for mm, 1 for m, via .py gives in mm, gui gives m
read_color = True




deg_x = np.deg2rad(0)
deg_y = np.deg2rad(0)
deg_z = np.deg2rad(0)
rot_mat_x = np.array([[1, 0, 0], [0, np.cos(deg_x), -np.sin(deg_x)], [0, np.sin(deg_x), np.cos(deg_x)]])
rot_mat_y = np.array([[np.cos(deg_y), 0, np.sin(deg_y)], [0, 1, 0], [-np.sin(deg_y), 0, np.cos(deg_y)]])
rot_mat_z = np.array([[np.cos(deg_z), -np.sin(deg_z), 0], [np.sin(deg_z), np.cos(deg_z), 0], [0, 0, 1]])

size_list = []
time_list = []

def pcd2bin(pcd_data):
    num_points = np.shape(pcd_data)[0]
    points = np.zeros([num_points, 6], dtype=np.float32)
    points[: , :3] = pcd_data[:,:3]


    mask2 = points[:,2] < max_depth*unit
    points = points[mask2]

    #subsample
    perc = ppc / len(points)
    mask = np.random.choice([True, False], size=len(points), p=[perc, 1-perc])
    points = points[mask]

    points[:, :3] = points[:, :3] * (scale / unit)
    if read_color:
        for i in range(len(points)):
            char_array = struct.unpack('BBBB', pcd_data[i,3])
            points[i,3:] = char_array[:3]
    else:
        points_color = np.random.randint(size= (len(points), 3),low=0, high=255)
        points[:,3:] = points_color






    #rotate points
    for i in range(len(points)):
        points[i,:3]= np.matmul(rot_mat_x,points[i,:3])
        points[i,:3]= np.matmul(rot_mat_y,points[i,:3])
        points[i,:3]= np.matmul(rot_mat_z,points[i,:3])



    points = points.astype(np.float32)
    print(np.shape(points))
    size_list.append(np.shape(points)[0])
    return points



def parse_args():
    parser = ArgumentParser()
    parser.add_argument('svo_file', help='svo_file_path as str')
    parser.add_argument('model', help='Config file')
    parser.add_argument('weights', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--pred-score-thr',
        type=float,
        default=0.3,
        help='bbox score threshold')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='outputs',
        help='Output directory of prediction and visualization results.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Show online visualization results')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=-1,
        help='The interval of show (s). Demo will be blocked in showing'
        'results, if wait_time is -1. Defaults to -1.')
    parser.add_argument(
        '--no-save-vis',
        action='store_true',
        help='Do not save detection visualization results')
    parser.add_argument(
        '--no-save-pred',
        action='store_true',
        help='Do not save detection prediction results')
    parser.add_argument(
        '--print-result',
        action='store_true',
        help='Whether to print the results.')
    call_args = vars(parser.parse_args())

    #call_args['inputs'] = dict(points=call_args.pop('pcd_folder'))

    if call_args['no_save_vis'] and call_args['no_save_pred']:
        call_args['out_dir'] = ''

    init_kws = ['model', 'weights', 'device']
    init_args = {}
    for init_kw in init_kws:
        init_args[init_kw] = call_args.pop(init_kw)

    # NOTE: If your operating environment does not have a display device,
    # (e.g. a remote server), you can save the predictions and visualize
    # them in local devices.
    if os.environ.get('DISPLAY') is None and call_args['show']:
        print_log(
            'Display device not found. `--show` is forced to False',
            logger='current',
            level=logging.WARNING)
        call_args['show'] = False

    return init_args, call_args


def main():
    init_args, call_args = parse_args()

    inferencer = LidarDet3DInferencer(**init_args)

    input_path = call_args.pop('svo_file')
    init_parameters = sl.InitParameters()
    init_parameters.set_from_svo_file(input_path)

    # Open the ZED
    zed = sl.Camera()
    err = zed.open(init_parameters)

    point_cloud = sl.Mat()
    i = 0
    start_time = time.time()
    while i < 10000:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            if i % downsampling == 0:
                st = time.time()
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
                point_cloud_np = point_cloud.get_data()[:, :, :4]
                point_cloud_np = point_cloud_np.reshape((-1, 4))
                points = pcd2bin(point_cloud_np)
                call_args['inputs'] = dict(points=points)
                inferencer.num_visualized_imgs = i  # can not generate output files else
                inferencer(**call_args)
                print("showing frame " + str(int(i/downsampling+1)))
                if call_args['out_dir'] != '' and not (call_args['no_save_vis']
                                                       and call_args['no_save_pred']):
                    print_log(
                        f'results have been saved at {call_args["out_dir"]}',
                        logger='current')
                et = time.time()
                time_dif = round(et-st,4)
                time_list.append(time_dif)
                print("Inference on frame in "+str(time_dif)+" s")
            i += 1
        elif zed.grab() == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            end_time = time.time()
            time_dif = end_time - start_time
            time_pf = round(time_dif / (int(i / downsampling) + 1), 4)
            print("Average time per frame: " + str(time_pf) + " s")
            print("Average time per frame (time on used frames only): " + str(round(np.mean(time_list),4)) + " s")
            print("Average FPS: "+str(round(1/time_pf,4)))
            break
    zed.close()
    print("Average points per cloud: " + str(round(np.mean(size_list), 0)))











if __name__ == '__main__':
    main()
