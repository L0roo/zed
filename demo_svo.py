# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os
from argparse import ArgumentParser
import numpy as np
import time
import pyzed.sl as sl
import struct

from mmengine.logging import print_log

from mmdet3d.apis import LidarDet3DInferencer


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
max_frames = 50

max_depth = 100.0 # in m (runs only on z coordinate) set above 50 to disable
max_dist = 4.0
scale = 1.0 # scale up a bit gives higher pred scores
ppc = 70000 #points per cloud, makes perc obsolete
read_color = True
rotate = True


hrot_matrix_1=np.array([[-0.2472, 0.7015, -0.6684, 0.8397],
 [-0.9609, -0.0887, 0.2623, -0.2136],
 [0.1247, 0.7071, 0.6960, -0.7072],
 [0.0000, 0.0000, 0.0000, 1.0000]])

hrot_matrix_2 = np.array([[0.8580, -0.1040, 0.5030, -0.5229],
 [0.5100, 0.2892, -0.8101, 0.8699],
 [-0.0612, 0.9516, 0.3012, -0.6123],
 [0.0000, 0.0000, 0.0000, 1.0000]])


#not used at the moment
deg_x = np.deg2rad(0)
deg_y = np.deg2rad(0)
deg_z = np.deg2rad(0)
rot_mat_x = np.array([[1, 0, 0], [0, np.cos(deg_x), -np.sin(deg_x)], [0, np.sin(deg_x), np.cos(deg_x)]])
rot_mat_y = np.array([[np.cos(deg_y), 0, np.sin(deg_y)], [0, 1, 0], [-np.sin(deg_y), 0, np.cos(deg_y)]])
rot_mat_z = np.array([[np.cos(deg_z), -np.sin(deg_z), 0], [np.sin(deg_z), np.cos(deg_z), 0], [0, 0, 1]])
np.random.seed(427)

unit = 1000 # 1000 for mm, 1 for m, via .py gives in mm, gui gives m (use 1000)
size_list = []
time_list = []
inferencer_time_list = []
filter_time_list = []
depth_filter_time_list = []
sub_filter_time_list = []
color_filter_time_list = []
pos_time_list = []
quat_time_list = []




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




def pcd2bin(point_cloud):
    points = point_cloud.get_data()[:, :, :4]
    points = points.reshape((-1, 4))

    st_depth = time.time()
    if max_depth < 50:
        mask2 = points[:,2] < max_depth*unit
        points = points[mask2]
    if max_dist < 50:
        mask3 = np.linalg.norm(points[:,:3], axis=1) < max_dist*unit
        points = points[mask3]
    et_depth = time.time()

    #subsample
    st_sub = time.time()
    perc = ppc / len(points)
    mask = np.random.choice([True, False], size=len(points), p=[perc, 1-perc])
    points = points[mask]
    et_sub = time.time()

    st_color = time.time()
    points[:, :3] = points[:, :3] * (scale / unit)
    if read_color:
        col_list = []
        for i in range(len(points)):
            col_list.append(struct.unpack('BBBB', points[i,3])[:3])
        points = np.hstack((points[:,:3],np.array(col_list)))

    else:
        points_color = np.random.randint(size= (len(points), 3),low=0, high=255)
        points = np.hstack((points[:,:3],points_color))
    et_color = time.time()




    # rotate wit quaternion matrix
    st_quat = time.time()
    if rotate:
        ones = np.ones((points.shape[0], 1))
        homogeneous_points = np.hstack((points[:,:3], ones))
        hom_transformed = (hrot_matrix_1 @ homogeneous_points.T).T

        # Apply the quaternion rotation, seems to do the same thing as above
        #transformed_point_cloud_homogeneous = homogeneous_points @ hrot_matrix_1.T

        # Convert back to Cartesian coordinates (Nx3 array)
        points[:,:3] = hom_transformed[:, :3]
    et_quat = time.time()


    #rotate points
    #for i in range(len(points)):
    #    points[i,:3]= np.matmul(rot_mat_x,points[i,:3])
    #    points[i,:3]= np.matmul(rot_mat_y,points[i,:3])
    #    points[i,:3]= np.matmul(rot_mat_z,points[i,:3])



    points = points.astype(np.float32)
    print(np.shape(points))
    size_list.append(np.shape(points)[0])

    depth_filter_time_list.append(et_depth-st_depth)
    sub_filter_time_list.append(et_sub-st_sub)
    color_filter_time_list.append(et_color-st_color)
    quat_time_list.append(et_quat-st_quat)

    return points


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
    i=0
    frame_counter = 0
    start_time = time.time()
    while frame_counter < max_frames:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            if i % downsampling == 0:
                st = time.time()
                frame_counter +=1
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
                st_filter = time.time()
                call_args['inputs'] = dict(points=pcd2bin(point_cloud))
                et_filter = time.time()
                inferencer.num_visualized_imgs = i  # can not generate output files else
                st_inferencer = time.time()
                inferencer(**call_args)
                et_inferencer = time.time()

                inferencer_time_list.append(et_inferencer-st_inferencer)
                filter_time_list.append(et_filter-st_filter)

                print("showing frame " + str(frame_counter))
                if call_args['out_dir'] != '' and not (call_args['no_save_vis']
                                                       and call_args['no_save_pred']):
                    print_log(
                        f'results have been saved at {call_args["out_dir"]}',
                        logger='current')
                et = time.time()
                time_dif = round(et-st,4)
                time_list.append(time_dif)
                print("Inference on frame in "+str(time_dif)+" s")
                #st_pos = time.time()
                #i += 1
                #zed.set_svo_position(i*downsampling)
                #et_pos = time.time()
                #pos_time_list.append(et_pos-st_pos)
            i+=1

        elif zed.grab() == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            #zed.set_svo_position(0)
            break
    zed.close()
    end_time = time.time()
    time_dif = end_time - start_time
    time_pf = round((time_dif / frame_counter), 4) * 1000
    print("Average time per frame total: " + str(time_pf) + " ms")
    print("Average time per frame (time on used frames only): " + str(round(np.mean(time_list), 4) * 1000) + " ms")
    #print("Average time per frame (svo pos change): " + str(round(np.mean(pos_time_list), 4) * 1000) + " ms")
    print("Average time per frame (inference): " + str(round(np.mean(inferencer_time_list), 4) * 1000) + " ms")
    print("Average time per frame (transform pc): " + str(round(np.mean(filter_time_list), 4) * 1000) + " ms")
    print("Average time per frame (depth filter): " + str(round(np.mean(depth_filter_time_list), 4) * 1000) + " ms")
    print("Average time per frame (subsample): " + str(round(np.mean(sub_filter_time_list), 4) * 1000) + " ms")
    print("Average time per frame (color): " + str(round(np.mean(color_filter_time_list), 4) * 1000) + " ms")
    print("Average time per frame (quat rotation): " + str(round(np.mean(quat_time_list), 4) * 1000) + " ms")
    print("Average FPS: " + str(round(1000 / time_pf, 4)))
    print("Average points per cloud: " + str(round(np.mean(size_list), 0)))











if __name__ == '__main__':
    main()
