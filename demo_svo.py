import logging
import os
from argparse import ArgumentParser
import numpy as np
import time
import pyzed.sl as sl
import struct
import open3d as o3d

from mmengine.logging import print_log

from mmdet3d.apis import LidarDet3DInferencer

import json


'''
To run this file: 
mmdetection3d environment with minkowski engine (tricky, see readme)
download chosen model and configuration, set path accordingly in example below
svo file from zed camer, also set path
adjust parameters in this file
example:
python demo_svo.py "data/HD1080_SN34783283_15-12-38.svo" /home/pdz/PythonProjects/mmdetection3d/mmdetection3d/projects/TR3D/configs/tr3d_1xb16_scannet-3d-18class.py /home/pdz/PythonProjects/mmdetection3d/mmdetection3d/pdz/tr3d_1xb16_scannet-3d-18class.pth --pred-score-thr=0.09 --wait-time=0.0 --show
or alternatively
python demo_svo.py "data/small_object_svo/HD2K_SN38580376_small_obj.svo2" /home/pdz/PythonProjects/mmdetection3d/mmdetection3d/configs/fcaf3d/fcaf3d_2xb8_scannet-3d-18class.py /home/pdz/PythonProjects/mmdetection3d/mmdetection3d/pdz/fcaf3d_8x2_scannet-3d-18class_20220805_084956.pth --pred-score-thr=0.09 --show 

important: if no prediction score is above the threshold no image will be displayed, set wait time to 0 to change frames automatically

TR3D scannnet: python demo_svo.py "data/small_object_svo/HD2K_small2.svo2" /home/pdz/PythonProjects/mmdetection3d/mmdetection3d/projects/TR3D/configs/tr3d_1xb16_scannet-3d-18class.py /home/pdz/PythonProjects/mmdetection3d/mmdetection3d/pdz/tr3d_1xb16_scannet-3d-18class.pth --pred-score-thr=0.07 --show 
FCAF3d: python demo_svo.py "data/small_object_svo/HD2K_small2.svo2" /home/pdz/PythonProjects/mmdetection3d/mmdetection3d/configs/fcaf3d/fcaf3d_2xb8_scannet-3d-18class.py /home/pdz/PythonProjects/mmdetection3d/mmdetection3d/pdz/fcaf3d_8x2_scannet-3d-18class_20220805_084956.pth --pred-score-thr=0.07 --show 
TR3D s3dis model doesn't match exacltly: python demo_svo.py "data/small_object_svo/HD2K_small2.svo2" /home/pdz/PythonProjects/mmdetection3d/mmdetection3d/projects/TR3D/configs/tr3d_1xb16_s3dis-3d-5class.py /home/pdz/PythonProjects/mmdetection3d/mmdetection3d/pdz/tr3d_1xb16_s3dis-3d-5class.pth --pred-score-thr=0.85 --show 

'''



downsampling = 5 # each x frame is used (takes some time to skip frames)
max_frames = 1# amount of frames till end

max_depth = 100.6 # in m (runs only on z coordinate) set above 50 to disable
max_dist = 1.5 # in m, set above 50 to disable
scale = 7.0 # scale up a bit gives higher pred scores since original net was trained on larger scenes
ppc = 60000 #points per cloud
filter_scale = 1.0 #need to initialy sample more points to compensate for points filtered out (small2: 5.4 for dist 1.5,no outlier)
read_color = True # read color or choose random colors
rotate = True # rotate and translate according to homogenous transform matrix hrot_matrix
outlier_removal = True # have open3d remove statistical outliers
depth_filter = True # have depth filter active
out_nb = 20 # parameters for outlier removal
out_std = 1.0

vis_col = True # if true doesn't run infernece but shows with color
json_in = 'data/json/small2_tr3d_7.json'
json_in_2 = 'outputs/preds/00000000.json'
bb_thr = 0.07


hrot_matrix_1=np.array([[-0.2472, 0.7015, -0.6684, 0.8397],
 [-0.9609, -0.0887, 0.2623, -0.2136],
 [0.1247, 0.7071, 0.6960, -0.7072],
 [0.0000, 0.0000, 0.0000, 1.0000]])

hrot_matrix = np.array([[0.8580, -0.1040, 0.5030, -0.5229],
 [0.5100, 0.2892, -0.8101, 0.8699],
 [-0.0612, 0.9516, 0.3012, -0.6123],
 [0.0000, 0.0000, 0.0000, 1.0000]])


np.random.seed(427) # fix random seed for repeatability

unit = 1000

size_list = [] # size of point clouds statistics
time_list = [] # time lists for time statistics
inferencer_time_list = []
filter_time_list = []
depth_filter_time_list = []
sub_filter_time_list = []
color_filter_time_list = []
pos_time_list = []
quat_time_list = []
outlier_time_list = []
zed_time_list = []




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




def filter_points(point_cloud):
    '''

    :param point_cloud: Pointcloud from zed.retrieve_measure
    :return: filtered and transformed pointcloud  in numpy array
    '''
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


    # oulier removal
    st_out = time.time()
    if outlier_removal:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:,:3])
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=out_nb,std_ratio=out_std)
        points = points[ind]
    et_out = time.time()




    # rotate with matrix
    st_quat = time.time()
    if rotate:
        ones = np.ones((points.shape[0], 1))
        homogeneous_points = np.hstack((points[:,:3], ones))
        hom_transformed = (hrot_matrix @ homogeneous_points.T).T
        points[:,:3] = hom_transformed[:, :3]
    et_quat = time.time()

    # get colors
    st_color = time.time()
    if read_color:
        col_list = []
        for i in range(len(points)):
            col_list.append(struct.unpack('BBBB', points[i, 3])[:3])
        points = np.hstack((points[:, :3], np.array(col_list)))

    else:
        # random colors
        points_color = np.random.randint(size=(len(points), 3), low=0, high=255)
        points = np.hstack((points[:, :3], points_color))
    et_color = time.time()

    # scale points
    points[:, :3] = points[:, :3] * (scale / unit)

    points = points.astype(np.float32)
    print(np.shape(points))
    size_list.append(np.shape(points)[0])

    depth_filter_time_list.append(et_depth-st_depth)
    sub_filter_time_list.append(et_sub-st_sub)
    color_filter_time_list.append(et_color-st_color)
    quat_time_list.append(et_quat-st_quat)
    outlier_time_list.append((et_out-st_out))

    return points


def main():
    init_args, call_args = parse_args()

    inferencer = LidarDet3DInferencer(**init_args)

    input_path = call_args.pop('svo_file')
    init_parameters = sl.InitParameters()
    init_parameters.set_from_svo_file(input_path)
    init_parameters.depth_mode = sl.DEPTH_MODE.NEURAL # NEURAL_PLUS better but slower than NEURAL

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
                st_zed = time.time()
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
                et_zed = time.time()

                st_filter = time.time()
                call_args['inputs'] = dict(points=filter_points(point_cloud))
                et_filter = time.time()

                st_inferencer = time.time()
                inferencer.num_visualized_imgs = i  # can not generate output files else

                if vis_col:
                    vis = o3d.visualization.Visualizer()
                    vis.create_window()

                    pcd = o3d.geometry.PointCloud()
                    points = filter_points(point_cloud)
                    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
                    pcd.colors = o3d.utility.Vector3dVector(points[:, 3:] / 256)

                    f = open(json_in)
                    data = json.load(f)

                    scores = data['scores_3d']
                    bboxes = data['bboxes_3d']

                    # Filter bounding boxes based on the score threshold
                    #filtered_bboxes = [bbox for score, bbox in zip(scores, bboxes) if score > bb_thr]
                    filtered_bboxes = []
                    for i in range(len(scores)):
                        if scores[i] > bb_thr:
                            filtered_bboxes.append(bboxes[i])



                    # o3d.visualization.draw_geometries([pcd])
                    vis.add_geometry(pcd)
                    # Output the filtered bounding boxes
                    for bbox in filtered_bboxes:


                        center = np.array([bbox[0], bbox[1], bbox[2]])
                        length = np.array([bbox[3], bbox[4], bbox[5]])

                        min_bound = center - length /2
                        max_bound = center + length/2

                        aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
                        aabb.color = (1, 0, 0)
                        vis.add_geometry(aabb)
                        #aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound=max_bound, max_bound=min_bound)
                        #aabb.color = (1, 0, 0)
                        #vis.add_geometry(aabb)
                    #opt = vis.get_render_option()
                    #opt.line_width = 5.0

                    vis.run()
                    vis.destroy_window()




                else:
                    inferencer(**call_args)
                et_inferencer = time.time()

                inferencer_time_list.append(et_inferencer-st_inferencer)
                filter_time_list.append(et_filter-st_filter)
                zed_time_list.append(et_zed-st_zed)

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
    time_pf = round((time_dif / frame_counter) * 1000, 1)
    print("Average time per frame total: " + str(time_pf) + " ms")
    print("Average time per frame (time on used frames only): " + str(round(np.mean(time_list)*1000, 1)) + " ms")
    print("Average time per frame (zed retrieve): " + str(round(np.mean(zed_time_list)*1000, 1)) + " ms")
    #print("Average time per frame (svo pos change): " + str(round(np.mean(pos_time_list), 1) * 1000) + " ms")
    print("Average time per frame (inference): " + str(round(np.mean(inferencer_time_list)*1000, 1)) + " ms")
    print("Average time per frame (transform pc): " + str(round(np.mean(filter_time_list)*1000, 1)) + " ms")
    print("Average time per frame (depth filter): " + str(round(np.mean(depth_filter_time_list)*1000, 1)) + " ms")
    print("Average time per frame (subsample): " + str(round(np.mean(sub_filter_time_list)*1000, 1)) + " ms")
    print("Average time per frame (outlier removal): " + str(round(np.mean(outlier_time_list)*1000, 1)) + " ms")
    print("Average time per frame (color): " + str(round(np.mean(color_filter_time_list)*1000, 1)) + " ms")
    #print("Average time per frame (quat rotation): " + str(round(np.mean(quat_time_list), 1) * 1000) + " ms")
    print("Average FPS: " + str(round(1000 / time_pf, 1)))
    print("Average points per cloud: " + str(round(np.mean(size_list), 0)))
    print("Times: "+ str(time_pf) + ","+str(round(np.mean(time_list) * 1000, 1))+","
          +str(round(np.mean(inferencer_time_list) * 1000, 1)) +","
          + str(round(np.mean(filter_time_list) * 1000, 1)) +","
          + str(round(np.mean(depth_filter_time_list) * 1000, 1))+","+str(round(np.mean(sub_filter_time_list) * 1000, 1)) +","
          + str(round(np.mean(outlier_time_list) * 1000, 1)) + ","+ str(round(np.mean(color_filter_time_list) * 1000, 1)))











if __name__ == '__main__':
    main()
