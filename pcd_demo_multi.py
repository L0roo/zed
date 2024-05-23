# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os
from argparse import ArgumentParser
import numpy as np
import time

from mmengine.logging import print_log

from mmdet3d.apis import LidarDet3DInferencer

# needs to be run in mmdetection env


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('pcd_folder', help='Point cloud file folder in .bin (str)')
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
    # TODO: Support inference of point cloud numpy file.
    init_args, call_args = parse_args()

    inferencer = LidarDet3DInferencer(**init_args)
    '''
    inferencer(**call_args)

    if call_args['out_dir'] != '' and not (call_args['no_save_vis']
                                           and call_args['no_save_pred']):
        print_log(
            f'results have been saved at {call_args["out_dir"]}',
            logger='current')

    '''
    input_base = call_args.pop('pcd_folder')

    num_files = len(os.listdir(input_base))
    print("Amount of files: " + str(num_files))
    for i in range(num_files):
        input_path = input_base + "/" + str(i) + ".bin"
        point_cloud = np.fromfile(input_path, dtype=np.float32)
        point_cloud = point_cloud.reshape(-1, 6)

        call_args['inputs'] = dict(points=point_cloud)
        inferencer(**call_args)
        print("showing file "+str(i))
        if call_args['out_dir'] != '' and not (call_args['no_save_vis']
                                               and call_args['no_save_pred']):
            print_log(
                f'results have been saved at {call_args["out_dir"]}',
                logger='current')


if __name__ == '__main__':
    main()
