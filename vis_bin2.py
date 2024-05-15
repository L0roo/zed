import torch
import numpy as np

from mmdet3d.visualization import Det3DLocalVisualizer
from mmdet3d.structures import LiDARInstance3DBoxes

points = np.fromfile('data/bin/test1.bin', dtype=np.float32)
points = points.reshape(-1, 6)
visualizer = Det3DLocalVisualizer()
# set point cloud in visualizer
visualizer.set_points(points)

bboxes_3d = LiDARInstance3DBoxes(
    torch.tensor([[-0.1765555590391159,
-1.4405971765518188, -5.0545244216918945, 1.6508995294570923, 0.6726411581039429, 0.3388897478580475, 0.0]]))
# Draw 3D bboxes
#visualizer.draw_bboxes_3d(bboxes_3d)


visualizer.show()