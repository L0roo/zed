# Object Detection in 3D using point clouds obtained from ZED Stereocamera
## .py files to use
- demo_svo.py runs inference on one svo files directly
- demo_svo_multi2.py runs inference on two svo files where it fuses the two point clouds
- demo_live.py runs inference directly on connected camera


## Installation
Installation of packages is a bit complicated and messy. I followed the installation instructions from mmdetection3d and MinkowskiEngine, but had to change quiet some things.
```
Ubuntu 22.04
Python 3.8
CUDA 12.1
pytorch 2.3.0
DO:
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -U openmim
mim install mmengine
mim install 'mmcv>=2.0.0rc4'
mim install 'mmdet>=3.0.0'

git clone https://github.com/open-mmlab/mmdetection3d.git -b dev-1.
cd mmdetection3d
pip install -v -e .


Minkowski:
sudo apt install build-essential python3-dev libopenblas-dev
pip install ninja
pip install setuptools==59.8.0
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine

Now you need to add some lines to some files:
1 - .../MinkowskiEngine/src/convolution_kernel.cuh
Add header:
#include <thrust/execution_policy.h>
2 - .../MinkowskiEngine/src/coordinate_map_gpu.cu
Add headers:
#include <thrust/unique.h>
#include <thrust/remove.h>
3 - .../MinkowskiEngine/src/spmm.cu
Add headers:
#include <thrust/execution_policy.h>
#include <thrust/reduce.h> 
#include <thrust/sort.h>
4 - .../MinkowskiEngine/src/3rdparty/concurrent_unordered_map.cuh
Add header:
#include <thrust/execution_policy.h>
5 - .../MinkowskiEngine/src/convolution_gpu.cu and in .../MinkowskiEngine/src/broadcast_gpu.cu
Add headers:
#include <thrust/execution_policy.h>
#include <thrust/unique.h>
#include <thrust/remove.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>


Then you should be able to run:
python setup.py install --user
pip install open3d
```
## Additional infos
- If the visualisation outputs only a white screen this means that no predicted boudning box is above the threshold. You can lower the threshold by setting the flag --pred-score-thr=0.01
- Use flag --show for visual output
- Use flag --wait-time=0.0, if not set it will block on the first frame
- The installation of mmdetection3d doesn't work perfectly but seems to be okay for TR3D

