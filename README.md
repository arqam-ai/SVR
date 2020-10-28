## Single View Reconstruction

## Training 

First, run the command 

```bash
git clone -b svr --depth 1 https://github.com/YefanZhou/SVR.git
```

Then, set up conda environment

```bash
conda create -n svr python=3.6 --yes
conda activate svr
conda install  pytorch torchvision cudatoolkit=10.1 -c pytorch --yes
```

Set up dependancy

```bash
cd SVR
pip install --requirement  requirements.txt
```

Download data

```bash
./download_what3d.sh
```

Before launch experiment, check the gpu id for each model

```bash
cd experiment/object_models
chmod +x object_models.sh
vim object_models.sh
```

modify the line16 to line20 gpu id based on your device 

```sh
CUDA1=0                                # gpu id for model1 
CUDA2=0                                # gpu id for model2 
CUDA3=1                                # gpu id for model3 
CUDA4=1                                # gpu id for model4 
CUDA5=1                                # gpu id for model5
```

Launch the experiment

```bash
./object_models.sh
```

## Experiment 

####model version 1

|                   Model                   | Hidden layers/Hidden width/latent space width |       Grid        | Train Loss           |              Chamfer Loss              | LR / Decay Step | Batch Size | Time   per epoch | Total Epoch |      |
| :---------------------------------------: | :-------------------------------------------: | :---------------: | -------------------- | :------------------------------------: | :-------------: | ---------- | ---------------- | ----------- | ---- |
|                 Oracle NN                 |                     0.048                     |                   |                      |                                        |                 |            |                  |             |      |
|    Folding + 14 FC + two loss backward    |                14 / 512 /  512                | regular 2d square |                      |                0.136726                |    3e-3/ 30     | 64         | 12 min           | 70          |      |
|    Folding + 7 FC + one loss backward     |                1  /  512 / 512                | regular 2d square |                      |        0.120851 **/** 0.115729         |    3e-3/ 30     | 64         | 7.9min           | 70          |      |
| Folding + 7 FC + ReLU + one loss backward |                7  /  512 / 512                | regular 2d square |                      | 0.108352 **/** 0.106899 **/** 0.11011  |    3e-3/ 30     | 64         | 7.9min           | 70          |      |
|             PSGN(Fan) vanila              |               2  /  512  / 512                |       None        | 0.1019/0.1056/ 0.099 | 0.111106 **/** 0.112874 **/** 0.109004 |    3e-3/ 30     | 64         | 3.6min           | 70          |      |
|            AtlasNet 4 patches             |               3 /   512 / 1024                | random  2d square |                      | 0.107248 **/** 0.106854 **/** 0.107671 |    3e-3/ 30     | 64         | 7.8min           | 70          |      |
|       AtlasNet 4 patches without BN       |               3 /   512 / 1024                | random  2d square |                      |                0.108662                |    3e-3/ 30     | 64         | 7.8min           | 70          |      |
|             AtlasNet 1 sphere             |               3 /   512 / 1024                | random  3d sphere |                      |                0.107628                |    3e-3/ 30     | 64         | 12 min(in P100)  | 70          |      |
|            AtlasNet 16 patches            |               3 /   512 / 1024                | random  2d square |                      |                0.107672                |    3e-3/ 30     | 64         | 16min(in P100)   | 70          |      |

####model version 2

|           Model            | Exp index | mode           | patches | Hidden layers/Hidden width/latent space width | Batch Norm |  mIoU  | Train Chamfer Loss | Test Chamfer Loss | LR / Decay Step | Batch Size | Time   per epoch | Total Epoch | GPU Occupancy | Model Size | Status |
| :------------------------: | --------- | -------------- | ------- | :-------------------------------------------: | ---------- | :----: | ------------------ | :---------------: | :-------------: | ---------- | ---------------- | ----------- | ------------- | ---------- | ------ |
|         Oracle NN          |           | object center  |         |                                               |            |        |                    |      0.08685      |                 |            |                  |             |               |            |        |
| Folding + 6 hidded + no BN |           | object  center |         |                6 / 512 /  512                 | no         |        | 0.0805             |     0.101191      |    1e-3/ 30     | 64         | 12 min           | 70          | 8737MiB       |            | done   |
|  Folding + 6 hidded + BN   |           | object center  |         |                6 / 512 /  512                 | Yes        |        | 0.0814             |     0.099870      |    1e-3/ 30     | 64         | 14 min           | 70          | 8737MiB       |            | done   |
|  Folding + 6 hidded + BN   |           | object center  |         |                6 / 512 /  512                 | Yes        |        | 0.0859             |     0.100859      |    1e-3/ 30     | 32         | 30 min           | 70          | 4729MiB       |            | done   |
|  Folding + 6 hidded + BN   |           | object center  |         |                  6/1024/1024                  | Yes        |        | 0.0850             |     0.100688      |    1e-3/ 30     | 32         | 30 min           | 70          |               |            | done   |
|   AtlasNet + 6 hidden+BN   |           | object center  | 1       |                6 / 512 /  512                 | Yes        |        | 0.0883             |     0.102240      |    1e-3/ 30     | 64         | 16min            | 70          |               |            | done   |
|   AtlasNet + 6 hidden+BN   |           | object center  | 1       |                  6/1024/1024                  | Yes        |        | 0.0922             |     0.102958      |    1e-3/ 30     | 32         | 64min            | 70          |               |            | done   |
|   AtlasNet + 2 hidden+BN   |           | object center  | 4       |                   2/512/512                   | Yes        | 0.3376 | 0.0850             |     0.101831      |    1e-3/ 30     | 64         | 538s             | 70          |               |            | done   |
|   AtlasNet + 6 hidden+BN   |           | object center  | 4       |                   6/512/512                   | Yes        |        | 0.0870             |     0.102758      |    1e-3/ 30     | 64         | 874s             | 70          |               |            | done   |
|   AtlasNet + 2 hidden+BN   |           | object center  | 8       |                   2/512/512                   | Yes        |        | 0.0857             |     0.103496      |    1e-3/ 30     | 64         | 561s             | 70          |               |            | done   |
|                            |           |                |         |                                               |            |        |                    |                   |                 |            |                  |             |               |            |        |
|                            |           |                |         |                                               |            |        |                    |                   |                 |            |                  |             |               |            |        |
|         Oracle NN          |           | viewer  center |         |                                               |            |        |                    |      0.1069       |                 |            |                  |             |               |            |        |
|  Folding + 6 hidded + BN   |           | viewer center  |         |                6 / 512 /  512                 | no         |        | 0.0857             |     0.102078      |    1e-3/ 30     | 64         | 12min            | 70          | 8737MiB       |            | done   |
|  Folding + 6 hidded + BN   |           | viewer center  |         |                6 / 512 /  512                 | Yes        |        | 0.0861             |     0.099518      |    1e-3/ 30     | 64         |                  | 70          | 4729MiB       |            | done   |
|  Folding + 6 hidded + BN   |           | viewer center  |         |                6 / 512 /  512                 | Yes        |        | 0.0920             |     0.100486      |    1e-3/ 30     | 32         | 15min            | 70          | 4729MiB       |            | done   |
|   AtlasNet + 6 hidden+BN   |           | viewer center  | 1       |                6 / 512 /  512                 | Yes        |        | 0.0930             |     0.102212      |    1e-3/ 30     | 64         | 15min            | 70          | 8737MiB       |            | done   |
|   AtlasNet + 6 hidden+BN   |           | viewer center  | 4       |                6 / 512 /  512                 | Yes        |        | 0.0927             |     0.102251      |    1e-3/ 30     | 64         | 871s             | 70          |               |            | done   |
|   AtlasNet + 6 hidden+BN   |           | viewer center  | 4       |                   2/512/512                   | Yes        |        | 0.0933/ 0.0929     | 0.102922 /0.10251 |    1e-3/ 30     | 64         | 536s             | 70          |               |            | done   |
|                            |           |                |         |                                               |            |        |                    |                   |                 |            |                  |             |               |            |        |
|                            |           |                |         |                                               |            |        |                    |                   |                 |            |                  |             |               |            |        |





#### model detail:

resnet  atlasnet coderepo resnet

currently all the initialization are all default 

##### ##Initialization

check the PyTorch module to see their default initialization https://github.com/pytorch/pytorch/tree/master/torch/nn/modules

For Conv2d 

```python
for m in self.modules():
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
```

For Linear, For conv1d

```python
init.kaiming_uniform_(self.weight, a=math.sqrt(5))
if self.bias is not None:
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)
```

https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

How to initialize weights in Pytorch

https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch

## Save models

```bash
cd SVR/experiment
zip -r object_models.zip object_models
# Download the object_models.zip
```

###DEMO

Visualization for point cloud 

https://github.com/zekunhao1995/PointFlowRenderer

https://github.com/YefanZhou/Pointcloud-Learning/blob/master/utils/visualize.py

Visualization for network

https://github.com/waleedka/hiddenlayer

how to calcuate model size 

