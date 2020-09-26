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

|                   Model                   | Hidden layers/Hidden width/latent space width |       Grid        |              Chamfer Loss              | LR / Decay Step | Batch Size | Time   per epoch | Total Epoch |      |
| :---------------------------------------: | :-------------------------------------------: | :---------------: | :------------------------------------: | :-------------: | ---------- | ---------------- | ----------- | ---- |
|                 Oracle NN                 |                     0.048                     |                   |                                        |                 |            |                  |             |      |
|    Folding + 14 FC + two loss backward    |                14 / 512 /  512                | regular 2d square |                0.136726                |    3e-3/ 30     | 64         | 12 min           | 70          |      |
|    Folding + 7 FC + one loss backward     |                1  /  512 / 512                | regular 2d square |        0.120851 **/** 0.115729         |    3e-3/ 30     | 64         | 7.9min           | 70          |      |
| Folding + 7 FC + ReLU + one loss backward |                7  /  512 / 512                | regular 2d square | 0.108352 **/** 0.106899 **/** 0.11011  |    3e-3/ 30     | 64         | 7.9min           | 70          |      |
|             PSGN(Fan) vanila              |               2  /  512  / 512                |       None        | 0.111106 **/** 0.112874 **/** 0.109004 |    3e-3/ 30     | 64         | 3.6min           | 70          |      |
|            AtlasNet 4 patches             |               3 /   512 / 1024                | random  2d square | 0.107248 **/** 0.106854 **/** 0.107671 |    3e-3/ 30     | 64         | 7.8min           | 70          |      |
|       AtlasNet 4 patches without BN       |               3 /   512 / 1024                | random  2d square |                0.108662                |    3e-3/ 30     | 64         | 7.8min           | 70          |      |
|             AtlasNet 1 sphere             |               3 /   512 / 1024                | random  3d sphere |                0.107628                |    3e-3/ 30     | 64         | 12 min(in P100)  | 70          |      |
|            AtlasNet 16 patches            |               3 /   512 / 1024                | random  2d square |                0.107672                |    3e-3/ 30     | 64         | 16min(in P100)   | 70          |      |

####model version 2

|           Model            | Hidden layers/Hidden width/latent space width | Batch Norm | Grid | Chamfer Loss | LR / Decay Step | Batch Size | Time   per epoch | Total Epoch |          | Status                          |
| :------------------------: | :-------------------------------------------: | ---------- | :--: | :----------: | :-------------: | ---------- | ---------------- | ----------- | -------- | ------------------------------- |
|         Oracle NN          |                     0.048                     |            |      |              |                 |            |                  |             |          |                                 |
| Folding + 6 hidded + no BN |                6 / 512 /  512                 | no         |      |              |    1e-3/ 30     | 64         | 12 min           | 70          | 14949MiB | running in seu/object_models_v2 |
|  Folding + 6 hidded + BN   |                6 / 512 /  512                 | Yes        |      |              |    1e-3/ 30     | 64         |                  |             | 8737MiB  | running in seu/object_models_v2 |
|  Folding + 6 hidded + BN   |                6 / 512 /  512                 | Yes        |      |              |    1e-3/ 30     | 32         |                  |             | 6517MiB  | running in seu/object_models_v2 |
|                            |                                               |            |      |              |                 |            |                  |             |          |                                 |
|                            |                                               |            |      |              |                 |            |                  |             |          |                                 |
|                            |                                               |            |      |              |                 |            |                  |             |          |                                 |
|                            |                                               |            |      |              |                 |            |                  |             |          |                                 |
|                            |                                               |            |      |              |                 |            |                  |             |          |                                 |





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

