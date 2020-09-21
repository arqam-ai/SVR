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

## Models

|                   Model                   | Hidden layers/Hidden width/latent space width |       Grid        |              Chamfer Loss              | LR / Decay Step | Batch Size | Time   per epoch | Total Epoch |      |
| :---------------------------------------: | :-------------------------------------------: | :---------------: | :------------------------------------: | :-------------: | ---------- | ---------------- | ----------- | ---- |
|                 Oracle NN                 |                                               |                   |                                        |                 |            |                  |             |      |
|    Folding + 14 FC + two loss backward    |                14 / 512 /  512                | regular 2d square |                0.136726                |    3e-3/ 30     | 64         | 12 min           | 70          |      |
|    Folding + 7 FC + one loss backward     |                1  /  512 / 512                | regular 2d square |        0.120851 **/** 0.115729         |    3e-3/ 30     | 64         | 7.9min           | 70          |      |
| Folding + 7 FC + ReLU + one loss backward |                7  /  512 / 512                | regular 2d square | 0.108352 **/** 0.106899 **/** 0.11011  |    3e-3/ 30     | 64         | 7.9min           | 70          |      |
|             PSGN(Fan) vanila              |               2  /  512  / 512                |       None        | 0.111106 **/** 0.112874 **/** 0.109004 |    3e-3/ 30     | 64         | 3.6min           | 70          |      |
|            AtlasNet 4 patches             |               3 /   512 / 1024                | random  2d square | 0.107248 **/** 0.106854 **/** 0.107671 |    3e-3/ 30     | 64         | 7.8min           | 70          |      |
|       AtlasNet 4 patches without BN       |               3 /   512 / 1024                | random  2d square |                0.108662                |    3e-3/ 30     | 64         | 7.8min           | 70          |      |
|             AtlasNet 1 sphere             |               3 /   512 / 1024                | random  3d sphere |                0.107628                |    3e-3/ 30     | 64         | 12 min(in P100)  | 70          |      |
|            AtlasNet 16 patches            |               3 /   512 / 1024                | random  2d square |                0.107672                |    3e-3/ 30     | 64         | 16min(in P100)   | 70          |      |



## Save models

```bash
cd SVR/experiment
zip -r object_models.zip object_models
# Download the object_models.zip
```

