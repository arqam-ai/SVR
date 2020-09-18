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

|               Model                | Chamfer Loss | Converge Epoch | LR / Decay step | Batch Size |
| :--------------------------------: | :----------: | :------------: | :-------------: | :--------: |
|             Oracle NN              |    0.048     |                |                 |            |
| Folding + 6 FC + two loss backward |              |                |                 |            |
| Folding + 6 FC + one loss backward |   0.120851   |       45       |    3e-3/ 40     |     64     |
| Folding + 18FC + one loss backward |              |                |                 |            |
|          PSGN(Fan) vanila          |   0.111106   |       60       |    3e-3 / 40    |     64     |
|         AtlasNet 4 patches         |   0.107248   |       60       |    3e-3 / 40    |     64     |
|         AtlasNet 1 sphere          |              |                |                 |            |

