git clone -b svr --depth 1 https://github.com/YefanZhou/SVR.git
conda create -n svr python=3.6 --yes
conda activate svr
conda install  pytorch torchvision cudatoolkit=10.1 -c pytorch --yes
pip install --requirement  requirements.txt
python utils/PyTorchEMD/setup.py build