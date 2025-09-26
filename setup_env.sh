### create eonerf venv
conda create -n eonerf -c conda-forge python=3.8 libgdal

pip install torch==1.13.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
pip install torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
pip install nerfacc

export PATH=/usr/local/cuda-11.6/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64\ {LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
