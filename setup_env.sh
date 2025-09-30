### create eonerf venv
conda create -n eonerf -c conda-forge python=3.9 libgdal

# for CUDA 12.1
export CUDA_HOME=/usr/local/cuda-12
export PATH=/usr/local/cuda-12/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12/lib64\ {LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
pip install --no-cache-dir torch==2.1.0+cu121 torchvision==0.16.0+cu121 --index-url https://download.pytorch.org/whl/cu121

pip install --no-cache-dir git+https://github.com/nerfstudio-project/nerfacc.git@v0.5.2 # this can take some time
pip install pyproj==3.0.1


