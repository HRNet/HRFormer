git clone https://github.com/NVIDIA/apex
cd apex
/data/anaconda/envs/pytorch1.7.1/bin/python -m pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./