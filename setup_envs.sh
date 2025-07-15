#!/bin/bash

# Exit on any error
set -e

echo "=== Creating first environment: train-viz ==="
conda create -y -n train-viz python=3.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate train-viz

conda install -y numpy scipy scikit-learn matplotlib pandas
# NOTE: Update CUDA VERSION here: 12.4 (nembus) / 11.8 (prak-inf-1-10)
conda install -y pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt

python -m ipykernel install --user --name train-viz --display-name "Python (train-viz)"

conda deactivate

echo "=== Creating second environment: phate-env ==="
conda create -y -n phate-env python=3.10
conda activate phate-env

pip install "numpy<2.0" "pandas>=1.5,<2.0"
pip install phate m-phate
pip install notebook jupyterlab ipykernel ipywidgets ipympl
pip install tqdm h5py umap torchvision

python -m ipykernel install --user --name phate-env --display-name "Python (phate-env)"

conda deactivate

echo "=== Done! Both environments are ready. ==="

# conda remove --name phate-env --all
# conda remove --name train-viz --all