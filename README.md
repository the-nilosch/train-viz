# train-viz
Visualizing the training

### Venv (Old setup)

Create:
```
python -m venv .venv
```

Activate:
```
source .venv/bin/activate
```
Windows:
```
source venv/Scripts/activate
```

Saved packages:
```
pip freeze | grep -v '^\-e' | sed '/ @ /d' > requirements.txt
conda env export --no-builds > reuqirements.yml
```

### First Environment (train-viz)

```
conda create -n train-viz python=3.10
conda activate train-viz

conda install numpy scipy scikit-learn matplotlib pandas
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia

pip install -r requirements.txt

python -m ipykernel install --user --name train-viz --display-name "Python (train-viz)"
```


### Second Environment (phate-env)

```
conda create -n phate-env python=3.10

conda activate phate-env

pip install "numpy<2.0" "pandas>=1.5,<2.0"
pip install phate m-phate
pip install notebook jupyterlab ipykernel ipywidgets ipympl
pip install tqdm h5py umap torchvision
```

Register to jupyter kernel
```
python -m ipykernel install --user --name phate-env --display-name "Python (phate-env)"
```

Currently not needed:
```
pip install -r requirements-phate-env.txt
```


Saved packages:
```
pip freeze | grep -v '^\-e' | sed '/ @ /d' > requirements-phate-env.txt
conda env export --no-builds > phate-env.yml
```

## Initialize Subpackages
### Neuro-Visualizer
```
git clone https://github.com/elhamod/NeuroVisualizer.git
cd NeuroVisualizer
pip install -r requirements.txt
```
**For Windows:** On Windows folders named `aux` are not allowed - download as ZIP file and rename the folder `aux` to `neuro_aux`

