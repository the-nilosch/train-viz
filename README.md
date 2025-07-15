# train-viz
Visualizing the training

### Venv

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

### Second Environment for PHATE

```
conda create -n phate-env python=3.10

conda activate phate-env

conda install numpy=1.24 pandas=1.5
pip install phate m-phate

pip install -r requirements-phate-env.txt
```

Saved packages:
```
pip freeze | grep -v '^\-e' | sed '/ @ /d' > requirements-phate-env.txt
conda env export --no-builds > phate-env.yml
```

### Initialize Neuro-Visualizer
```
git clone https://github.com/elhamod/NeuroVisualizer.git
cd NeuroVisualizer
pip install -r requirements.txt
```
**For Windows:** On Windows folders named `aux` are not allowed - download as ZIP file and rename the folder `aux` to `neuro_aux`

