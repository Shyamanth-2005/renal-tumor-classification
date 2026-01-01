# Renal-Tumor-Classification-Project

...

## Workflows

1.  Update config.yaml
2.  Update secrets.yaml [Optional]
3.  Update params.yaml
4.  Update the entity
5.  Update the configuration manager in src config
6.  Update the components
7.  Update the pipeline
8.  Update the main.py
9.  Update the dvc.yaml
10. app.py

...

...

# How to run ?

...

### STEPS:

Clone the repository

```bash
https://github.com/Shyamanth-2005/renal_tumor_classification.git

```

### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n your_env_name python=3.8 -y

```

```bash
conda activate your_env_name

```

### STEP 02- install the requirements

```bash
pip install -r requirements.txt
```

## MLflow

- [Documentation](https://mlflow.org/docs/latest/index.html)

##### cmd

- mlflow ui

### dagshub

[dagshub](https://dagshub.com/)

MLFLOW_TRACKING_URL=https://dagshub.com/Shyamanth-2005/renal_tumor_classification.mlflow/
MLFLOW_TRACKING_USERNAME=Shyamanth-2005 \
MLFLOW_TRACKING_PASSWORD=1d22cf43202806cbe2bce95333b79a30fa13b1cb \
python script.py

Run this to export as env variables:

```bash

export MLFLOW_TRACKING_URL=https://dagshub.com/Shyamanth-2005/renal_tumor_classification.mlflow

export MLFLOW_TRACKING_USERNAME=Shyamanth-2005

export MLFLOW_TRACKING_PASSWORD=1d22cf43202806cbe2bce95333b79a30fa13b1cb

```

### DVC

- update dvc.yaml

```bash
dvc init
dvc repro
dvc dag
```
