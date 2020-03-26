# ECE C147/C247 Final Project

Construct CNN, CNN+RNN, VAE for EEG data.

## Environment Setup

### using virtualenv

create a virtual environment and use the following command:

```bash
pip install -r requirements.txt
```

### using conda

create a conda environment using the following commands:

```bash
conda create -y -name rnn python=3.7
```

install all the packages using the requirements file

```bash
conda install -f -y -q --name rnn --file conda-requirements.txt -c pytorch
```

Activate the environment using:

```bash
conda activate rnn
```

deactivate the environment using:

```bash
conda deactivate
```

## Running the code

The classifier can be run by invoking:

```bash
python3 main.py
```

All the models and parameters can be found in the file **common/params.py**. Feel free to play with all the models.
