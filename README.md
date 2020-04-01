# ECE C147/C247 Final Project

This is the class project repo for ECE C147/C247 (Neural Networks and Deep Learning) Winter 2019 taught by Jonathan Kao. This project contains several deep learning frameworks built for classifying EEG data. VAE (Variational Auto-encoder) was built by Mengyao Shi. ResNet was built by Gaohong Liu. DeepConvNet and CRNN were built by Hengda Shi.

Team members are: Hengda Shi, Gaohong Liu, Mengyao Shi, Zhiying Li.

## Environment Setup

### using conda

create a conda environment using the following commands:

```bash
conda create -y -name project python=3.7
```

install all the packages using the requirements file

```bash
conda install -f -y -q --name project --file requirements.txt -c pytorch
```

Activate the environment using:

```bash
conda activate project
```

deactivate the environment using:

```bash
conda deactivate
```

## Running the code

The main logic that controls all the classifiers is in `main.py`. Even though you could run the project with the following commands, the dataset is purposely encrypted due to class policy and you technically cannot run the classifiers by yourself. The classification result is listed in the report. Feel free to check it out.

```bash
python3 main.py
```

All the models and parameters can be found in the file **common/params.py**.

