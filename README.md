# Buffer based threshold adaptation
In this repo, we model the problem of setting the threshold in an early exit framework based on the input buffer size

## Requirements

```Requirements
conda env create -f environment.yaml
```

## Datasets
Datasets can be downloaded from the dataset.ipynb file, will be adding more datasets soon.

## Step-1: Training the early exit model and creating a csv file that saves confidences of exits, prediction of exits and true labels 

Befor this step, do not forget to set the correct dataset path in the param.py file.

To fine-tune a pre-trained language model and train the internal classifiers follow the command:

```Training and creating the csv of confidences
python main.py --pretrain --src QNLI
```

## Step2: Run the simulation

After this step run this command to get the simulation results:

```Simulation run
python simulation.py
```

param.py file contains all the parameters related to the simulation.


