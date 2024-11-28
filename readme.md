# RNALoc-LM
A pre-trained RNA language model powered for RNA subcellular localization prediction.

This repository contains the code for the RNA Subcellular Localization Prediction Model (RNALoc-LM), which can be used to predict the subcellular localization of three types of RNA: lncRNA, circRNA, and miRNA. 

Also the web server for prediction and visualization are available at http://csuligroup.com:8000/RNALoc-LM.

Before start, you need to create the environment required for the project. You can choose to create the environment directly using the RNALoc-LM.yaml file, or use the requirements.txt file to create the environment.

### Create Environment with RNALoc-LM.yaml

First, create the required environment.

```python
cd RNALoc-LM
conda env create -f RNALoc-LM.yaml
```

Then, activate the "RNALoc-LM" environment and enter into the workspace.

```python
conda activate RNALoc-LM
```

### Create Environment with requirements.txt

```python
numpy==1.20.3
tqdm==4.62.3
scikit-learn==0.24.0
pickleshare==0.7.5
pandas==1.3.1
pytorch==1.9.0
ptflops==0.6.6
rna-fm==0.2.2
```
It should be noted that you need to use ```pip install rna-fm==0.2.2``` to install rna-fm.

### Usage

First of all, you need to download pre-trained models from [this gdrive link](https://drive.google.com/drive/folders/1VGye74GnNXbUMKx6QYYectZrY7G2pQ_J?usp=share_link) and put the pth files into the `pretrained` folder.

Then, you can train the model in a very simple way. For different RNAs, just train the corresponding python files.

```python
python lncRNA.py
python circRNA.py
python miRNA.py
```

To change model hyperparameters, you can modify them in the corresponding files.

The model hyperparameters are as follows:

```python
data_dir: the folder where the data is located
output_dir: the location where the generated log files are saved
pretrained_dir: the folder where the pre-trained model is located
seed: random Seed
gpu_id: specified gpuid
rna_type: RNA type, including lncRNA, circRNA, miRNA
train_file: the name of training dataset
test_file: the name of testing dataset
embedding_file: the name of training dataset embedding file
test_embedding_file: the name of testing dataset embedding file
model_path: the model location
log_name: the name of log file
epochs: num of epoch
batch_size: batch size
lr: learning rate
input_dim: the input dimensions
num_filters: the number of convolution kernels
filter_sizes: different convolution kernel sizes
dropout: dropout
hidden_dim: dimensions of LSTM hidden layers
num_layers: number of LSTM layers
num_heads: number of attention heads
```

Or you can directly use the trained downstream model to make predictions on the test set.

```python
python test.py
```

### Citation

Min Zeng, Xinyu Zhang, Yiming Li, Chengqian Lu, Rui Yin, Fei Guo, Min Li*, “RNALoc-LM: a pre-trained RNA language model powered for RNA subcellular localization prediction”.



