# DeepDTAGen
## 💡 Description
Our proposed framework, GraphDTAGen, is based on multitask deep learning framework. To predict the interactions between drugs and proteins and generate new variants of drugs with a strong affinity with existing proteins and also generate target-spacific drugs. We used Graph Encoder and Transformer Decoder to handle graph data (i.e., drug compounds) and Gated-CNN to extract features from target protein sequences. Since the model performs two tasks under the multitask learning environment. Which usually suffers from optimization issues such as conflicting gradients. Therefore we also integrated the Fetter Gradients (FetterGards) algorithm into our model to address this concern—the FetterGards algorithm successfully mitigates any potential conflicts that may arise among the tasks gradients during training.

## 📋 Table of Contents
1. [💡 Description](#description)  
2. [🔍 Dataset](#dataset)  
3. [🧠 Model Architecture](#model-architecture) 
4. [📁 Source codes](#sourcecode) 
5. [⚙️ Installation and Requirements](#installation)  
6. [🚀 Usage](#usage)  
7. [🤖🎛️ Training](#training)  
8. [📊 Results](#results)  
9. [🥇 Contribution](#contribution)  
10. [📧 Contact](#contact)  
11. [🙏 Acknowledgments](#acknowledgments)


## 🔍 Datasets:
### Dataset Source:
The KIBA and Davis dataset files were obtained from the data repository hosted at https://github.com/hkmztrk/DeepDTA/tree/master/data, while the BindingDB dataset was sourced from https://github.com/YongtaoQian/DoubleSG-DTA/blob/main/data.
### Description:
The preprocessed files of these datasets utilized in this study are available at xyz.com

### Preprocessing:
During the dataset preprocessing phase, we converted these datasets CSV files into PyTorch format to take advantage of its advanced capabilities for training neural networks.
### Dataset Size:
**Davis size**
+ The Davis dataset consists of a total of 30056 interactions.
+ The training set of Davis is consists of a total 25042 interactions
+ The testing set of Davis is consists of a total 5010 interactions  
**KIBA size**
+ The KIBA dataset consists of a total of 118254 interactions.
+ The training set of KIBA is consists of a total 98545 interactions
+ The testing set of KIBA is consists of a total 19709 interactions 
**BindingDB size**
+ The BindingDB dataset consists of a total of 56525 interactions.
+ The training set of BindingDB is consists of a total 45220 interactions
+ The testing set of BindingDB is consists of a total 11305 interactions   
### Sample Entry:
+ **Sample ID:** 12345  
+ **Drug SMILES:** CC(=O)Nc1cnc2[nH]cnc2c1N  
+ **Target Protein Sequence:** MGGKQDKIYLVLENGKTLKFPMILYGMLVYKLLNKFRNEEYDVLDKILEKKDGNFIMKVKNGKLCDLFIFSKKDINPN  
+ **Affinity Value (pKd):** 7.82  

## 🧠 Model Architecture
The DeepDTAGen architecture for drug discovery generally consists of the following components:

1. 💊⚛️ **Graph-Encoder module**: The Graph-Encoder module, denoted as q(ZDrug|X,A), is designed to process graph data represented by node feature vectors X and adjacency matrix A. The input data is organized in mini-batches of size 
[batch_size, Drug_features], where each drug is characterized by its feature vector.The goal of the Drug Encoder is to transform this high-dimensional input into a lower-dimensional representation. Typically, the Drug Encoder employs a multivariate Gaussian distribution to map the input data points to a continuous range of possible values between 0 and 1. This results in novel features that are derived from the original drug features, providing a new representation of each drug. Further the condition vector C added. However, when dealing with affinity prediction, it is necessary to keep the actual representation of the input drug to make accurate predictions. Thus, we utilized the Drug Encoder to yield a pair of outputs as follows  

 (I): For the affinity prediction task, we use the features obtained prior to the mean and log variance operation (PMVO). These features are more appropriate for predicting drug affinity, as they retain the original characteristics of the input drug without being altered by the AMVO process.

(II): For novel drug generation, we utilize the feature obtained after performing the mean and log variance operation (AMVO). This feature captures the Underlying of the input drug and is suitable for generating new drug compounds.  

2. 💊 **Transformer-Decoder Module**: The Transformer-Decoder p(DrugSMILES|ZDrug) uses latent space (AMVO) and Modified Target SMILES (MST) and generates novel drug SMILES in an autoregressive manner. 

3. 🔄 **Gated-CNN Module for Target-Proteins**: The Gated Convolutional Neural Network (GCNN) block is specifically designed to extract the features of amino acid sequences. The GCNN takes the protein sequences in the form of the embedding matrix, where each amino acid is represented by 128 feature vectors. The Gated-CNN internally splits the word embedding matrix into two parts, which are then passed through the CV-Unit and GV-Unit. The final output (extracted features) is produced by taking the element-wise product of the CV-Unit and GV-Unit. For more detail, please visit the paper.

4. 🎯 **Prediction (Fully-Connected Module)**: Our model's prediction block utilizes the extracted features from both the Drug Encoder (PMVO) and GCNN for target proteins. These features are concatenated and then passed through the designed architecture for predicting affinity.

![Model](model.jpg)

## 📁 Source codes:
The whole implementation of DeepDTAGen is based on PyTorch.  

+ create_data.py: This script generates data in PyTorch format.   
+ utils.py: Within this module, there's a variety of useful functions and classes employed by other scripts within the codebase. One notable class is TestbedDataset, which is specifically utilized by create_data.py to generate data in PyTorch format. Additionally, there's the tokenizer class responsible for preparing data for the transformer decoder. 
+ training.py: This module will train the DeepDTAGen model.
+ models.py: This module receives graph data as input for drugs while sequencing data for protein with corresponding actual labels (Affinity values). 
+ FetterGrads.py: This script FetterGrad.py is the implementation of our proposed algorithm Fetter Gradients.  
+ test.py: The script test.py is utilized to assess the performance of our saved models.  
+ generata.py: The generate.py script is employed to create drugs based on a given condition using latent space and random noise. 

## ⚙️ Installation and Requirements
You'll need to run the following commands in order to run the codes
```sh
conda env create -f environment.yml  
```
it will download all the required libraries

Or install Manually...
```sh
conda create -n DeepDTAGen python=3.8
conda activate DeepDTAGen
+ python 3.8.11
+ conda install -y -c conda-forge rdkit
+ conda install pytorch torchvision cudatoolkit -c pytorch
```
```sh
pip install torch-cluster==1.6.0+pt112cu102
```
```sh
pip install torch-scatter==2.1.0+pt112cu102
```
```sh
 pip install torch-sparse==0.6.16+pt112cu102
```
```sh
pip install torch-spline-conv==1.2.1+pt112cu102
```
```sh
pip install torch-geometric==2.2.0
```
```sh
pip pip install fairseq==0.10.2
```
```sh
pip install einops==0.6.0
```

## 🚀 Usage
write it please.

## 🤖🎛️ Training
The DeepDTAGen is trained using PyTorch and PyTorch Geometric libraries, with the support of NVIDIA GeForce RTX 2080 Ti GPU for the back-end hardware.

i.Create Data
```sh
conda activate DeepDTAGen
python create_data.py
```
The create_data.py script generates four PyTorch-formatted data files from: kiba_train.csv, kiba_test.csv, davis_train.csv, davis_test.csv,  bindingdb_train.csv, and bindingdb_test.csv and store it data/processed/, consisting of  kiba_train.pt, kiba_test.pt, davis_train.pt, davis_test.pt, bindingdb_train.pt, and bindingdb_test.pt.

ii. Train the model 
```sh
conda activate DeepDTAGen
python training.py 0 0
```
To specify the dataset, use the first argument where 0 represents 'Davis', 1 represents 'KIBA', and 2 represent 'bindingdb'. For the Cuda index, use the second argument where 0, 1, 2, and 3 represent cuda:0, cuda:1, cuda:2, and cuda:3, respectively.

## 📊 Results

###Davis  
Table 1: The performance (results) of proposed model on Davis dataset under the respective metrics 
| Model        | MSE                 | CI                  | RM2                 | 
|--------------|---------------------|---------------------|---------------------|
| DeepDTAGen   | 0.214               | 0.890               | 0.705               |


###KIBA
Table 2: The performance (results) of proposed model on KIBA dataset under the respective metrics
| Model       | MSE                 | CI                  | RM2                 |
|-------------|---------------------|---------------------|---------------------|
| DeepDTAGen  | 0.146               | 0.897               | 0.765               |


###BindingDB
Table 2: The performance (results) of proposed model on BindingDB dataset under the respective metrics
| Model       | MSE                 | CI                  | RM2                 |
|-------------|---------------------|---------------------|---------------------|
| DeepDTAGen  | 0.458               | 0.876               | 0.760               |


## 🥇 Contribution
The main contribution of our study are outlined below.  
I. The porposed model offers two functions (affinity prediction and novel drug generation)
II. We considered a natural representation of drugs in the form of graphs with a comprehensive set of atom feature.
III. The proposed model can also generate taget aware drugs.
IV. Dealing with the Multi-Task Learning environment (MTL), we introduced the FetterGards optimization algorithm to tackle the general challenges of MTL, such as conflicting gradients.
IV. The proposed model achieved the least Means Square Error in the affinity prediction task compared to the previous baseline models. 

## 📧 Contact
Have a question? or suggestion Feel free to reach out to me!.  

**📨 Email:** [Connect with me](pirmasoomshah@gmail.com)
**🌐 Google Site:** [Pir Masoom Shah](https://sites.google.com/view/pirmasoomshah/home?authuser=0)

## 📜 Reference
paper reference

<!-- ## 🙏 Acknowledgments

write it please -->
