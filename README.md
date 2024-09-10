# DeepDTAGen
## 💡 Description
This is the implementation of DeepDTAGen: Multitask deep learning framework for Predicting Drug-Target Affinity and Generating Target-Specific Drugs.

## 📋 Table of Contents
1. [💡 Description](#description)  
2. [🔍 Dataset](#dataset)  
3. [🧠 Model Architecture](#model-architecture) 
4. [🛠️ Preprocessing](#Preprocessing)
5. [📊 System Requirements](#System-requirements)
6. [⚙️ Installation and Requirements](#installation)  
7. [📁 Source codes](#sourcecode) 
8. [🖥️ Demo](#demo)   
9. [🤖🎛️ Training](#training)  
10. [📧 Contact](#contact)  
11. [🙏 Acknowledgments](#acknowledgments)


## 🔍 Datasets:
### Dataset Source:
The KIBA and Davis datasets were obtained from the data repository hosted at https://github.com/hkmztrk/DeepDTA/tree/master/data, while the BindingDB dataset was obtained from https://github.com/YongtaoQian/DoubleSG-DTA/blob/main/data.

## 🧠 Model Architecture
The DeepDTAGen architecture consists of the following components:

1. 💊⚛️ **Graph-Encoder module**: The Graph-Encoder module, denoted as q(ZDrug|X,A), is designed to process graph data represented as node feature vectors X and adjacency matrix A. The input data is organized in mini-batches of size 
[batch_size, Drug_features], where each drug is characterized by its feature vector.The goal of the Drug Encoder is to transform this high-dimensional input into a lower-dimensional representation. Typically, the Drug Encoder employs a multivariate Gaussian distribution to map the input data points to a continuous range of possible values between 0 and 1. This results in novel features that are derived from the original drug features, providing a new representation of each drug. Further the condition vector C added. However, when dealing with affinity prediction, it is necessary to keep the actual representation of the input drug to make accurate predictions. Thus, we utilized the Drug Encoder to yield a pair of outputs as follows  

 (I): For the affinity prediction task, we use the features obtained prior to the mean and log variance operation (PMVO). These features are more appropriate for predicting drug affinity, as they retain the original characteristics of the input drug without being altered by the AMVO process.

(II): For novel drug generation, we utilize the feature obtained after performing the mean and log variance operation (AMVO).

2. 🔄 **Gated-CNN Module for Target-Proteins**: The Gated Convolutional Neural Network (GCNN) block is specifically designed to extract the features of target sequences. The GCNN takes the protein sequences in the form of the embedding matrix, where each amino acid is represented by 128 feature vectors and extracts the features as output. 
3. 💊 **Transformer-Decoder Module**: The Transformer-Decoder p(DrugSMILES|ZDrug) uses latent space (AMVO) and Modified Target SMILES (MST) and generates novel drug SMILES in an autoregressive manner ((More details are available in the main article section 1.3)). 


4. 🎯 **Prediction (Fully-Connected Module)**: The prediction block utilizes the extracted features from the Drug Encoder (PMVO) and GCNN for target proteins and predicts the affinity between the given drug and the target.

![Model](model.jpg)

##🛠️ Preprocessing
+ Drugs: The SMILES string representation are converted to the chemical structure using the RDKit library. We then use NetworkX to further convert it to graph representation.
+ Proteins: The protein sequence convert it into a numerical representation using label encoding. Further some more steps preprocessing steps were applied (more detail are provided in the main text).


## System requirements 
+ Operating System: Ubuntu 16.04.7 LTS
+ CPU: Intel(R) Xeon(R) Silver 4114 CPU @ 2.20GHz
+ GPU: GeForce RTX 2080 Ti
+ CUDA: 10.2


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
+ The whole installation maximum takes about 30 minutes.

## 📁 Source codes:
The whole implementation of DeepDTAGen is based on PyTorch.  

+ create_data.py: This script generates data in PyTorch format.   
+ utils.py: Within this module, there's a variety of useful functions and classes employed by other scripts within the codebase. One notable class is TestbedDataset, which is specifically utilized by create_data.py to generate data in PyTorch format. Additionally, there's the tokenizer class responsible for preparing data for the transformer decoder. 
+ training.py: This module will train the DeepDTAGen model.
+ models.py: This module receives graph data as input for drugs while sequencing data for protein with corresponding actual labels (Affinity values). 
+ FetterGrads.py: This script FetterGrad.py is the implementation of our proposed algorithm Fetter Gradients.  
+ test.py: The script test.py is utilized to assess the performance of our saved models.  
+ generata.py: The generate.py script is employed to create drugs based on a given condition using latent space and random noise. 

## Demo
We have provided a DEMO directory, having two files "DEMO_Affinity.py" and "DEMO_Generation.py". "DEMO_Affinity.py" can be used to demonstrate affinity prediction, allowing users to test our model using a sample input. While "DEMO_Generation.py", can be used for drug generation, providing a test case for evaluating our model's performance in generating drugs. 
+ DEMO_Affinity.py for affinity prediction 
+ DEMO_Generation.py for drug generation.
Running these files takes approximately 1 to 2 seconds.
Expected results for the given input in the DEMO_Affinity.py is (predicted affinity between the given inputs: 6.255425453186035)
Expected result for the given input in the DEMO_Generation.py is (generated drug: O=C(c1cc(C(F)(F)F)ccc1F)N(C1CCN(C(=O)c2ccc(Br)cc2)CC1)C(=O)N1CCCC1 based on the given input)

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


## 📧 Contact
Have a question? or suggestion Feel free to reach out to me!.  

**📨 Email:** [Connect with me](pirmasoomshah@gmail.com)
**🌐 Google Site:** [Pir Masoom Shah](https://sites.google.com/view/pirmasoomshah/home?authuser=0)

## 📜 Reference
paper reference

<!-- ## 🙏 Acknowledgments

write it please -->
