# Variational Deep Matrix Factorization with Side Information for Collaborative Filtering(VDMF)
 
- This the original implementation of Variational Deep Matrix Factorization with Side Information for Collaborative Filtering(VDMF).
- This is the code for dataset movielens-100k, if you want to apply this model to ml-1M or book-crossing, you can implement the pre-processing code according to ../base/prepro.py and change the size of layers in the NN based on our paper.
- We've implemented both explicit version(users' scores towards items remain unchanged) and implicit version(scores are transfered into 0 and 1)

## 0. Requirements
#### General
- Python (verified on 3.4.3)

#### Python Packages
- tensorflow (deep learning library, we implement this project in v1.7.0)

## 1. Pre-processing
First, prepare data. Donwload SQuAD data and GloVe and nltk corpus
Download movielens-100k dataset and put the data into ../data

Second, Preprocess ml-100k dataset:
```
python prepro.py
```

## 2. Training
The model was originally trained in a cluster system with 55 CPU and no GPU.
If you want to train it in GPU, you can easily change the device setting in model.py and model_ex.py.
If trained in GPU, the model requires at least 12GB of GPU RAM.
If your GPU RAM is smaller than 12GB, you can either decrease batch size.
In cluster system, the training lasted nearly 12 hours.

To train the explicit model, run:
```
python main_ex.py -m train -r 80
```
the opt '-r' means the rate of the training set with the whole data set, -r can be 60(60%),80(80%) and 95(95%)

To train the implicit model, run:
```
python main.py -m train -r 80
```

During the training process, program will print the RMSD, recall(to save time, recall here only include the average recall of 10 users) each 100 epoch.
All the logs generated in training process and the trained network will be store in ../data/summary_r after training is done


## 3. Test

You can test the model with RMSD and recall.

To test RMSD, run
```
python main.py -m rmsd -r 80 -c 2000
```
for implicit model or 
```
python main_ex.py -m rmsd -r 80 -c 2000
```
for explicit model.
The oprand '-c' means the the checkpoint where the model load the pre-trained network. For example, '-c 2000' means the model will load the parameters of network after epoch 2000.

To test RMSD, run
```
python main.py -m recall -r 80 -c 2000
```
for implicit model or 
```
python main_ex.py -m recall -r 80 -c 2000
```
for explicit model.