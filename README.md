# Neural Network Toolbox

![Wheel](https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/f07d43c1-fda1-43b0-94df-aa10813e1b90/6023aed2-2735-4a12-a6d6-01ffa0d1b173/images/screenshot.PNG)


## Introduction
* This toolbox contains six type of neural networks 
  + Artificial neural network ( ANN ) 
  + Feed Forward Neural Network ( FFNN )
  + Cascade Forward Neural Network ( CFNN ) 
  + Recurrent Neural Network ( RNN ) 
  + Generalized Regression Neural Network ( GRNN )
  + Probabilistic Neural Network ( PNN )  

* The < Main.m file > shows the examples of how to use these neural network programs with the benchmark dataset


## Usage
The main function *jnn* is used to perform the neural network. You may switch the algorithm by simply changes the 'ffnn' to [other abbreviations](/README.md#list-of-available-neural-network-methods)   
* If you wish to use feed forward neural network ( FFNN ) then you may write
```code 
NN = jnn('ffnn',feat,label,opts);
```

* If you want to use recurrent neural network ( RNN ) then you may write
```code 
NN = jnn('rnn',feat,label,opts); 
```

## Input
* *feat*    : feature vector matrix ( Instance *x* Features )
* *label*   : label matrix ( Instance *x* 1 )
* *opts*    : parameter settings
  + *tf*    : choose either hold-out / *k*-fold 
  + *ho*    : ratio of testing data in hold-out validation
  + *kfold* : number of folds in *k*-fold cross-validation


## Output
* *NN* : Neural Network model ( It contains several results )  
  + *acc* : classification accuracy 
  + *con* : confusion matrix
  + *t*   : computational time (s)


## How to choose the validation scheme?
There are two types of validation strategies are listed as follows:
  + Hold-out validation
```code 
opts.tf    = 1;
opts.ho    = 0.3;   % 30% data for testing 
```
  + *K*-fold cross-validation
```code 
opts.tf    = 2
opts.kfold = 10;    % 10-fold cross-validation
```

### Example 1 : Feed Forward Neural Network ( FFNN ) with hold-out validation
```code 
% Benchmark dataset 
load iris.mat; 

% Perform neural network 
opts.tf        = 1;
opts.ho        = 0.3;
opts.H         = [10, 10];
opts.Maxepochs = 50;
NN = jnn('ffnn',feat,label,opts); 

% Accuracy
accuracy = NN.acc;
% Confusion matrix
confmat  = NN.con; 
```

### Example 2 : Multi-layer Neural Network ( MNN ) with *k*-fold cross-validation
```code 
% Benchmark dataset 
load iris.mat; 

% Perform neural network 
opts.tf        = 2;
opts.kfold     = 10;
opts.H         = [10, 10, 10];
opts.Maxepochs = 50;
NN = jnn('nn',feat,label,opts); 

% Accuracy
accuracy = NN.acc;
% Confusion matrix
confmat  = NN.con; 
```


## Requirement
* MATLAB 2014 or above
* Statistics and Machine Learning Toolbox


## List of available neural network methods
* Use the *opts* to set the specific parameters
* The NN, FFNN, CFNN, and RNN have two extra parameters
  + *H*          : hidden layer sizes
  + *Maxepochs*  : maximum number of epochs
* The GRNN and PNN have one extra parameter
  + *nSpread*    : number of spread


| No. | Abbreviation | Name                                   | Extra Parameter(s)   |
|-----|--------------|----------------------------------------|----------------------|
| 06  | 'nn'         | Neural Network                         | opts.H = [10, 10];   |
| 05  | 'ffnn'       | Feed Forward Neural Network            | opts.Maxepochs = 50; |
| 04  | 'cfnn'       | Cascade Forward Neural Network         |                      |
| 03  | 'rnn'        | Recurrent Neural Network               |                      |
|-----|--------------|----------------------------------------|----------------------|
| 02  | 'grnn'       | Generalized Regression Neural Network  | opts.nSpread = 1;    |
| 01  | 'pnn'        | Probabilistic Neural Network           | opts.nSpread = 0.1;  |  



  
  
  

