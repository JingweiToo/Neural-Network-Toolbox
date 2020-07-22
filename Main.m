%-------------------------------------------------------------------------%
%  Neural Network (NN) source codes demo version                          %
%                                                                         %
%  Programmer: Jingwei Too                                                %
%                                                                         %
%  E-Mail: jamesjames868@gmail.com                                        %
%-------------------------------------------------------------------------%


%---Input------------------------------------------------------------------
% feat:      features (intances x features)
% label:     labelling (labels x 1)
% kfold:     Number of cross-validation
% Hiddens:   Hidden layers (Hidden layers up to three, such as [10,10,10]) 
% Maxepochs: Maximum number of Epochs
%---Output-----------------------------------------------------------------
% A struct that contains three results as follows:
% fold: Accuracy for each fold
% acc:  Average accuracy over k-folds
% con:  Confusion matrix
%--------------------------------------------------------------------------


%% Neural Network
clc, clear
% Benchmark dataset 
load iris.mat; 

% (1) Perform neural network (NN)
kfold=10; Hiddens=10; Maxepochs=50;
NN=jNN(feat,label,kfold,Hiddens,Maxepochs);

% (2) Perform neural network with multiple layers (MNN)
kfold=10; Hiddens=[10,10]; Maxepochs=50;
MNN=jNN(feat,label,kfold,Hiddens,Maxepochs);

% (3) Perform feed-foward neural network (FFNN)
kfold=10; Hiddens=10; Maxepochs=50;
FFNN=jFFNN(feat,label,kfold,Hiddens,Maxepochs);

% (4) Perform cascade foward neural network (CFNN)
kfold=10; Hiddens=10; Maxepochs=50;
CFNN=jCFNN(feat,label,kfold,Hiddens,Maxepochs);

% (5) Perform recurrent neural network (RNN)
kfold=10; Hiddens=10; Maxepochs=50;
RNN=jRNN(feat,label,kfold,Hiddens,Maxepochs);

% (6) Perform generalized regression neural network (GRNN)
kfold=10; nSpread=1; % Number of spread in GRNN
GRNN=jGRNN(feat,label,kfold,nSpread);

% (7) Perform probabilistic neural network (PNN)
kfold=10; nSpread=0.1; % Number of spread in PNN
PNN=jPNN(feat,label,kfold,nSpread);





