%----------------------------------------------------------------------%
%  Neural Network (NN) toolbox                                         %
%----------------------------------------------------------------------%

%---Input--------------------------------------------------------------
% feat           : feature vector (instances x features)
% label          : label vector (instances x 1)
% opts           : parameter settings
% opts.tf        : types of validation method
% opts.ho        : ratio of testing set in hold-out validation
% opts.kfold     : number of k in k-fold cross-validation
% opts.H         : hidden layers (up to three, such as [10,10,10]) 
% opts.Maxepochs : maximum number of Epochs
% opts.nSpread   : number of spreads  

%---Output-------------------------------------------------------------
% A struct that contains three results as follows:
% acc            : Overall accuracy 
% con            : Confusion matrix
% t              : Computational time (s)
%----------------------------------------------------------------------


%% Feed Forward Neural Network (FFNN) with hold-out validation
clc, clear
% Benchmark dataset 
load iris.mat; 

% Perform neural network 
opts.tf        = 1;
opts.ho        = 0.3;
opts.H         = 10;
opts.Maxepochs = 50;
NN = jnn('ffnn',feat,label,opts); 

% Accuracy
accuracy = NN.acc;
% Confusion matrix
confmat  = NN.con; 


%% Multi-layer Neural Network (MNN) with k-fold cross-validation
clc, clear
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




