% Neural Newtork Toolbox 

function NN = jnn(type,feat,label,opts)
switch type
  case 'nn'     ; fun = @mNeuralNetwork; 
  case 'ffnn'   ; fun = @mFeedForwardNeuralNetwork; 
  case 'cfnn'   ; fun = @mCascadeForwardNeuralNetwork; 
  case 'rnn'    ; fun = @mRecurrentNeuralNetwork; 
  case 'grnn'   ; fun = @mGeneralizedRegressionNeuralNetwork; 
  case 'pnn'    ; fun = @mProbabilisticNeuralNetwork; 
end
tic; 

NN = fun(feat,label,opts); 

% Store 
time = toc; 
NN.t = time;

fprintf('\n Processing Time (s): %f % \n',time); fprintf('\n');
end



