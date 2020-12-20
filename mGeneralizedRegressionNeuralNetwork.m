% Generalized Regression Neural Network 

function GRNN = mGeneralizedRegressionNeuralNetwork(feat,label,opts)
% Parameters
num_spread = 1;
kfold      = 10; 
tf         = 2; 

if isfield(opts,'kfold'), kfold = opts.kfold; end
if isfield(opts,'ho'), ho = opts.ho; end
if isfield(opts,'tf'), tf = opts.tf; end
if isfield(opts,'nSpread'), num_spread = opts.nSpread; end 

pred2  = [];
ytest2 = [];

% [Hold-out]
if tf == 1
  fold  = cvpartition(label,'HoldOut',ho);
  K     = 1;
  
% [Cross-validation] 
elseif tf == 2
  fold  = cvpartition(label,'KFold',kfold);
  K     = kfold;
  Afold = zeros(kfold,1);
end

for i = 1:K
  % Call train & test data
  trainIdx = fold.training(i); testIdx = fold.test(i);
  xtrain   = feat(trainIdx,:); ytrain  = label(trainIdx);
  xtest    = feat(testIdx,:);  ytest   = label(testIdx); 
  % Training the model
  net      = newgrnn(xtrain',dummyvar(ytrain)',num_spread); 
  % Perform testing
  pred     = net(xtest');
  % Confusion matrix
  [~, con] = confusion(dummyvar(ytest)',pred);
  % Get accuracy for each fold
  Afold(i) = sum(diag(con)) / sum(con(:));
  % Store temporary result for each fold
  pred2    = [pred2(1:end,:), pred]; 
  ytest2   = [ytest2(1:end); ytest];
end
% Overall confusion matrix
[~, confmat] = confusion(dummyvar(ytest2)',pred2);
confmat      = transpose(confmat);
% Average accuracy over k-folds
acc = mean(Afold);
% Store results 
GRNN.acc  = acc;
GRNN.con  = confmat;

if tf == 1
  fprintf('\n Classification Accuracy (GRNN-HO): %g %%',100 * acc);
elseif tf == 2
  fprintf('\n Classification Accuracy (GRNN-CV): %g %%',100 * acc);
end
end

