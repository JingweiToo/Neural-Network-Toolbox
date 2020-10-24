function FFNN=jFFNN(feat,label,kfold,Hiddens,Maxepochs)
if length(Hiddens)==1
    h1=Hiddens(1); net=feedforwardnet(h1);
elseif length(Hiddens)==2
    h1=Hiddens(1); h2=Hiddens(2); net=feedforwardnet([h1 h2]);
elseif length(Hiddens)==3
    h1=Hiddens(1); h2=Hiddens(2);
    h3=Hiddens(3); net=feedforwardnet([h1 h2 h3]); 
end
fold=cvpartition(label,'kfold',kfold); 
pred2=[]; ytest2=[]; Afold=zeros(kfold,1); 
for i=1:kfold
  trainIdx=fold.training(i); testIdx=fold.test(i);
  xtrain=feat(trainIdx,:); ytrain=label(trainIdx);
  xtest=feat(testIdx,:); ytest=label(testIdx);
  net.trainParam.epochs= Maxepochs;
  net=train(net,xtrain',dummyvar(ytrain)');
  pred=net(xtest'); 
  [~,con]=confusion(dummyvar(ytest)',pred);
  Afold(i)=100*sum(diag(con))/sum(con(:));
  pred2=[pred2(1:end,:),pred]; ytest2=[ytest2(1:end);ytest]; 
end
[~,confmat]=confusion(dummyvar(ytest2)',pred2); confmat=transpose(confmat);
acc=mean(Afold);
FFNN.fold=Afold; FFNN.acc=acc; FFNN.con=confmat; 
fprintf('\n Classification Accuracy (FFNN): %g %%',acc); 
end

