% Programmer: Jingwei Too

function GRNN=jGRNN(feat,label,kfold,nSpread)
fold=cvpartition(label,'kfold',kfold);
pred2=[]; ytest2=[]; Afold=zeros(kfold,1); 
for i=1:kfold
	trainIdx=fold.training(i); testIdx=fold.test(i);
  xtrain=feat(trainIdx,:); ytrain=label(trainIdx);
  xtest=feat(testIdx,:); ytest=label(testIdx);
  net=newgrnn(xtrain',dummyvar(ytrain)',nSpread); 
  pred=net(xtest');
  [~,con]=confusion(dummyvar(ytest)',pred);
  Afold(i)=100*sum(diag(con))/sum(con(:));
  pred2=[pred2(1:end,:),pred]; ytest2=[ytest2(1:end);ytest];
end
[~,confmat]=confusion(dummyvar(ytest2)',pred2); confmat=transpose(confmat);
acc=mean(Afold);
GRNN.fold=Afold; GRNN.acc=acc; GRNN.con=confmat; 
fprintf('\n Classification Accuracy (GRNN): %g %%',acc); 
end

