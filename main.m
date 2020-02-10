clc; clearvars; close all; rng(0);

nRules=20; % number of rules
alpha=.01; % initial learning rate
Nbs=64; % batch size
eta=0.05; % L2 regularization coefficient
lambda=10; % UR coefficient
nIt=500; % number of iterations

temp=load('Vehicle.mat'); data=temp.data;
X=data(:,1:end-1); y0=data(:,end);
labels=unique(y0); y=nan(size(y0));
for i=1:length(labels)
    y(y0==labels(i))=i;
end
X = zscore(X); [N0,M]=size(X);
N=round(N0*.7);
idsTrain=datasample(1:N0,N,'replace',false);
XTrain=X(idsTrain,:); yTrain=y(idsTrain);
XTest=X; XTest(idsTrain,:)=[]; yTest=y; yTest(idsTrain)=[];

% MBGD-UR-BN
[EntropyTrain,AccTest]=MBGD_UR_BN(XTrain,yTrain,XTest,yTest,alpha,eta,lambda,nRules,nIt,Nbs);

% Plot results
figure('Position', get(0, 'Screensize'));
subplot(121);
plot(EntropyTrain,'linewidth',1);
xlabel('Iteration'); ylabel('Training cross-entropy');
subplot(122);
plot(AccTest,'linewidth',2);
set(gca,'yscale','log');
xlabel('Iteration'); ylabel('Test accuracy');


