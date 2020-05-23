function [EntropyTrain,AccTest,mStepSize,stdStepSize]=...
    MBGD_UR_BN(XTrain,yTrain,XTest,yTest,alpha,eta,lambda,nRules,nIt,Nbs)

% alpha: initial learning rate
% eta: L2 regularization coefficient
% lambda: uniform regularization coefficient
% nRules: number of rules
% nIt: maximum number of iterations
% Nbs: batch size

beta1=0.9; beta2=0.999;  epsilon=1e-8;

[N,D]=size(XTrain); NTest=size(XTest,1);
classLabels=unique(yTrain); C=length(classLabels);
Nbs=min(N,Nbs);
B=2*rand(nRules,D+1,C)-1; % Rule consequents
% k-means initialization
[~,M,sumd] = kmeans(XTrain,nRules,'replicate',3);
M=M'; sumd(sumd==0)=mean(sumd); Sigma=repmat(sumd',D,1)/D;
minSigma=0.01*min(Sigma(:));
MTrain=mean(XTrain);    Sigma2Train=var(XTrain);

%% Iterative update
EntropyTrain=zeros(1,nIt); AccTest=EntropyTrain; mStepSize=EntropyTrain; stdStepSize=EntropyTrain;
mM=0; vM=0; mB=0; mSigma=0; vSigma=0; vB=0; pPred=nan(Nbs,C);
mGamma=0; vGamma=0; mBeta=0; vBeta=0; gamma=1; beta=0;
yPred=nan(1,C); yR=zeros(nRules,C,Nbs);
for it=1:nIt
    deltaC=zeros(D,nRules); deltaSigma=deltaC;  deltaB=2*eta*B; deltaB(:,1)=0; % consequent
    deltaGamma=0; deltaBeta=0;
    f=ones(Nbs,nRules); % firing level of rules
    fBar=f;
    idsTrain=datasample(1:N,Nbs,'replace',false);
    Mbs=mean(XTrain(idsTrain,:));
    Sigma2bs=var(XTrain(idsTrain,:));
    XTrainBN0=(XTrain(idsTrain,:)-repmat(Mbs,Nbs,1))./sqrt(repmat(Sigma2bs,Nbs,1)+epsilon);
    XTrainBN=gamma*XTrainBN0+beta; % batch normalized input
    
    for n=1:Nbs
        for r=1:nRules
            f(n,r)=prod(exp(-(XTrain(idsTrain(n),:)'-M(:,r)).^2./(2*Sigma(:,r).^2)));
        end
        fBar(n,:)=f(n,:)/sum(f(n,:));
        
        for c=1:C
            yR(:,c,n)=B(:,:,c)*[1; XTrainBN(n,:)'];
            yPred(c)=fBar(n,:)*yR(:,c,n); % prediction
        end
        pPred(n,:)=exp(yPred);
        pPred(n,:)=pPred(n,:)/sum(pPred(n,:));
    end
    
    % Compute delta
    for n=1:Nbs
        temp1=zeros(Nbs,nRules);
        for r=1:nRules
            for i=1:nRules
                temp1(n,r)=temp1(n,r)+(sum(fBar(:,i))/Nbs-1/nRules)*((i==r)-fBar(n,i));
            end
        end
        temp2=sum(temp1)*2*lambda/Nbs;
        for c=1:C
            temp0=pPred(n,c)-(c==yTrain(idsTrain(n)));
            for r=1:nRules
                temp=temp0*(yR(r,c)-fBar(n,:)*yR(:,c))*fBar(n,r);
                if ~isnan(temp) && abs(temp)<inf
                    % delta of c, sigma, and b
                    for d=1:D
                        deltaC(d,r)=deltaC(d,r)+temp*(XTrain(idsTrain(n),d)-M(d,r))/Sigma(d,r)^2 ...
                            +temp2(r)*fBar(n,r)*(XTrain(idsTrain(n),d)-M(d,r))/Sigma(d,r)^2;
                        deltaSigma(d,r)=deltaSigma(d,r)+temp*(XTrain(idsTrain(n),d)-M(d,r))^2/Sigma(d,r)^3 ...
                            +temp2(r)*fBar(n,r)*(XTrain(idsTrain(n),d)-M(d,r))^2/Sigma(d,r)^3;
                        deltaB(r,d+1,c)=deltaB(r,d+1,c)+temp0*fBar(n,r)*XTrainBN(n,d);
                        deltaGamma=deltaGamma+temp0*fBar(r)*B(r,d+1,c)*XTrainBN0(n,d);
                        deltaBeta=deltaBeta+temp0*fBar(r)*B(r,d+1,c);
                    end
                    % delta of b0
                    deltaB(r,1,c)=deltaB(r,1,c)+temp0*fBar(n,r);
                end
            end
        end
        % Training cross-entropy error
        EntropyTrain(it)=EntropyTrain(it)-log(pPred(n,yTrain(idsTrain(n)))); % only count cross-entropy
    end
    
    % Test error
    XTestBN=gamma*(XTest-repmat(MTrain,NTest,1))./sqrt(repmat(Sigma2Train,NTest,1)+epsilon)+beta; % batch normalized input
    yPredTest=nan(NTest,1);
    f=ones(1,nRules); % firing level of rules
    for n=1:NTest
        for r=1:nRules
            f(r)=prod(exp(-(XTest(n,:)'-M(:,r)).^2./(2*Sigma(:,r).^2)));
        end
        for c=1:C
            yR(:,c)=B(:,:,c)*[1; XTestBN(n,:)'];
            yPred(c)=f*yR(:,c); % prediction
        end
        [~,yPredTest(n)]=max(yPred);
    end
    AccTest(it)=mean(yPredTest==yTest);
    
    
    % AdaBound
    mM=beta1*mM+(1-beta1)*deltaC;
    vM=beta2*vM+(1-beta2)*deltaC.^2;
    mMHat=mM/(1-beta1^it);
    vMHat=vM/(1-beta2^it);
    mSigma=beta1*mSigma+(1-beta1)*deltaSigma;
    vSigma=beta2*vSigma+(1-beta2)*deltaSigma.^2;
    mSigmaHat=mSigma/(1-beta1^it);
    vSigmaHat=vSigma/(1-beta2^it);
    mB=beta1*mB+(1-beta1)*deltaB;
    vB=beta2*vB+(1-beta2)*deltaB.^2;
    mBHat=mB/(1-beta1^it);
    vBHat=vB/(1-beta2^it);
    mGamma=beta1*mGamma+(1-beta1)*deltaGamma;
    vGamma=beta2*vGamma+(1-beta2)*deltaGamma.^2;
    mGammaHat=mGamma/(1-beta1^it);
    vGammaHat=vGamma/(1-beta2^it);
    mBeta=beta1*mBeta+(1-beta1)*deltaBeta;
    vBeta=beta2*vBeta+(1-beta2)*deltaBeta.^2;
    mBetaHat=mBeta/(1-beta1^it);
    vBetaHat=vBeta/(1-beta2^it);
    % update C, Sigma and B, using AdaBound
    lb=alpha*(1-1/((1-beta2)*it+1));
    ub=alpha*(1+1/((1-beta2)*it));
    lrM=min(ub,max(lb,alpha./(sqrt(vMHat)+10^(-8))));
    M=M-lrM.*mMHat;
    lrSigma=min(ub,max(lb,alpha./(sqrt(vSigmaHat)+10^(-8))));
    Sigma=max(minSigma,Sigma-lrSigma.*mSigmaHat);
    lrB=min(ub,max(lb,alpha./(sqrt(vBHat)+10^(-8))));
    B=B-lrB.*mBHat;
    lrGamma=min(ub,max(lb,alpha./(sqrt(vGammaHat)+10^(-8))));
    gamma=gamma-lrGamma.*mGammaHat;
    lrBeta=min(ub,max(lb,alpha./(sqrt(vBetaHat)+10^(-8))));
    beta=beta-lrBeta.*mBetaHat;
    lr=[lrM(:); lrSigma(:); lrB(:)];
    mStepSize(it)=mean(lr); stdStepSize(it)=std(lr);
end
