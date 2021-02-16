%% This code is for NATIM model
% This model was coded to predict prognosis of breast cancer with NAC.
clear all
clc
load Input.mat

% Randomization
rng(10)
n = randperm(size(matrix,1));
trainX = matrix(n(1:177),:);
trainY = label(n(1:177),:);
textX = matrix(n(178:end),:);
textY = label(n(178:end),:);

% Normalization
[Ptrain,PS] = mapminmax(trainX');
Ptrain = Ptrain';
Ptext = mapminmax('apply',textX',PS);
Ptext = Ptext';

% Building the NATIM model
cmd = [' -t 2',' -c ',num2str(4.594793419988138),' -g ',num2str(6.062866266041591), '-b 1'];
model = svmtrain(trainY,Ptrain,cmd);

% Calculating the predicitve scores in the test cohort by NATIM model
[predict_label,accuracy,roc] = svmpredict(textY,Ptext,model);
[X,Y,THRE,AUC,OPTROCPT,SUBY,SUBYNAMES]=perfcurve(textY,roc,'1');

