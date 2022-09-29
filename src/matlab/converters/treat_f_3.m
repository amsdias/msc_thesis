use_data=[]; 
use_label=[];
load('../data/test.mat');

[numcases, numbatches]=size(features);
N=numcases;

data = [features ones(N,1)];
w1probsL = 1./(1 + exp(-data*w1L));
w1probsR = 1./(1 + exp(-data*w1R));
w1probs = [((w1probsL + w1probsR)/2) ones(N,1)];

w2probsL = 1./(1 + exp(-w1probs*w2L));
w2probsR = 1./(1 + exp(-w1probs*w2R));
w2probs = [((w2probsL + w2probsR)/2) ones(N,1)];

w3probsL = w2probs*w3L;
w3probsR = w2probs*w3R;
w3probs = (w3probsL + w3probsR)/2;

test_features = w3probs(:,:);
test_labels = str2num(labels(:,4));

load('../data/training.mat');
[numcases, numbatches]=size(features);
N=numcases;

data = [features ones(N,1)];
w1probsL = 1./(1 + exp(-data*w1L));
w1probsR = 1./(1 + exp(-data*w1R));
w1probs = [((w1probsL + w1probsR)/2) ones(N,1)];

w2probsL = 1./(1 + exp(-w1probs*w2L));
w2probsR = 1./(1 + exp(-w1probs*w2R));
w2probs = [((w2probsL + w2probsR)/2) ones(N,1)];

w3probsL = w2probs*w3L;
w3probsR = w2probs*w3R;
w3probs = (w3probsL + w3probsR)/2;

training_features = w3probs(:,:);
training_labels = str2num(labels(:,4));

save(['saved_features/feat_f_3L_' num2str(iter)], 'training_features', 'training_labels', 'test_features', 'test_labels')
