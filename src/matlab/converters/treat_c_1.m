use_data=[]; 
use_label=[];
load('../data/test1.mat');

[numcases, numbatches]=size(features);
N=numcases;

data = [features_test ones(N,1)];
w1probs = data*w1;

test_features = w1probs(:,:);
test_labels = str2num(labels_test(:,4));
%scatter3(processed_data_conventional(:,1),processed_data_fuzzy(:,2),processed_data_fuzzy(:,3),1,processed_labels_fuzzy)
load('../data/training1.mat');

[numcases, numbatches]=size(features_training);
N=numcases;

data = [features_training ones(N,1)];
w1probs = data*w1;

training_features = w1probs(:,:);
training_labels = str2num(labels_training(:,4));

save(['../saved_features/feat_c_1L_' num2str(iter)], 'training_features', 'training_labels', 'test_features', 'test_labels')