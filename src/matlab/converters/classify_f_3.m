for iter = 1:30 
    %load 'data/dataset';
    rng('shuffle', 'twister');
    load '../data/test';
    load (['../saved_variables/classify_weightsf3_' num2str(iter)])

    %targets = labels_test(:,2:8);
    N = 10080;
    randomorder=randperm(N);
    
    data = features(randomorder, :);
    targets_temp = str2num(labels(:,4));
    targets = targets_temp(randomorder, :);
    data = [data ones(N,1)];
    w1probsL = 1./(1 + exp(-data*w1L)); 
    w1probsR = 1./(1 + exp(-data*w1R));
    w1probs = [((w1probsL + w1probsR)/2) ones(N,1)];
 
    w2probsL = 1./(1 + exp(-w1probs*w2L));
    w2probsR = 1./(1 + exp(-w1probs*w2R));
    w2probs = [((w2probsL + w2probsR)/2) ones(N,1)];

    w3probsL = 1./(1 + exp(-w2probs*w3L));
    w3probsR = 1./(1 + exp(-w2probs*w3R));
    w3probs = [((w3probsL + w3probsR)/2) ones(N,1)];

    targetoutL = exp(w3probs*w_classL);
    targetoutR = exp(w3probs*w_classR);
    targetout = (targetoutL + targetoutR)/2;
    targetout = targetout./repmat(sum(targetout,2),1,7);

    [I, J]=max(targetout,[],2);
    [I1, J1]=max(targets,[],2);
    counter=length(find(J==J1));
    correct = zeros(N, 1);
    correct(1) = counter;
    T = table(J1, J, correct);
    T.Properties.VariableNames = {'Real' 'Predicted' 'Correct'};
    writetable(T, ['classification_results/f3_' num2str(iter) '.csv']);
    clear
end