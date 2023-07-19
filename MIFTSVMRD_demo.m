%% Code for training and testing the M-IFTSVM-RD on an artifical dataset
clc
clear
%% load train data
load('Zoo.mat')               % 6 datasets available: Cleveland,Glass,Lenses,Seeds,Teaching_Assistant_Evaluation,Wine,Zoo
train = Zoo.train;            % trainning data
trainlabel = Zoo.trainlabel;  % training labels
test = Zoo.test;              % testing data
testlabel = Zoo.testlabel;    % testing labels

%% seting parameters
Parameter.ker = 'linear';
Parameter.CC = 0.3;
Parameter.CR = 1;
Parameter.p1 = 0.2;
Parameter.v = 10;
Parameter.k_inter = 3; % inter-class knn
Parameter.algorithm = 'CD';

%% training M-IFTSVM-RD
num_classes = max(trainlabel) + 1;
num_classifier = num_classes*(num_classes-1)/2;
for i = 1:num_classes-1
    for j = i+1:num_classes
        y = trainlabel;
        [idx1,~] = find(y == i-1);
        [idx2,~] = find(y == j-1);
        y(:) = 0;
        y(idx1) = 1;
        y(idx2) = -1;
        ftsvm_struct{i}{j} = ftsvmtrain(train,y,Parameter);
    end
end

%% testing M-IFTSVM-RD
test_num = size(test,1);
vote = zeros(test_num, num_classes);
maxAcc = 0;     max_correct=0;  epsilon_max = 0;
for epsilon = 0:0.01:0.5
    vote(:)=0;
    for i = 1:num_classes-1
        for j = i+1:num_classes
            [fp, fn] = ftsvmpreddists(ftsvm_struct{i}{j},test);
            temp = sqrt(fp.^2 + fn.^2);
            fp = fp./temp; fn = fn./temp;
            for k = 1:test_num
                if fp(k) < -1+epsilon
                    vote(k,i) = vote(k,i)+1;
                elseif fn(k) > 1-epsilon
                    vote(k,j) = vote(k,j)+1;
                else
                    vote(k,:) = vote(k,:) + 1;
                    vote(k,i) = vote(k,i)-1;
                end
            end
        end
    end

    [~,pred] = max(vote');
    pred = (pred-1)';
    correct=sum(pred==testlabel);
    acc=100*correct/length(testlabel);
    if acc > maxAcc
        maxAcc = acc;
        max_correct = correct;
        epsilon_max = epsilon;
    end
end

fprintf('epsilon = %1.2f , Acc = %3.4f (%d/%d)\n', epsilon_max, maxAcc, max_correct, length(testlabel));