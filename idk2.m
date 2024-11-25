close all
clear;
clc;
%%
train = readtable('dataclean2.csv');
train.PassengerId = [];
train.GroupCount = [];
train.Name = [];
train.CabinDeck = [];
train.CabinNum = [];
train.CabinSide = [];

test = readtable('test_space.csv');
test.Cabin = [];
test.Name = [];
test.HomePlanet = grp2idx(test.HomePlanet); % Converts Earth, Europa, Mars to 1, 2, 3
test.CryoSleep = double(strcmp(test.CryoSleep, 'True')); % 1 if 'True', 0 if 'False'
test.Destination = double(strcmp(test.Destination, '55 Cancri e ')) + ...
                   2 * double(strcmp(test.Destination, 'PSO J318.5-22 ')) + ...
                   3 * double(strcmp(test.Destination, 'TRAPPIST-1e '));
test.VIP = double(strcmp(test.VIP, 'True')); % 1 if 'True', 0 if 'False'
test.Age(isnan(test.Age)) = mean(test.Age, 'omitnan');

edges = [0, 12, 19, 32, 59, inf];  % Define the age range boundaries
categories = {'0', '1', '2', '3', '4'};  % String categories for each range

% Map the ages in the train and test sets to the appropriate numeric category
train.Age = discretize(train.Age, edges, 'Categorical', categories);
test.Age = discretize(test.Age, edges, 'Categorical', categories);

% Convert to numeric (discretize will return a categorical type, so we convert it to double)
test.Age = double(test.Age);
train.Age = double(train.Age);

columnTypes1 = varfun(@class, train, 'OutputFormat', 'table');
columnTypes2 = varfun(@class, test, 'OutputFormat', 'table');
train
test
%%
X_train = train;
X_train.Transported = [];
Y_train = train.Transported;

X_test = test;
% X_test = test{:, 2:end-1};

RandomForest = TreeBagger(1000, X_train, Y_train, 'Method', 'classification', ...
                          'MinLeafSize', 5, 'MaxNumSplits', 100, 'OOBPrediction', 'on');
[Y_predict, scores] = predict(RandomForest, X_test);
Y_predict = str2double(Y_predict);

Y_train_predict = predict(RandomForest, X_train);
Y_train_predict = str2double(Y_train_predict);
Accuracy1 = sum(Y_train_predict == Y_train) / length(Y_train);

Transported = Y_predict;
Transported = string(Transported);  % Convert logical to string
Transported = replace(Transported, {'1', '0'}, {'True', 'False'});  % Convert to "True" or "False"
Transported

T = table(test.PassengerId, Transported, 'VariableNames', {'PassengerId', 'Transported'});
writetable(T, 'NEWsubmission.csv');

figure(1);
Stats = confusionmatStats(Y_train, Y_train_predict);
fprintf('Classification Report:\n')
fprintf('\nAccuracy: %.20f\n', Accuracy1);
fprintf('Precision: %.20f\n', Stats.precision);
fprintf('Recall: %.20f\n', Stats.recall);
fprintf('F1-Score: %.20f\n', Stats.F1);

%%
function stats = confusionmatStats(trueLabels, predictedLabels)
    ConfusionMatrix = confusionmat(trueLabels, predictedLabels);
    
    tp = ConfusionMatrix(2, 2);
    fp = ConfusionMatrix(1, 2);
    fn = ConfusionMatrix(2, 1);
    tn = ConfusionMatrix(1, 1);

    precision = tp / (tp + fp);
    recall = tp / (tp + fn);
    F1 = 2 * (precision * recall) / (precision + recall);

    stats.precision = precision;
    stats.recall = recall;
    stats.F1 = F1;


    ConfusionMatrix
    confusionchart(ConfusionMatrix)
end