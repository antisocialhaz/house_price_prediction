%% 🏠 House Price Prediction using Machine Learning
% Author: Hamza Zaz
% Date: October 2025
% Description:
% Predicts housing prices using multiple ML algorithms and compares performance.
% Dataset: Boston Housing Dataset

clearvars; clc; close all;

%% STEP 1: Load Dataset
url = 'https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv';
data = readtable(url);

disp('Dataset preview:');
disp(head(data));

%% STEP 2: Preprocessing
% Remove missing data
data = rmmissing(data);

% Separate features and target
X = data{:, 1:end-1};
y = data.MEDV;

% Normalize features
X = normalize(X);

% Split data (80% train, 20% test)
cv = cvpartition(size(X,1),'HoldOut',0.2);
XTrain = X(training(cv),:);
yTrain = y(training(cv),:);
XTest  = X(test(cv),:);
yTest  = y(test(cv),:);

%% STEP 3: Train Models
disp('Training models...');

mdl_lin = fitlm(XTrain, yTrain); % Linear Regression
mdl_ridge = fitrlinear(XTrain, yTrain, 'Learner','leastsquares', 'Lambda',1, 'Regularization','ridge');
mdl_lasso = fitrlinear(XTrain, yTrain, 'Learner','leastsquares', 'Lambda',0.1, 'Regularization','lasso');
mdl_tree = fitrtree(XTrain, yTrain);
mdl_rf = fitrensemble(XTrain, yTrain, 'Method','Bag');

models = {mdl_lin, mdl_ridge, mdl_lasso, mdl_tree, mdl_rf};
names = {'Linear','Ridge','Lasso','Tree','Random Forest'};

%% STEP 4: Evaluate Models
fprintf('\nModel Performance on Test Data:\n');
for i = 1:length(models)
    yPred = predict(models{i}, XTest);
    rmse = sqrt(mean((yTest - yPred).^2));
    R2 = corr(yTest, yPred)^2;
    fprintf('%-15s | RMSE = %.3f | R² = %.3f\n', names{i}, rmse, R2);
end

%% STEP 5: Visualize Results (Best Model: Random Forest)
bestModel = mdl_rf;
yPred = predict(bestModel, XTest);

figure;
scatter(yTest, yPred, 60, 'filled')
xlabel('Actual Prices ($1000s)');
ylabel('Predicted Prices ($1000s)');
title('Actual vs Predicted House Prices (Random Forest)');
grid on;

% Line of perfect prediction
hold on; plot([min(yTest), max(yTest)], [min(yTest), max(yTest)], 'r--', 'LineWidth', 1.5);

%% STEP 6: Feature Importance
imp = predictorImportance(bestModel);
figure;
bar(imp);
xlabel('Feature Index');
ylabel('Importance');
title('Feature Importance - Random Forest');
grid on;

%% STEP 7: Save Model
saveLearnerForCoder(bestModel, 'HousePricePredictor');
disp('✅ Model saved as HousePricePredictor.mat');

%% STEP 8: Example Prediction
% Example house: CRIM=0.03, ZN=18, INDUS=2.3, CHAS=0, NOX=0.4, RM=6.5, AGE=60, DIS=4.2, RAD=4, TAX=270, PTRATIO=18, B=390, LSTAT=6
sampleHouse = [0.03 18 2.3 0 0.4 6.5 60 4.2 4 270 18 390 6];
sampleHouse = normalize(sampleHouse, 'center', mean(X), 'scale', std(X));
predictedPrice = predict(bestModel, sampleHouse);
fprintf('\nPredicted Price for sample house: $%.2fk\n', predictedPrice);