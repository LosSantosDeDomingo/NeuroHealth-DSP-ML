%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Background Information
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Publisher(s): Jose Caraballo
% School: Florida Atlantic University
% Professor: Dr. Hanqi Zhuang
% Sponsor: Dr. Sree Ranjani Rajendran
% Database: CHB-MIT Scalp EEG Database
% GitHub Repository Link: https://github.com/LosSantosDeDomingo/NeuroHealth-DSP-ML
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% References
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (1) Research Paper "Design and Implementation of a RISC-V SoC for Real-Time Epilepsy
%                     Detection on FPGA" by Jiangwei He and Co.
% (2) https://www.mathworks.com/help/
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Purpose of EEG_Model_Training
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The purpose of this program is to train a Support Vector Machine (SVM)
% model to classify seizure data by extracting features from EEG recorded
% brain signals. This model can also be used to simulate the overall
% behavior of the system's output when identifying possible seizure data.
% The model will also be implementing Kernel functions such as a Linear,
% Polynomial, and Radial Basis Function (RBF).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Improvement Status
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Immediate Improvements for Current Version
%
%
% Possible Improvements for Later Version
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Version Info
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Version: 1
% Data Created: 06/16/2025
% Last Revision: 10/16/2025
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Model Code
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Clear Workspace, Command Window, and Figures
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
clc;
close all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Folder Validation
function checkFolder(folder)
    if ~isfolder(folder)
        errorMessage = sprintf('Error: The following folder does not exist:\n%s\nPlease specify a new folder.', folder);
        uiwait(warndlg(errorMessage));
        databaseLocation = uigetdir();
        if databaseLocation == 0
            return;
        end
    end
end

% Retrieve Folder Information
function [filePattern, desiredFiles, size] = getFolderInfo(inputFolder)
    filePattern = fullfile(inputFolder, '**/*.mat');
    desiredFiles = dir(filePattern);
    size = length(desiredFiles);
end

% Retrieve all desired files and folders
function [fileName, folderName, fullName] = fileRetrieval(desiredFiles, file)
    fileName = desiredFiles(file).name;
    folderName = desiredFiles(file).folder;
    fullName = fullfile(folderName, fileName);
end

% Validate File
function validateFile(fullFileName)
    if ~isfile(fullFileName)
    errorMessage = sprintf('Error: The following folder does not exist:\n%s\nPlease specify a new file.', fullFileName);
    uiwait(warndlg(errorMessage));
    fullFileName = uigetdir();
        if fullFileName == 0
            return;
        end
    end    
end

% Validate that output folder Exist
function validateOutputFolder(outputFolders)
    fprintf('Locating Output Folders...\n');
    numberOfOutputFolders = size(outputFolders,2);
    for folder = 1:numberOfOutputFolders
        currentFolder = outputFolders{folder};
        if ~isfolder(currentFolder)
            mkdir(currentFolder);
        end
        fprintf('The %s folder was found...\n', currentFolder);
    end
end

% Accuracy Metric
function accuracy = calculateAccuracyMetric(yTest, yPredict)
    accuracy = sum(yPredict == yTest) / length(yTest);
end

% Confusion Matrix
function [confusionMatrix, TP, FP, TN, FN] = calculateConfusionMat(yTest, yPredict)
    confusionMatrix = confusionmat(yTest, yPredict);
    TP = confusionMatrix(2,2);
    FP = confusionMatrix(1,2);
    TN = confusionMatrix(1,1);
    FN = confusionMatrix(2,1);
end

% Precision Metric
function precision = calculatePrecisionMetric(TP, FP)
    precision = TP / (TP + FP);
end

% Recall Metric
function recall = calculateRecallMetric(TP, FN)
    recall = TP / (TP + FN);
end

% F1-Score Metric
function f1Score = calculateF1Score(precision, recall)
    f1Score = 2 * (precision * recall) / (precision + recall);
end

% Display Metrics
function displayMetrics(cvLoss, accuracy, precision, recall, f1Score)
    fprintf("Cross Validation Loss: %.4f\n", cvLoss);
    fprintf("Accuracy: %.4f\n", accuracy);
    fprintf("Precision: %.4f\n", precision);
    fprintf("Recall: %.4f\n", recall);
    fprintf("F1-Score: %.4f\n\n", f1Score);
end

% Calculate Class Weights
function [trainedSVM, yPredict] = trainSVM(xTrainNorm, yTrain, xTestNorm, kernelType)
    % Display Training Status
    fprintf('Training SVM (%s Kernel) Classifier...\n', upper(kernelType));

    % Train SVM
    if kernelType == "sigmoid"
        trainedSVM = fitcsvm(xTrainNorm, yTrain, ...
                             'KernelFunction', kernelType, ...
                             'Standardize', false,...
                             'KernelScale', 1.0);

    elseif kernelType == "polynomial"
        trainedSVM = fitcsvm(xTrainNorm, yTrain, ...
                             'KernelFunction', kernelType, ...
                             'PolynomialOrder', 2, ...
                             'BoxConstraint', 1, ...
                             'Standardize', false,...
                             'KernelScale','auto');  
    else
        trainedSVM = fitcsvm(xTrainNorm, yTrain, ...
                             'KernelFunction', kernelType, ...
                             'Standardize', false,...
                             'KernelScale','auto');        
    end

    % Predict
    yPredict = predict(trainedSVM, xTestNorm);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stage 1: Collect Preprocessed Files (.mat files)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Running Model Training Script...\n\n');

% Preprocessed Signal Locations
inputFolder = 'F:\ML Preprocessed Data';
numberOfFolders = length(inputFolder);

% Validate Folders Exist
checkFolder(inputFolder);

% Collect Desired Subfolders and Files
[baseFilePattern, baseDesiredFiles, baseFoldesize] = getFolderInfo(inputFolder);

% Collect Desired File
[fileName, folderName, fullName] = fileRetrieval(baseDesiredFiles, 1); % 1 Because there was only one file

% Verify File
validateFile(fullName);

% Display Files
fprintf('Locating Input Files...\n');
fprintf('Total number of .mat files located within the ML Preprocessing Folder: %d\n\n', baseFoldesize);

% Output Folder
outputFolder = {'F:\nomalizationParameters', 'F:\linearSVM_EEG', 'F:\RBFSVM_EEG', 'F:\polynomialSVM_EEG'};
validateOutputFolder(outputFolder);


% Load File
load(fullName);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stage 2: Data Prep
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\n\nModel Training Has Begun...\n\n');

% Preparing Train/Test Split
rng(42); % Ensures that data splitting is reproducible.
cvHold = cvpartition(labelVector, 'Holdout', 0.2);
trainIndex  = training(cvHold);
testIndex   = test(cvHold);

% Trained Variables
xTrain = featuredRows(trainIndex, :);
yTrain = labelVector(trainIndex);

% Test Variables
xTest = featuredRows(testIndex, :);
yTest = labelVector(testIndex);

% Normalize Dataset
xTrainMean                  = mean(xTrain);
xTrainSTD                   = std(xTrain);
xTrainSTD(xTrainSTD == 0)   = 1;
xTrainNorm                  = (xTrain - xTrainMean) ./ xTrainSTD;
xTestNorm                   = (xTest - xTrainMean) ./ xTrainSTD;

% Training Samples
fprintf('Checking for Training and Testing Samples...\n');
fprintf('Training Samples: %d\n', size(xTrain,1));
fprintf('Testing Samples: %d\n\n', size(xTest,1));
fprintf('Checking Amount of Features...\n');
fprintf('Features: %d\n\n', size(xTrainNorm,2));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stage 3: Model Training
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SVM Linear
fprintf('SVM Linear Training has begun...\n');
[SVM_Linear, yPredictSVM_Linear] = trainSVM(xTrainNorm, yTrain, xTestNorm, 'linear');
fprintf('SVM Linear Training has ended...\n\n');

% SVM RBF
fprintf('SVM RBF Training has begun...\n');
[SVM_RBF, yPredictSVM_RBF] = trainSVM(xTrainNorm, yTrain, xTestNorm, 'rbf');
fprintf('SVM RBF Training has ended...\n\n');

% SVM Polynomial
fprintf('SVM Polynomial Training has begun...\n');
[SVM_Poly, yPredictSVM_Poly] = trainSVM(xTrainNorm, yTrain, xTestNorm, 'polynomial');
fprintf('SVM Polynomial Training has ended...\n\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stage 4: Cross Validations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Setup Cross Validation
fprintf('Cross Validating Models...\n');
cvKFold = cvpartition(yTrain, "KFold", 10); % 10 iterations

% Cross Validate Models
cvModelLinear   = crossval(SVM_Linear, 'CVPartition', cvKFold);
cvModelRBF      = crossval(SVM_RBF, 'CVPartition', cvKFold);
cvModelPoly     = crossval(SVM_Poly, 'CVPartition', cvKFold);

% Check for loss
cvModelLinearLoss   = kfoldLoss(cvModelLinear);
cvModelRBFLoss      = kfoldLoss(cvModelRBF);
cvModelPolyLoss     = kfoldLoss(cvModelPoly);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stage 5: Model Evaluation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Accuracy Metric
fprintf('Calculating Accuracy Metrics...\n');
accuracyMetricLinear    = calculateAccuracyMetric(yTest, yPredictSVM_Linear);
accuracyMetricRBF       = calculateAccuracyMetric(yTest, yPredictSVM_RBF);
accuracyMetricPoly      = calculateAccuracyMetric(yTest, yPredictSVM_Poly);

% Confusion Matrix
fprintf('Calculating Confusion Matrix...\n');
[confusionMatrixLinear, linear_TP, linear_FP, linear_TN, linear_FN] = calculateConfusionMat(yTest, yPredictSVM_Linear);
[confusionMatrixRBF, rbf_TP, rbf_FP, rbf_TN, rbf_FN]                = calculateConfusionMat(yTest, yPredictSVM_RBF);
[confusionMatrixPoly, poly_TP, poly_FP, poly_TN, poly_FN]           = calculateConfusionMat(yTest, yPredictSVM_Poly);

% Precision Metric
fprintf('Calculating Precision Metrics...\n');
precisionMetricLinear   = calculatePrecisionMetric(linear_TP, linear_FP);
precisionMetricRBF      = calculatePrecisionMetric(rbf_TP, rbf_FP);
precisionMetricPoly     = calculatePrecisionMetric(poly_TP, poly_FP);

% Recall Metric
fprintf('Calculating Recall Metrics...\n');
recallMetricLinear  = calculateRecallMetric(linear_TP, linear_FN);
recallMetricRBF     = calculateRecallMetric(rbf_TP, rbf_FN);
recallMetricPoly    = calculateRecallMetric(poly_TP, poly_FN);

% F1 Score Metric
fprintf('Calculating F1 Score Metrics...\n\n');
f1MetricLinear  = calculateF1Score(precisionMetricLinear, recallMetricLinear);
f1MetricRBF     = calculateF1Score(precisionMetricRBF, recallMetricRBF);
f1MetricPoly    = calculateF1Score(precisionMetricPoly, recallMetricPoly);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stage 6: Display Results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Displaying Training Metrics...\n');

% Linear
fprintf('SVM Linear Metrics:\n');
displayMetrics(cvModelLinearLoss, accuracyMetricLinear, precisionMetricLinear, recallMetricLinear, f1MetricLinear);

% RBF
fprintf('SVM RBF Metrics:\n');
displayMetrics(cvModelRBFLoss, accuracyMetricRBF, precisionMetricRBF, recallMetricRBF, f1MetricRBF);

% Poly
fprintf('SVM Polynomial Metrics:\n');
displayMetrics(cvModelPolyLoss, accuracyMetricPoly, precisionMetricPoly, recallMetricPoly, f1MetricPoly);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stage 7: Save Inference Parameter Results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Saving Training Results...\n')

% Save Normalization Parameters
normalization_params = struct();
normalization_params.xTrainMean = xTrainMean;
normalization_params.xTrainSTD = xTrainSTD;
save(fullfile(outputFolder{1}, 'normalization_params.mat'), 'normalization_params');
fprintf('Saved normalization parameters\n');

% Linear
SVM_Linear            = fitPosterior(SVM_Linear, xTrainNorm, yTrain);
compactLinearModel    = compact(SVM_Linear);
outputFile1           = 'LinearSVM.mat';
saveLearnerForCoder(compactLinearModel, fullfile(outputFolder{2}, outputFile1));
fprintf('Saved Linear SVM\n');

% RBF
SVM_RBF             = fitPosterior(SVM_RBF, xTrainNorm, yTrain);
compactPolyModel    = compact(SVM_RBF);
outputFile2         = 'RBFSVM.mat';
saveLearnerForCoder(compactPolyModel, fullfile(outputFolder{3}, outputFile2));
fprintf('Saved RBF SVM\n');

% Poly
SVM_Poly            = fitPosterior(SVM_Poly, xTrainNorm, yTrain);
compactPolyModel    = compact(SVM_Poly);
outputFile3         = 'PolySVM.mat';
saveLearnerForCoder(compactPolyModel, fullfile(outputFolder{4}, outputFile3));
fprintf('Saved Polynomial SVM\n');

fprintf('Saving Completed...\n\n')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('Script Has Finished...\n')
