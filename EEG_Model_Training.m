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
% Purpose of Program
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The purpose of this program is to train a Support Vector Machine (SVM)
% model to classify seizure data by extracting features from EEG recorded
% brain signals. This model can also be used to simulate the overall
% behavior of the system's output when identifying possible seizure data.
% The model will also be implementing Kernel functions such as a Sigmoid,
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
% Last Revision: 06/16/2025
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

% Accuracy Metric
function accuracy = calculateAccuracyMetric(yTest, yPredict)
    accuracy = sum(yPredict == yTest) / length(yTest);
end

% Confusion Matrix
function [confusionMatrix, TP, FP, TN, FN] = calculateConfusionMat(yTest, yPredict)
    confusionMatrix = confusionmat(yTest, yPredict);
    TP = confusionMatrix(2,2);
    FP = confusionMatrix(1,2);
    TN = confusionMatrix(2,1);
    FN = confusionMatrix(1,1);
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stage 1: Collect Preprocessed Files (.mat files)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Running Model Training Script...\n\n');

% Preprocessed Signal Locations
inputFolder = 'D:\ML Preprocessed Data';
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
fprintf('Total number of .mat files located within the ML Preprcessing Folder: %d\n\n', baseFoldesize);

% Load File
load(fullName);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stage 2: Data Prep
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Model Training Has Begun...\n\n');
% Split Dataset
rng(42);
cv = cvpartition(labelVector, 'HoldOut', 0.2);

xTrain = featuredRows(cv.training, :);
yTrain = labelVector(cv.training);

xTest = featuredRows(cv.test, :);
yTest = labelVector(cv.test);

% Normalize Dataset
xTrainMean = mean(xTrain);
xTrainSTD = std(xTrain);
xTrainNorm = (xTrain - xTrainMean) ./ xTrainSTD;
xTestNorm  = (xTest - xTrainMean) ./ xTrainSTD;

% Training Samples
fprintf('Checking for Training and Testing Samples...\n');
fprintf('Training Samples: %d\n', size(xTrain,1));
fprintf('Testing Samples: %d\n\n', size(xTest,1));
fprintf('Checking Amount of Features...\n');
fprintf('Features: %d\n\n', size(xTrainNorm,2));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stage 3: Model Training
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SVM RBF
fprintf('Training SVM RBF Classifier...\n');
SVM_RBF = fitcsvm(xTrainNorm, yTrain,"KernelFunction", "rbf");
yPredictSVM_RBF = predict(SVM_RBF, xTestNorm);

% SVM Polynomial
fprintf('Training SVM Polynomial Classifier...\n');
SVM_Poly = fitcsvm(xTrainNorm, yTrain, "KernelFunction", "polynomial");
yPredictSVM_Poly = predict(SVM_Poly, xTestNorm);

% SVM Sigmoid
fprintf('Training SVM Sigmoid Classifier...\n\n');
SVM_Sigmoid = fitcsvm(xTrainNorm, yTrain, "KernelFunction", "sigmoid");
yPredictSVM_Sigmoid = predict(SVM_Sigmoid, xTestNorm);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stage 4: Model Evaluation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Accuracy Metric
accuracyMetricRBF = calculateAccuracyMetric(yTest, yPredictSVM_RBF);
accuracyMetricPoly = calculateAccuracyMetric(yTest, yPredictSVM_Poly);
accuracyMetricSigmoid = calculateAccuracyMetric(yTest, yPredictSVM_Sigmoid);

% Confusion Matrix
[confusionMatrixRBF, rbf_TP, rbf_FP, rbf_TN, rbf_FN] = calculateConfusionMat(yTest, yPredictSVM_RBF);
[confusionMatrixPoly, poly_TP, poly_FP, poly_TN, poly_FN] = calculateConfusionMat(yTest, yPredictSVM_Poly);
[confusionMatrixSigmoid, sig_TP, sig_FP, sig_TN, sig_FN] = calculateConfusionMat(yTest, yPredictSVM_Sigmoid);

% Precision Metric
precisionMetricRBF = calculatePrecisionMetric(rbf_TP, rbf_FP);
precisionMetricPoly = calculatePrecisionMetric(poly_TP, poly_FP);
precisionMetricSigmoid = calculatePrecisionMetric(sig_TP, sig_FP);

% Recall Metric
recallMetricRBF = calculateRecallMetric(rbf_TP, rbf_FN);
recallMetricPoly = calculateRecallMetric(poly_TP, poly_FN);
recallMetricSigmoid = calculateRecallMetric(sig_TP, sig_FN);

% F1 Score Metric
f1MetricRBF = calculateF1Score(precisionMetricRBF, recallMetricRBF);
f1MetricPoly = calculateF1Score(precisionMetricPoly, recallMetricPoly);
f1MetricSigmoid = calculateF1Score(precisionMetricSigmoid, recallMetricSigmoid);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stage 5: Display Results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Displaying Training Metrics...\n');

% RBF
fprintf('SVM RBF Metrics:\n');
fprintf("Accuracy: %d\n", accuracyMetricRBF);
fprintf("Precision: %d\n", precisionMetricRBF);
fprintf("Recall: %d\n", recallMetricRBF);
fprintf("F1-Score: %d\n\n", f1MetricRBF);

% Poly
fprintf('SVM Polynomial Metrics:\n');
fprintf("Accuracy: %d\n", accuracyMetricPoly);
fprintf("Precision: %d\n", precisionMetricPoly);
fprintf("Recall: %d\n", recallMetricPoly);
fprintf("F1-Score: %d\n\n", f1MetricPoly);

% Sigmoid
fprintf('SVM Sigmoid Metrics:\n');
fprintf("Accuracy: %d\n", accuracyMetricSigmoid);
fprintf("Precision: %d\n", precisionMetricSigmoid);
fprintf("Recall: %d\n", recallMetricSigmoid);
fprintf("F1-Score: %d\n\n", f1MetricSigmoid);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stage 6: Save Training Results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
