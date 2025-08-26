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
% Last Revision: 06/24/2025
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
function validateOutputFolder(outputFolder)
fprintf('Locating Output Folder...\n');
    if ~isfolder(outputFolder)
        mkdir(outputFolder);
    end
fprintf('The %s folder was found...\n', outputFolder);

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
function displayMetrics(accuracy, precision, recall, f1Score)
    fprintf("Accuracy: %.2f\n", accuracy);
    fprintf("Precision: %.2f\n", precision);
    fprintf("Recall: %.2f\n", recall);
    fprintf("F1-Score: %.2f\n\n", f1Score);
end

% Calculate Class Weights
function [trainedSVM, yPredict] = trainWeightedSVMModel(xTrainNorm, yTrain, xTestNorm, kernelType)
    % Display Training Status
    fprintf('Training SVM (%s Kernel) Classifier...\n', upper(kernelType));

    % Unique Classes
    % classLabels = unique(yTrain);
    % numClasses = numel(classLabels);

    % Class Counts (Using logical indexing for speed)
    % classCounts = arrayfun(@(x) sum(yTrain == x), classLabels);

    % Class Weights (Inverse frequency)
    % classWeights = sum(classCounts) ./ (numClasses * classCounts);

    % Sample Weights
    % sampleWeights = arrayfun(@(x) classWeights(classLabels == x), yTrain);

    % Train SVM
    if kernelType == "sigmoid"
        trainedSVM = fitcsvm(xTrainNorm, yTrain, ...
                             'KernelFunction', kernelType, ...
                             'Standardize', true,...
                             'KernelScale', 1.0);

    elseif kernelType == "polynomial"
        trainedSVM = fitcsvm(xTrainNorm, yTrain, ...
                             'KernelFunction', kernelType, ...
                             'PolynomialOrder', 2, ...
                             'BoxConstraint', 1, ...
                             'Standardize', true,...
                             'KernelScale','auto');  
    else
        trainedSVM = fitcsvm(xTrainNorm, yTrain, ...
                             'KernelFunction', kernelType, ...
                             'Standardize', true,...
                             'KernelScale','auto');        
    end

    % Predict
    yPredict = predict(trainedSVM, xTestNorm);
end

% Collect Inference Parameters
function [bias, supportVector, alpha, labels, coefficients] = calculateParameters(model, yTrain)
    bias = model.Bias;
    supportVector = model.SupportVectors;
    alpha = model.Alpha;

    % Ensure yTrain is column vector
    if isrow(yTrain)
        yTrain = yTrain';
    end

    % Labels for support vectors
    labels = yTrain(model.IsSupportVector);

    % Sanity check
    if length(alpha) ~= length(labels)
        warning("Alpha and label length mismatch");
    end

    % Compute coefficients
    coefficients = alpha .* labels;
end

% Display Inference Parameters
function displayParameters(bias, supportVector, alpha, labels, coefficients)
    fprintf("Bias: %.4f\n", bias);
    
    % Display Support Vectors
    numSupportVectors = size(supportVector, 1);
    fprintf("Number of Support Vectors: %d\n", numSupportVectors);
    
    if numSupportVectors > 0
        fprintf("Support Vectors (first 5 rows):\n");
        disp(supportVector(1:min(5, numSupportVectors), :));
    else
        fprintf("Support Vectors: None found.\n");
    end

    % Display Alpha
    if ~isempty(alpha)
        fprintf("Alpha Coefficients (first 10):\n");
        disp(alpha(1:min(10, end)));
    else
        fprintf("Alpha Coefficients: Empty\n");
    end

    % Display Labels
    if ~isempty(labels)
        fprintf("Labels of Support Vectors (first 10):\n");
        disp(labels(1:min(10, end)));
    else
        fprintf("Labels of Support Vectors: Empty\n");
    end

    % Display Coefficients
    if ~isempty(coefficients)
        fprintf("Coefficients (first 10):\n");
        disp(coefficients(1:min(10, end)));
    else
        fprintf("Coefficients: Empty\n");
    end
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
fprintf('Total number of .mat files located within the ML Preprocessing Folder: %d\n\n', baseFoldesize);

% Output Folder
outputFolder = 'D:\EEG Model Training\';
validateOutputFolder(outputFolder);

% Load File
load(fullName);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stage 2: Data Prep
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Model Training Has Begun...\n\n');
%cv = cvpartition(labelVector, 'HoldOut', 0.2); % Adjusts the ratio between basline and seizure

% Preparing Train/Test Split
rng(42); % Ensures that data splitting is reproducible.
featureSize = size(featuredRows, 1);
index = randperm(featureSize);
splitIndex = round(0.8 * featureSize);
trainIndex = index(1:splitIndex);
testIndex = index(splitIndex+1:end);

% Trained Variables
xTrain = featuredRows(trainIndex, :);
yTrain = labelVector(trainIndex);

% Test Variables
xTest = featuredRows(testIndex, :);
yTest = labelVector(testIndex);

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
% SVM Linear (Only used to confirm script works)
fprintf('SVM Linear Training has begun...\n');
[SVM_Linear, yPredictSVM_Linear] = trainWeightedSVMModel(xTrainNorm, yTrain, xTestNorm, 'linear');
fprintf('SVM Linear Training has ended...\n\n');

% SVM RBF
fprintf('SVM RBF Training has begun...\n');
[SVM_RBF, yPredictSVM_RBF] = trainWeightedSVMModel(xTrainNorm, yTrain, xTestNorm, 'rbf');
fprintf('SVM RBF Training has ended...\n\n');

% SVM Polynomial
fprintf('SVM Polynomial Training has begun...\n');
[SVM_Poly, yPredictSVM_Poly] = trainWeightedSVMModel(xTrainNorm, yTrain, xTestNorm, 'polynomial');
fprintf('SVM Polynomial Training has ended...\n\n');

% SVM Sigmoid
% fprintf('SVM Sigmoid Training has begun...\n');
% [SVM_Sigmoid, yPredictSVM_Sigmoid] = trainWeightedSVMModel(xTrainNorm, yTrain, xTestNorm, 'sigmoid');
% fprintf('SVM Sigmoid Training has ended...\n\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stage 4: Model Evaluation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Accuracy Metric
fprintf('Calculating Accuracy Metrics...\n');
accuracyMetricRBF = calculateAccuracyMetric(yTest, yPredictSVM_RBF);
accuracyMetricPoly = calculateAccuracyMetric(yTest, yPredictSVM_Poly);
% accuracyMetricSigmoid = calculateAccuracyMetric(yTest, yPredictSVM_Sigmoid);

% Confusion Matrix
fprintf('Calculating Confusion Matrix...\n');
[confusionMatrixRBF, rbf_TP, rbf_FP, rbf_TN, rbf_FN] = calculateConfusionMat(yTest, yPredictSVM_RBF);
[confusionMatrixPoly, poly_TP, poly_FP, poly_TN, poly_FN] = calculateConfusionMat(yTest, yPredictSVM_Poly);
% [confusionMatrixSigmoid, sig_TP, sig_FP, sig_TN, sig_FN] = calculateConfusionMat(yTest, yPredictSVM_Sigmoid);

% Precision Metric
fprintf('Calculating Precision Metrics...\n');
precisionMetricRBF = calculatePrecisionMetric(rbf_TP, rbf_FP);
precisionMetricPoly = calculatePrecisionMetric(poly_TP, poly_FP);
% precisionMetricSigmoid = calculatePrecisionMetric(sig_TP, sig_FP);

% Recall Metric
fprintf('Calculating Recall Metrics...\n');
recallMetricRBF = calculateRecallMetric(rbf_TP, rbf_FN);
recallMetricPoly = calculateRecallMetric(poly_TP, poly_FN);
% recallMetricSigmoid = calculateRecallMetric(sig_TP, sig_FN);

% F1 Score Metric
fprintf('Calculating F1 Score Metrics...\n\n');
f1MetricRBF = calculateF1Score(precisionMetricRBF, recallMetricRBF);
f1MetricPoly = calculateF1Score(precisionMetricPoly, recallMetricPoly);
% f1MetricSigmoid = calculateF1Score(precisionMetricSigmoid, recallMetricSigmoid);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stage 5: Display Results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Displaying Training Metrics...\n');

% RBF
fprintf('SVM RBF Metrics:\n');
displayMetrics(accuracyMetricRBF, precisionMetricRBF, recallMetricRBF, f1MetricRBF);

% Poly
fprintf('SVM Polynomial Metrics:\n');
displayMetrics(accuracyMetricPoly, precisionMetricPoly, recallMetricPoly, f1MetricPoly);

% Sigmoid
% fprintf('SVM Sigmoid Metrics:\n');
% displayMetrics(accuracyMetricSigmoid, precisionMetricSigmoid, recallMetricSigmoid, f1MetricSigmoid);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stage 6: Calculate Inference Parameter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Calculating Inference Parameters...\n');

% RBF
[bias_RBF, supportV_RBF, alpha_RBF, labels_RBF, coefficients_RBF] = calculateParameters(SVM_RBF, yTrain);

% Polynomial
[bias_Poly, supportV_Poly, alpha_Poly, labels_Poly, coefficients_Poly] = calculateParameters(SVM_Poly, yTrain);

% Sigmoid
% [bias_Sig, supportV_Sig, alpha_Sig, labels_Sig, coefficients_Sig] = calculateParameters(SVM_Sigmoid, yTrain);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stage 7: Inference Parameter Results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Displaying Inference Parameters Results...\n');

% RBF
fprintf('SVM RBF Inference Parameters:\n');
displayParameters(bias_RBF, supportV_RBF, alpha_RBF, labels_RBF, coefficients_RBF);

% Polynomial
fprintf('SVM Polynomial Inference Parameters:\n');
displayParameters(bias_Poly, supportV_Poly, alpha_Poly, labels_Poly, coefficients_Poly);

% Sigmoid
% fprintf('SVM Sigmoid Inference Parameters:\n');
% displayParameters(bias_Sig, supportV_Sig, alpha_Sig, labels_Sig, coefficients_Sig);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stage 8: Save Inference Parameter Results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Saving Training Results...\n')
outputFile = 'EEG_Trained_ML_Results.mat';

% Save Results of Training
save(fullfile(outputFolder, outputFile), ...
    "xTrainMean", "xTrainSTD", ...
     "accuracyMetricRBF", "accuracyMetricPoly",...
     "confusionMatrixRBF", "confusionMatrixPoly",...
     "precisionMetricRBF", "precisionMetricPoly",...
     "recallMetricRBF", "recallMetricPoly",...
     "f1MetricRBF", "f1MetricPoly",...
     "bias_RBF", "supportV_RBF", "alpha_RBF", "labels_RBF", "coefficients_RBF",...
     "bias_Poly", "supportV_Poly", "alpha_Poly", "labels_Poly", "coefficients_Poly");

% Save miniature polynomial SVM model for coder inference
compactPolyModel = compact(SVM_Poly);
outputFile2 = 'miniModelPolySVM.mat';
saveLearnerForCoder(compactPolyModel, fullfile(outputFolder, outputFile2));

fprintf('Saving Completed...\n\n')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('Script Has Finished...\n')
