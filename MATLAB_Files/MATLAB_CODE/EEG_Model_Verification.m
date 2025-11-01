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
% Purpose of EEG_Model_Verification
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Needs to be filled
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
% Data Created: 08/24/2025
% Last Revision: 10/18/2025
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
function folder = checkFolder(folder, numberOfFolders)
    for inputFolder = 1:numberOfFolders
        currentFolder = folder{inputFolder};
        if ~isfolder(currentFolder)
            errorMessage = sprintf('Error: The following folder does not exist:\n%s\nPlease specify a new folder.', currentFolder);
            uiwait(warndlg(errorMessage));
            selectedFolder = uigetdir();
            if selectedFolder == 0
                return;
            end
            folder{inputFolder} = selectedFolder;
        end
    end
end

% Retrieve Folder Information
function [filePattern, desiredFiles, size] = getFolderInfo(inputFolder)
    filePattern  = fullfile(inputFolder, '**/*.mat');
    desiredFiles = dir(filePattern);
    size         = length(desiredFiles);
end

% Retrieve all desired files and folders
function [fileName, folderName, fullName] = fileRetrieval(desiredFiles, file)
    fileName   = desiredFiles(file).name;
    folderName = desiredFiles(file).folder;
    fullName   = fullfile(folderName, fileName);
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

function features = extractFeatures(struct)
    if isfield(struct, 'windowedEEG_RP_Table')
        features = table2array(struct.windowedEEG_RP_Table);
    elseif isfield(struct, 'windowedSeizureEEG_RP_Table')
        features = table2array(struct.windowedSeizureEEG_RP_Table);
    else
        fieldNames = fieldnames(struct);
        for i = 1:numel(fieldNames)
            if istable(struct.(fieldNames{i}))
                features = table2array(struct.(fieldNames{i}));
                return;
            end
        end
        error('extractFeatures:NoTable', 'No expected table variable found.');
    end
end

function [mu, sigma, predictorOrder] = modelMeta(model)
    % Locate Mu
    if isprop(model,'Mu') && ~isempty(model.Mu)
        mu = model.Mu(:).'; 
    else 
        mu = 0; 
    end
    
    % Locate Sigma
    if isprop(model,'Sigma') && ~isempty(model.Sigma)
        sigma = model.Sigma(:).'; 
    else 
        sigma = 1; 
    end
    
    % Locate Predictor Names
    if isprop(model,'ExpandedPredictorNames') && ~isempty(model.ExpandedPredictorNames)
        predictorOrder = string(model.ExpandedPredictorNames);
    elseif isprop(model,'PredictorNames') && ~isempty(model.PredictorNames)
        predictorOrder = string(model.PredictorNames);
    else
        predictorOrder = [];
    end
end

function scores = modelScores(model, features, mu, sigma, predictorOrder)
    % Normalize with guard for zero sigma
    if numel(mu)==size(features,2) && numel(sigma)==size(features,2)
        features = (features - mu) ./ sigma;
        features(:, sigma==0) = 0;
    end
    
    % Predict using table with names when order is known
    if ~isempty(predictorOrder) && numel(predictorOrder)==size(features,2)
        T = array2table(features, 'VariableNames', cellstr(predictorOrder));
        [label, score] = predict(model, T);
    else
        [label, score] = predict(model, features);
    end
    
    % Check for posterior
    hasPosterior = isprop(model,'ScoreTransform') && ~isempty(model.ScoreTransform) && ~strcmpi(model.ScoreTransform,'none');
    positiveTokens = ["1","seizure","positive","Seizure","True"];
    
    if hasPosterior && ~isempty(score)
        if isprop(model,'ClassNames') && ~isempty(model.ClassNames)
            classNames  = string(model.ClassNames);
            hit = ismember(classNames, positiveTokens);
            if any(hit) 
                positiveColumn = find(hit,1,'first'); 
            else
                positiveColumn = min(2, size(score,2));
            end
        else
            positiveColumn = min(2, size(score,2));
        end
        scores = score(:, min(positiveColumn, size(score,2)));
    else
        labels = string(label);
        scores = double(ismember(labels, positiveTokens));
    end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stage 1: Prepare Necessary Files (.mat files)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Running Inference Model...\n\n');

% Input Folders
signalFolders = { ...
    'F:\nomalizationParameters', 'F:\testingBaseFFT', 'F:\testingSeizureFFT', ...
    'F:\linearSVM_EEG', 'F:\RBFSVM_EEG', 'F:\polynomialSVM_EEG'...
    };
numberOfFolders = length(signalFolders);

% Validate folders (may update paths)
signalFolders = checkFolder(signalFolders, numberOfFolders);

% Output Folder
outputFolder = 'F:\inferenceResults\';
validateOutputFolder(outputFolder);

% Desired Subfolders and Files
[~, normParamDesiredFiles, ~]              = getFolderInfo(signalFolders{1});
[~, baseDesiredFiles,    baseFoldesize]    = getFolderInfo(signalFolders{2});
[~, seizureDesiredFiles, seizureFoldesize] = getFolderInfo(signalFolders{3});
[~, linearModelDesiredFiles, ~]            = getFolderInfo(signalFolders{4});
[~, rbfModelDesiredFiles,    ~]            = getFolderInfo(signalFolders{5});
[~, polyModelDesiredFiles,   ~]            = getFolderInfo(signalFolders{6});

% Retrieve Necessary Path Names
[~, ~, normParamFullName]  = fileRetrieval(normParamDesiredFiles, 1);
[~, ~, linearModelFullName]  = fileRetrieval(linearModelDesiredFiles, 1);
[~, ~, rbfModelFullName]     = fileRetrieval(rbfModelDesiredFiles,    1);
[~, ~, polyModelFullName]    = fileRetrieval(polyModelDesiredFiles,   1);

% Load Models
fprintf('Loading models...\n');
linearSVM   = loadLearnerForCoder(linearModelFullName);
rbfSVM      = loadLearnerForCoder(rbfModelFullName);
polySVM     = loadLearnerForCoder(polyModelFullName);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stage 2: Gather Parameters Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load normalization parameters
fprintf('Loading training normalization parameters\n');
load(normParamFullName, "normalization_params");
trainingMu = normalization_params.xTrainMean;
trainingSigma = normalization_params.xTrainSTD;
    
fprintf('\nTraining Normalization Statistics:\n');
fprintf('  Mean (mu) range:   [%.4e, %.4e]\n', min(trainingMu), max(trainingMu));
fprintf('  Std (sigma) range: [%.4e, %.4e]\n', min(trainingSigma), max(trainingSigma));
fprintf('  Number of features: %d\n\n', numel(trainingMu));

% Extract predictor order
[~, ~, predictorOrderL] = modelMeta(linearSVM);
[~, ~, predictorOrderR] = modelMeta(rbfSVM);
[~, ~, predictorOrderP] = modelMeta(polySVM);

% Preallocate result vectors and predictions
% Baseline
yPredBaseLinear   = zeros(baseFoldesize,1);
yPredBaseRBF      = zeros(baseFoldesize,1);
yPredBasePoly     = zeros(baseFoldesize,1);
predBaseLinear    = false(baseFoldesize,1);
predBaseRBF       = false(baseFoldesize,1);
predBasePoly      = false(baseFoldesize,1);

% Seizure
yPredSeizureLinear   = zeros(seizureFoldesize,1);
yPredSeizureRBF      = zeros(seizureFoldesize,1);
yPredSeizurePoly     = zeros(seizureFoldesize,1);
predSeizureLinear    = false(seizureFoldesize,1);
predSeizureRBF       = false(seizureFoldesize,1);
predSeizurePoly      = false(seizureFoldesize,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stage 3: Perform Inference on Baseline Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('============================================================\n');
fprintf('BASELINE DATA INFERENCE\n');
fprintf('============================================================\n\n');

for file = 1:baseFoldesize
    [baseFileName, ~, baseFullName] = fileRetrieval(baseDesiredFiles, file);
    validateFile(baseFullName);
    fprintf('Baseline File (%d / %d): %s\n', file, baseFoldesize, baseFileName);

    baseLoad = load(baseFullName);
    featuresBaselineFFT = extractFeatures(baseLoad);

    % Linear
    scoresBaseL = modelScores(linearSVM, featuresBaselineFFT, trainingMu, trainingSigma, predictorOrderL);
    meanBaseLinear = max(0, min(1, mean(scoresBaseL, 'omitnan')));
    yPredBaseLinear(file) = meanBaseLinear;
    predBaseLinear(file)  = meanBaseLinear >= 0.5;
    fprintf('  Baseline Likelihood (Linear): %.2f%%\n', 100*meanBaseLinear);

    % RBF
    scoresBaseR = modelScores(rbfSVM, featuresBaselineFFT, trainingMu, trainingSigma, predictorOrderR);
    meanBaseRBF = max(0, min(1, mean(scoresBaseR, 'omitnan')));
    yPredBaseRBF(file)    = meanBaseRBF;
    predBaseRBF(file)     = meanBaseRBF >= 0.5;
    fprintf('  Baseline Likelihood (RBF):    %.2f%%\n', 100*meanBaseRBF);

    % Poly
    scoresBaseP = modelScores(polySVM, featuresBaselineFFT, trainingMu, trainingSigma, predictorOrderP);
    meanBasePoly = max(0, min(1, mean(scoresBaseP, 'omitnan')));
    yPredBasePoly(file)   = meanBasePoly;
    predBasePoly(file)    = meanBasePoly >= 0.5;
    fprintf('  Baseline Likelihood (Poly):   %.2f%%\n\n', 100*meanBasePoly);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stage 4: Perform Inference on Seizure Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('============================================================\n');
fprintf('SEIZURE DATA INFERENCE\n');
fprintf('============================================================\n\n');

for file = 1:seizureFoldesize
    [seizureFileName, ~, seizureFullName] = fileRetrieval(seizureDesiredFiles, file);
    validateFile(seizureFullName);
    fprintf('Seizure File  (%d / %d): %s\n', file, seizureFoldesize, seizureFileName);

    seizureLoad = load(seizureFullName);
    featuresSeizureFFT = extractFeatures(seizureLoad);

    % Linear
    scoresSeizureL = modelScores(linearSVM, featuresSeizureFFT, trainingMu, trainingSigma, predictorOrderL);
    meanSeizureLinear = max(0, min(1, mean(scoresSeizureL, 'omitnan')));
    yPredSeizureLinear(file) = meanSeizureLinear;
    predSeizureLinear(file)  = meanSeizureLinear >= 0.5;
    fprintf('  Seizure Likelihood (Linear): %.2f%%\n', 100*meanSeizureLinear);

    % RBF
    scoresSeizureR = modelScores(rbfSVM, featuresSeizureFFT, trainingMu, trainingSigma, predictorOrderR);
    meanSeizureRBF = max(0, min(1, mean(scoresSeizureR, 'omitnan')));
    yPredSeizureRBF(file)    = meanSeizureRBF;
    predSeizureRBF(file)     = meanSeizureRBF >= 0.5;
    fprintf('  Seizure Likelihood (RBF):    %.2f%%\n', 100*meanSeizureRBF);

    % Poly
    scoresSeizureP = modelScores(polySVM, featuresSeizureFFT, trainingMu, trainingSigma, predictorOrderP);
    meanSeizurePoly = max(0, min(1, mean(scoresSeizureP, 'omitnan')));
    yPredSeizurePoly(file)   = meanSeizurePoly;
    predSeizurePoly(file)    = meanSeizurePoly >= 0.5;
    fprintf('  Seizure Likelihood (Poly):   %.2f%%\n\n', 100*meanSeizurePoly);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stage 5: Save Results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Saving Desired Variables...\n')
outputFile = 'inferenceVariables.mat';
save([outputFolder, outputFile], ...
     'meanBaseLinear', 'meanBaseRBF', 'meanBasePoly',...
     'meanSeizureLinear', 'meanSeizureRBF', 'meanSeizurePoly')
fprintf('Saving Completed...\n\n')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Script Has Finished...\n');
