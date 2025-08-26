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
% 
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
% Data Created: 06/24/2025
% Last Revision: 
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
function checkFolder(folder, numberOfFolders)
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stage 1: Collect Preprocessed Files (.mat files)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Running Model Validity Script...\n\n');

% Preprocessed Signal Locations
signalFolders = {'D:\testingBaseFFT', 'D:\testingSeizureFFT', 'D:\EEG Model Training'};
numberOfFolders = length(signalFolders);

% Validate Folders Exist
checkFolder(signalFolders, numberOfFolders);

% Desired Subfolders and Files (Base Data)
[baseFilePattern, baseDesiredFiles, baseFoldesize] = getFolderInfo(signalFolders{1});

% Desired Subfolders and Files (Seizure Data)
[seizureFilePattern, seizureDesiredFiles, seizureFoldesize] = getFolderInfo(signalFolders{2});

% Desired Subfolders and Files (Models)
[modelFilePattern, modelDesiredFiles, modelFoldesize] = getFolderInfo(signalFolders{3});

% Total Files
totalFiles = baseFoldesize + seizureFoldesize;

% Preallocate result vectors before the loop
baselinePredictions = zeros(baseFoldesize, 1);
baselineScores = zeros(baseFoldesize, 1);
baselineTruth = zeros(baseFoldesize, 1);  % All 0 for baseline

% Preallocate seizure predictions and truth
seizurePredictions = zeros(seizureFoldesize, 1);
seizureScores = zeros(seizureFoldesize, 1);
seizureTruth = ones(seizureFoldesize, 1);  % Label 1 for seizure
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stage 2: Gather Baseline Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for file = 1:baseFoldesize
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Stage 2.1: File Retrieval
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    % Retrieve all desired files and folders
    [baseFileName, baseFolderName, baseFullName] = fileRetrieval(baseDesiredFiles, file);
    [inferenceFileName, inferenceFolderName, inferenceFullName] = fileRetrieval(modelDesiredFiles, 1);
    [modelFileName, modelFolderName, modelFullName] = fileRetrieval(modelDesiredFiles, 2);    
    
    % Validate File
    validateFile(baseFullName);
    
    % Display File Name and File Count
    fprintf('Baseline File (%d / %d): %s\n', file, baseFoldesize, baseFileName);

    % Load Files
    load(baseFullName); % Variable Name: windowedEEG_RP_Table (Windows X Channels)
    load(inferenceFullName);
    load(modelFullName);
    
    % Load Mini Model
    miniModel = loadLearnerForCoder(modelFullName);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Stage 2.2: Check Performance
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
    % Convert Table to an Array
    featuresFFT = table2array(windowedEEG_RP_Table);
    featuresFFT = (featuresFFT - xTrainMean) ./ xTrainSTD;
    
    % Assume majority vote if predicting on multiple windows
    [label, score] = predict(miniModel, featuresFFT);    
    majorityLabel = mode(label);
    meanScore = mean(score(:,2));  % score for class "1"
    meanScore = max(0, min(1, meanScore));  % clamp between 0 and 1
    
    baselinePredictions(file) = majorityLabel;
    baselineScores(file) = meanScore;
    
    % Display likelihood of a seizure
    fprintf('Seizure Likelihood: %.2f%%\n', meanScore * 100);
    if meanScore >= 0.70
        fprintf('High Seizure Risk (%.2f%% confidence)\n\n', meanScore * 100);
    elseif meanScore >= 0.50
        fprintf('Moderate Seizure Risk (%.2f%% confidence)\n\n', meanScore * 100);
    elseif meanScore >= 0.30
        fprintf('Low Seizure Risk (%.2f%% confidence)\n\n', meanScore * 100);
    else
        fprintf('Very Low Seizure Risk (%.2f%% confidence)\n\n', meanScore * 100);
    end
    
    % Flag uncertain predictions
    if meanScore >= 0.45 && meanScore <= 0.55
        fprintf('Uncertain classification: Consider rechecking this signal.\n\n');
    end
end

baselineAccuracy = sum(baselinePredictions == baselineTruth) / baseFoldesize;
fprintf('Baseline Accuracy: %.2f%%\n\n', baselineAccuracy * 100);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stage 3: Gather Seizure Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for file = 1:seizureFoldesize
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Stage 3.1: File Retrieval
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    [seizureFileName, seizureFolderName, seizureFullName] = fileRetrieval(seizureDesiredFiles, file);
    validateFile(seizureFullName);
    
    fprintf('Seizure File (%d / %d): %s\n', file, seizureFoldesize, seizureFileName);
    load(seizureFullName);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Stage 3.2: Check Performance
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
    % Normalize features same way
    featuresFFT = table2array(windowedEEG_RP_Table);
    featuresFFT = (featuresFFT - xTrainMean) ./ xTrainSTD;
    
    [label, score] = predict(miniModel, featuresFFT);
    label(label == -1) = 0;
    majorityLabel = mode(label);
    meanScore = mean(score(:,2));
    meanScore = max(0, min(1, meanScore));  % clamp between 0 and 1
    
    seizurePredictions(file) = majorityLabel;
    seizureScores(file) = meanScore;

    % Display likelihood of a seizure
    fprintf('Seizure Likelihood: %.2f%%\n', meanScore * 100);
    if meanScore >= 0.70
        fprintf('High Seizure Risk (%.2f%% confidence)\n\n', meanScore * 100);
    elseif meanScore >= 0.50
        fprintf('Moderate Seizure Risk (%.2f%% confidence)\n\n', meanScore * 100);
    elseif meanScore >= 0.30
        fprintf('Low Seizure Risk (%.2f%% confidence)\n\n', meanScore * 100);
    else
        fprintf('Very Low Seizure Risk (%.2f%% confidence)\n\n', meanScore * 100);
    end
    
    % Flag uncertain predictions
    if meanScore >= 0.45 && meanScore <= 0.55
        fprintf('Uncertain classification: Consider rechecking this signal.\n\n');
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stage 4: Final Accuracy and Summary
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
allPredictions = [baselinePredictions; seizurePredictions];
allTruth = [baselineTruth; seizureTruth];

seizureAccuracy = sum(seizurePredictions == seizureTruth) / seizureFoldesize;
fprintf('Seizure Accuracy: %.2f%%\n\n', seizureAccuracy * 100);
overallAccuracy = sum(allPredictions == allTruth) / length(allTruth);
fprintf('Overall Accuracy (Baseline + Seizure): %.2f%%\n\n\n', overallAccuracy * 100);
fprintf('Script Has Finished...\n');
