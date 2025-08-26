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
    errorMessage = sprintf('Error: The following file does not exist:\n%s\nPlease specify a new file.', fullFileName);
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
signalFolders = {'D:\testingBaseFFT', 'D:\testingSeizureFFT', 'D:\modelParametersEEG', 'D:\miniModelEEG'};
numberOfFolders = length(signalFolders);

% Validate Folders Exist
checkFolder(signalFolders, numberOfFolders);

% Desired Subfolders and Files
[~, baseDesiredFiles, baseFoldesize] = getFolderInfo(signalFolders{1});
[~, seizureDesiredFiles, seizureFoldesize] = getFolderInfo(signalFolders{2});
[~, paramDesiredFiles, paramFoldesize] = getFolderInfo(signalFolders{3});
[~, modelDesiredFiles, modelFoldesize] = getFolderInfo(signalFolders{4});

% Total Files
totalFiles = baseFoldesize + seizureFoldesize;

% Load Files
miniModel = loadLearnerForCoder(modelDesiredFiles);
load(paramDesiredFiles);
posCol = 2;

% Error Handling for a STD of 0
xTrainSTD(xTrainSTD == 0) = 1;

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
    [baseFileName, ~, baseFullName] = fileRetrieval(baseDesiredFiles, file);  
    
    % Validate File
    validateFile(baseFullName);
    
    % Display File Name and File Count
    fprintf('Baseline File (%d / %d): %s\n', file, baseFoldesize, baseFileName);

    % Load Files
    clear windowEEG_RP_Table windowedSeizureEEG_RP_Table
    load(baseFullName); % Variable Name: windowedEEG_RP_Table (Windows X Channels)
    featuresFFT = table2array(windowedEEG_RP_Table);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Stage 2.2: Check Performance
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
    % Normalize Data
    featuresFFT = (featuresFFT - xTrainMean) ./ xTrainSTD;
    featuresFFT(:,xTrainSTD == 0) = 0;
    
    % Predict scores for all windows
    [~, score] = predict(miniModel, featuresFFT);
    if isempty(score), error('Model did not return scores.'); end
    if size(score,2) < posCol, posCol = 1; end

    % Mean "seizure-likeness" and decision
    meanScore = mean(score(:, posCol));
    meanScore = max(0, min(1, meanScore));
    baselineScores(file)      = meanScore;
    baselinePredictions(file) = meanScore >= 0.5;

    fprintf('Seizure Likelihood: %.2f%%\n\n', meanScore*100);
end
baselineAccuracy = sum(baselinePredictions == baselineTruth) / max(1, baseFoldesize);
fprintf('Baseline Accuracy: %.2f%%\n\n', baselineAccuracy * 100);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stage 3: Gather Seizure Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for file = 1:seizureFoldesize
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Stage 3.1: File Retrieval
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    [seizureFileName, ~, seizureFullName] = fileRetrieval(seizureDesiredFiles, file);
    validateFile(seizureFullName);
    
    fprintf('Seizure File (%d / %d): %s\n', file, seizureFoldesize, seizureFileName);

    % Load seizure data
    clear windowEEG_RP_Table windowedSeizureEEG_RP_Table
    load(seizureFullName);
    featuresFFT = table2array(windowedSeizureEEG_RP_Table);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Stage 3.2: Check Performance
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
    % Normalize Data
    featuresFFT = (featuresFFT - xTrainMean) ./ xTrainSTD;
    featuresFFT(:, xTrainSTD==0) = 0;
    
    % Predict scores for all windows
    [~, score] = predict(miniModel, featuresFFT);
    if isempty(score), error('Model did not return scores.'); end
    if size(score,2) < posCol, posCol = 1; end
   
    % Mean "seizure-likeness" and decision
    meanScore = mean(score(:, posCol));
    meanScore = max(0, min(1, meanScore));
    seizureScores(file)      = meanScore;
    seizurePredictions(file) = meanScore >= 0.5;

    fprintf('Seizure Likelihood: %.2f%%\n\n', meanScore*100);
end
seizureAccuracy = sum(seizurePredictions == seizureTruth) / max(1, seizureFoldesize);
fprintf('Seizure Accuracy: %.2f%%\n', (100 * seizureAccuracy));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stage 4: Final Accuracy and Summary
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
allPredictions = [baselinePredictions; seizurePredictions];
allTruth = [baselineTruth; seizureTruth];
overallAccuracy = sum(allPredictions == allTruth) / max(1, numel(allTruth));
fprintf('Overall Accuracy: %.2f%%\n\n', (100 * overallAccuracy));

fprintf('Script Has Finished...\n');
