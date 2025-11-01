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
% Data Created: 05/30/2025
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

% Validate that output folder Exist
function validateOutputFolder(outputFolder)
fprintf('Locating Output Folder...\n');
    if ~isfolder(outputFolder)
        mkdir(outputFolder);
    end
fprintf('The %s folder was found...\n', outputFolder);

end

% Validate File
function fullFileName = validateFile(fullFileName)
    if ~isfile(fullFileName)
    errorMessage = sprintf('Error: The following folder does not exist:\n%s\nPlease specify a new file.', fullFileName);
    uiwait(warndlg(errorMessage));
    fullFileName = uigetdir();
        if fullFileName == 0
            return;
        end
    end    
end

% Check File Name For Seizure Data
function seizureFlag = checkSeizures(fileName)
    seizureChecker = "Seizure";
    seizureFlag = contains(fileName, seizureChecker);
end

% Create Table
function myTable = createTable(rowSize, columnSize, variableNames, rowNames)
    myTable = table('Size', [rowSize, columnSize], 'VariableTypes', {'cell', 'double'}, ...
                    'VariableNames', variableNames,'RowNames', rowNames);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stage 1: Locate Signal Folders (.mat files)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Desired Node List
desiredNodes = {'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', ... 
                'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2','FP2-F8', 'F8-T8', ...
                'T8-P8', 'P8-O2', 'FZ-CZ', 'CZ-PZ','P7-T7', 'T7-FT9', 'FT9-FT10', ...
                'FT10-T8', 'T8-P8'};
totalNodes = length(desiredNodes);

% Preprocessed Signal Locations
signalFolders = {'D:\processedSignalsFFT', 'D:\processedSeizureSignalFFT'};
numberOfFolders = length(signalFolders);

% Validate Folders Exist
signalFolders = checkFolder(signalFolders, numberOfFolders);

% Desired Subfolders and Files (Base Data)
[baseFilePattern, baseDesiredFiles, baseFoldesize] = getFolderInfo(signalFolders{1});

% Desired Subfolders and Files (Seizure Data)
[seizureFilePattern, seizureDesiredFiles, seizureFoldesize] = getFolderInfo(signalFolders{2});

% Total Files
totalFiles = 10; %baseFoldesize + seizureFoldesize;

% Display Files
fprintf('Running Machine Learning Model Training Script...\n\n');
fprintf('Locating Input Files...\n');
fprintf('Total number of .mat files located within the baseline folder: %d\n', baseFoldesize);
fprintf('Total number of .mat files located within the seizure folder: %d\n\n', seizureFoldesize);

% Output Folder
outputFolder = 'D:\ML Preprocessed Data\';
validateOutputFolder(outputFolder);

fprintf('\nGenerating A Master Table for EEG Data...\n');

% Table Naming
variableNames = {'Data', 'SeizureStatus'};
numberOfColumns = length(variableNames);
rowNames = "Table_" + (1:totalFiles)';

% Create Master Table
masterTable = createTable(totalFiles, numberOfColumns, variableNames, rowNames);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stage 2: Folder Looping (Baseline Files)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loop Through Folder
for file = 1:5 %baseFoldesize
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Stage 2.1: File Retrieval
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    % Retrieve all desired files and folders
    [baseFileName, baseFolderName, baseFullName] = fileRetrieval(baseDesiredFiles, file);
    
    % Validate File
    baseFullName = validateFile(baseFullName);

    % Check File Name For Seizure Data
    base_seizureFlag = checkSeizures(baseFileName);
    
    % Display File Name and File Count
    fprintf('(%d / %d) ', file, totalFiles);
    fprintf(1, 'Now processing %s\n', baseFileName);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Stage 2.2: Table Insertion
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     
    % Load File
    load(baseFullName); % Variable Name: windowedEEG_RP_Table (Windows X Channels)
    
    % Index Control
    rowIndex = rowNames{file};
    
    % Check if Table is Available and Insert Data
    masterTable.Data{rowIndex} = windowedEEG_RP_Table;
    masterTable.SeizureStatus(rowIndex) = base_seizureFlag;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stage 3: Folder Looping (Seizure Files)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for file = 1:5 %seizureFoldesize
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Stage 3.1: File Retrieval
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
    % Retrieve all desired files and folders
    [seizureFileName, seizureFolderName, seizureFullName] = fileRetrieval(seizureDesiredFiles, file);
            
    % Validate File
    seizureFullName = validateFile(seizureFullName);
        
    % Check File Name For Seizure Data
    seizureFlag = checkSeizures(seizureFileName);
            
    % Display File Name and File Count
    realCount = file + 5; %baseFoldesize;
    fprintf('(%d / %d) ', realCount, totalFiles);
    fprintf(1, 'Now processing %s\n', seizureFileName);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Stage 3.2: Table Insertion
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%          
    % Load File
    load(seizureFullName); % Variable Name: windowedSeizureEEG_RP_Table (Windows X Channels)
        
    % Index Control
    rowIndex = rowNames{realCount};

    % Check if Table is Available and Insert Data
    masterTable.Data{rowIndex} = windowedSeizureEEG_RP_Table;
    masterTable.SeizureStatus(rowIndex) = seizureFlag;

end

fprintf('\nSuccessfully Generated Master Table...\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stage 4: Flatten Dataset
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Unifying Dataset...\n')

% Create Rows and Columns
featuredRows = []; % n X 110
labelVector = []; % n 

% Loop Through Master Table
for index = 1:10 %totalFiles
    fprintf('(%d / %d) Tables Processed\n', index, totalFiles);
    % Append Rows
    convertedArray = table2array(masterTable.Data{index});
    featuredRows = [featuredRows; convertedArray];

    % Append Columns
    matrixSize = size(convertedArray, 1);
    Labels = repmat(masterTable.SeizureStatus(index), matrixSize, 1);
    labelVector = [labelVector; Labels];
end    

fprintf('Dataset Unification Completed...\n\n')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stage 5: File Saving
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Saving Desired Variables...\n')
outputFile = 'EEG_ML_Preprocessing.mat';
save([outputFolder outputFile], 'featuredRows', 'labelVector',...
     'masterTable', '-v7.3') %'-v7.3' used for variables greater than 2 GB
fprintf('Saving Completed...\n\n')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('Script Has Finished...\n')
