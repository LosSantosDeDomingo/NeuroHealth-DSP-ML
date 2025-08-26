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
% https://physionet.org/content/chbmit/1.0.0/
% Research Paper "Design and Implementation of a RISC-V SoC for Real-Time Epilepsy
% Detection on FPGA" by Jiangwei He and Co.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Purpose of Program
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The purpose of this program is to collect and preprocess all the 
% files that can be found within the CHB-MIT Scalp EEG Database.
% Each respective file will be checked for desired EEG Channels
% (or nodes), once confirmed, each file will be turned from a 
% time table to a numerical matrix. After the conversion, the matrix
% will the will be vectorized to allow for a 3D matrix before
% saving and exporting to an output folder as a .mat file.
%
% An additional method was provided for the creation of the 3D matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Improvement Status
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Immediate Improvements for Current Version:
% --------------------------------------------
%  
%
% Possible Improvements for Later Version:
% -----------------------------------------
% (1) Create more error-handling functions
% (2) Check how to remove the unique name warning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Version Info
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Version: 1
% Data Created: 04/26/2025
% Last Revision: 06/27/2025
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Clear Workspace, Command Window, and Figures 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; % Clear Workspace memory
clc; % Clear Command Window
close all; % Close all figures
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stage 1: Database Location
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Desired Node List
desiredNodes = {'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', ... 
                'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2','FP2-F8', 'F8-T8', ...
                'T8-P8', 'P8-O2', 'FZ-CZ', 'CZ-PZ','P7-T7', 'T7-FT9', 'FT9-FT10', ...
                'FT10-T8', 'T8-P8'};
totalNodes = length(desiredNodes);
minNodes = ceil(totalNodes / 2);

% Database Location
databaseLocation = 'D:\chb-mit-scalp-eeg-database-1.0.0';

% Validate Location and Display
if ~isfolder(databaseLocation)
    errorMessage = sprintf('Error: The following folder does not exist:\n%s\nPlease specify a new folder.', databaseLocation);
    uiwait(warndlg(errorMessage));
    databaseLocation = uigetdir();
    if databaseLocation == 0
        return;
    end
end

% Desired Subfolders and Files
filePattern = fullfile(databaseLocation, '**/*.edf');
desiredFiles = dir(filePattern);
folderSize = length(desiredFiles);

% Display Files
fprintf('Total number of .edf files located within the folder: %d\n', folderSize);

% Output Folder
outputFolders = {'D:\ProcessedEEG', 'D:\ProcessedSeizureEEG'};
numberOfOutputFolders = length(outputFolders);

fprintf('Locating Output Folders...\n');
for outputSeizureFolder = 1:numberOfOutputFolders
    currentFolder = outputFolders{outputSeizureFolder};
    if ~isfolder(currentFolder)
        mkdir(currentFolder);
    end
    fprintf('The %s folder was found...\n', currentFolder);
end

fprintf('\nBegin Processing All EDF Files...\n');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stage 2 Folder Looping
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loop Through Folder
for file = 1:folderSize
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Stage 2.1: File Retrieval
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Retrieve all desired files and folders
    fileNames = desiredFiles(file).name;
    folderNames = desiredFiles(file).folder;
    fullFileName = fullfile(folderNames, fileNames);
    
    % Display File Name and File Count
    fprintf('(%d / %d) ', file, folderSize);
    fprintf(1, 'Now processing %s\n', fileNames);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Stage 2.2 Signal Capture
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Search Selected Signal Header in EDF File
    documentInfo = edfinfo(fullFileName);
    availableSignals = documentInfo.SignalLabels;
    
    % Check which requested channels are available
    isAvailable = ismember(desiredNodes, availableSignals);
    numAvailable = sum(isAvailable);
    if numAvailable < minNodes
        fprintf('Not a sufficient amount of nodes... (%d/%d)\n', numAvailable, totalNodes);
        continue; % Use continue to skip this file, not break
    end

    % Display Total Number of Nodes Found / Nodes Checked
    totalSignals = length(isAvailable); 
    fprintf('Total Nodes found (%d / %d)\n', totalSignals, totalNodes);

    % Collect Signal Data
    multiChannelTable = edfread(fullFileName,"SelectedSignals", desiredNodes);
 
    % Convert Signal Time Table into a Numerical Array
    multiChannelArray = table2array(multiChannelTable);
    singleMatrix = cellfun(@vertcat, multiChannelArray, 'UniformOutput', false); % Removes Cells and Places them Vertically
    singleMatrix = cell2mat(singleMatrix);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Stage 2.3: Vectorization/Sliding Window
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Calculate Total Windows and Channels
    samples = 256;
    numberOfWindows = floor(length(singleMatrix) / samples);
    numberOfChannels = width(singleMatrix);

    % Truncate the Single Matrix
    usableSamples = numberOfWindows * samples;
    singleMatrix = singleMatrix(1:usableSamples, :);

    % Create Matrix
    windowedEEG = zeros(numberOfChannels, samples, numberOfWindows);
    
    % Vectorize
    windowedEEG(:,:,:) = permute(reshape(singleMatrix, samples, numberOfWindows, numberOfChannels), [3 1 2]);

    % Alternative Method | Sliding Window
    % Nest For Loop to Fill-in Matrix
    %for channel = 1:numberOfChannels
    %    for window = 1:numberOfWindows
            % Calculate Starting and Ending Index
    %        startIndex = ((window - 1) * samples) + 1; 
    %        endIndex = samples * window;
    %        signalInterval = startIndex:endIndex;

            % Fill In Matrix Elements
%            windowedEEG(:, window, channel) = singleMatrix(signalInterval, channel);
 %       end
 %   end
  
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Stage 2.4 File Saving
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    % Build expected path to the .edf.seizures file
    seizurePath = [fullFileName, '.seizures'];

    % Check if the corresponding .seizures file exists
    if isfile(seizurePath)
        % Save Results as a .mat File
        fixedFileName = replace(fileNames, '.edf', '.mat');
        save(['D:\ProcessedSeizureEEG\' fixedFileName], "windowedEEG");
        fprintf('Window was successfully stored, moving to next file...\n')
    else
        % Save Results as a .mat File
        fixedFileName = replace(fileNames, '.edf', '.mat');
        save(['D:\ProcessedEEG\' fixedFileName], "windowedEEG");
        fprintf('Window was successfully stored, moving to next file...\n')
    end

    % Display When File is Completed
    fprintf('%s has been processed, moving on to the next file...\n\n\n', fileNames);  
end

% Display When Preprocessing Has Been Completed
fprintf('All files have been processed...');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
