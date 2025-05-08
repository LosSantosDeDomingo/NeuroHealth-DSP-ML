%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Background Information
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Publisher(s): Jose Caraballo
% School: Florida Atlantic University
% Professor: Dr. Hanqi Zhuang
% Sponsor: Dr. Sree Ranjani Rajendran
% Database: CHB-MIT Scalp EEG Database
% GitHub Repository Link: [Need to Complete]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% References
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% https://physionet.org/content/chbmit/1.0.0/
% Research Paper "Design and Implementation of a RISC-V SoC for Real-Time Epilepsy
% Detection on FPGA" by Jiangwei He and Co.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Purpose of Program
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The purpose of this program is to test how to preprocess raw EEG signal
% data.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Improvement Status
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Immediate Improvements for Current Version:
% --------------------------------------------
% Scan through all the nodes in a single file
    % Do something to the collect section
%  
%
% Possible Improvements for Later Version:
% -----------------------------------------
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Version Info
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Version: 1
% Data Created: 04/26/2025
% Last Revision: 05/07/2025
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Clear Workspace, Command Window, and Figures 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; % Clear Workspace memory
clc; % Clear Command Window
close all; % Close all figures
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EEG Processing Code
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Specify folder location
myFolder = 'D:\chb-mit-scalp-eeg-database-1.0.0\chb-mit-scalp-eeg-database-1.0.0';

% Node List
nodeList = {'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', ...
            'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', ...
            'T8-P8', 'P8-O2', 'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9', ...
            'FT9-FT10', 'FT10-T8', 'T8-P8'};

% Validate folder
if ~isfolder(myFolder)
    errorMessage = sprintf('Error: The following folder does not exist:\n%s\nPlease specify a new folder.', myFolder);
    uiwait(warndlg(errorMessage));
    myFolder = uigetdir();
    if myFolder == 0
        return;
    end
end

% Get a list of all desired file names within the folder
filePattern = fullfile(myFolder, '**/*.edf');
desiredFiles = dir(filePattern);
folderSize = length(desiredFiles);

% Location of Scrapper Output
outputFolder = 'D:\ProcessedEEG';
if ~isfolder(outputFolder)
    mkdir(outputFolder);
end

% Main Loop
for i = 1:folderSize
    % Retrieve all desired files and folders
    fileNames = desiredFiles(i).name;
    folderNames = desiredFiles(i).folder;
    fullFileName = fullfile(folderNames, fileNames);
    fprintf(1, 'Now processing %s\n', fullFileName);    

    % Check Channel Availability
    documentInfo = edfinfo(fullFileName);
    availableSignals = documentInfo.SignalLabels;
    isAvailable = ismember(nodeList, availableSignals);

    if ~all(isAvailable)
        fprintf('Missing Channels in %s:\n', fileNames);
        disp(nodeList(~isAvailable));
        continue;
    end

    % Extract patient ID from path
    tokens = regexp(fullFileName, 'chb\d{2}', 'match');
    if isempty(tokens)
        warning('No patient ID found in: %s', fullFileName);
        continue;
    end
    patientID = tokens{1};
    excelFile = fullfile(outputFolder, [patientID '.xlsx']);
    
    % Read signals and resolve duplicate names
    try
        % Read all signals from file
        multiChannelMatrix = edfread(fullFileName, 'AssignUniqueLabels', true);
    
        % Filter only the signals from nodeList
        signalFilter = ismember(multiChannelMatrix.Properties.VariableNames, nodeList);
        multiChannelMatrix = multiChannelMatrix(:, signalFilter);

        if width(multiChannelMatrix) == 0
            warning('No matching channels found in %s after filtering.', fileNames);
            continue;
        end
    
        % Rename duplicate variable names if needed
        varNames = multiChannelMatrix.Properties.VariableNames;
        [~, ia, ic] = unique(varNames, 'stable');
    
        if length(ia) < length(varNames)
            for k = 1:length(varNames)
                if sum(strcmp(varNames{k}, varNames)) > 1
                    varNames{k} = [varNames{k} '_' num2str(k)];
                end
            end
            multiChannelMatrix.Properties.VariableNames = varNames;
        end
    catch ME
        warning("Failed to read %s: %s", fileNames, ME.message);
        continue;
    end

    % Convert to Numerical Matrix
    signalMatrix = table2array(multiChannelMatrix);  % Directly use numeric matrix

    % Window the Signal
    samples = 256;
    sampleTimes = seconds((0:samples - 1) / samples);
    numberOfWindows = floor(size(signalMatrix, 1) / samples);
    numberOfChannels = size(signalMatrix, 2);
    windowedEEG = zeros(samples, numberOfWindows, numberOfChannels);

    % Segment signals into windows
    for channel = 1:numberOfChannels
        for window = 1:numberOfWindows
            startIndex = (window - 1) * samples + 1;
            endIndex = window * samples;
            windowedEEG(:, window, channel) = signalMatrix(startIndex:endIndex, channel);
        end
    end

    % Write one sheet per channel (stack all windows)
    for channel = 1:numberOfChannels
        if channel > numel(varNames)
            warning('Channel index %d exceeds available variable names.', channel);
            continue;
        end
    
        nodeName = varNames{channel};
        cleanNode = matlab.lang.makeValidName(nodeName);  % Safe name for Excel
        
        % Preallocate columns
        fullAmplitude = zeros(samples * numberOfWindows, 1);
        fullTime = seconds(zeros(samples * numberOfWindows, 1));
        fullWindow = zeros(samples * numberOfWindows, 1);

        % Stack all windows into long columns
        for window = 1:numberOfWindows
            startIndex = (window - 1) * samples + 1;
            endIndex = window * samples;

            fullAmplitude(startIndex:endIndex) = windowedEEG(:, window, channel);
            fullTime(startIndex:endIndex) = sampleTimes;
            fullWindow(startIndex:endIndex) = window;
        end
    
        % Build and write timetable
        channelTimeTable = timetable(fullTime, fullAmplitude, fullWindow, ...
                                      'VariableNames', {'Amplitude', 'Window'});
    
        try
            writetimetable(channelTimeTable, excelFile, 'Sheet', cleanNode);
            fprintf('✓ Saved %s (Channel: %s)\n', excelFile, cleanNode);
        catch writeErr
            warning("Failed to write sheet %s: %s", cleanNode, writeErr.message);
        end
    end
end

% Check if Files were Processed Properly
fprintf('\nFinished processing all files.\n');
excelFiles = dir(fullfile(outputFolder, '*.xlsx'));
fprintf('Total Excel files created: %d\n', numel(excelFiles));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
