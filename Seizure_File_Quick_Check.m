%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Background Information
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Publisher(s): ChatGPT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Purpose of Program
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The purpose of this program is to check which files are associated
% with seizures. This program was used to check for both, so the user
% can manually arrange the files.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Clear Workspace, Command Window, and Figures 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; % Clear Workspace memory
clc; % Clear Command Window
close all; % Close all figures
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main Code
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Folder where .edf files are stored
databaseLocation = 'D:\chb-mit-scalp-eeg-database-1.0.0\chb-mit-scalp-eeg-database-1.0.0';

% Get all .edf files
edfFiles = dir(fullfile(databaseLocation, '**', '*.edf'));

% Initialize cell arrays to hold categorized files 
seizureFiles = {};
baselineFiles = {};

% Loop through each .edf file
for k = 1:length(edfFiles)
    edfFile = edfFiles(k);
    edfPath = fullfile(edfFile.folder, edfFile.name);

    % Build expected path to the .edf.seizures file
    seizurePath = [edfPath, '.seizures'];

    % Check if the corresponding .seizures file exists
    if isfile(seizurePath)
        seizureFiles{end+1} = edfPath;
    else
        baselineFiles{end+1} = edfPath;
    end
end

% Display summary
fprintf('Total EDF files found: %d\n', length(edfFiles));
fprintf('Files with seizure annotations: %d\n', numel(seizureFiles));
fprintf('Files without seizure annotations: %d\n', numel(baselineFiles));

fprintf('\nFiles with Seizure Annotations:\n');
for i = 1:numel(seizureFiles)
    fprintf('%d: %s\n', i, seizureFiles{i});
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
