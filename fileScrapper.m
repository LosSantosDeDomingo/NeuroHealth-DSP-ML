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
% The purpose of this program is to go through all the files from the
% CHB-MIT Scalp EEG Database. Each individual file will be read, processed,
% and concatenated on a master file that contains the data of all cases.
% This program was designed with the intent of streamlining the FFT & SVM
% process so each file doesn't have to be read every single time when the
% program runs. Additionally, by keep the program separate, I am ensuring
% that the main program can be reused for later testing and verification.
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
% Data Created: 04/26/2025
% Last Revision: 04/26/2025
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Model Code
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Clear Workspace, Command Window, and Figures 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
clc;
close all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Collect, Process, and Save a Sequence of Files
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Node String Array
nodeList = {'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2','FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 'FZ-CZ', 'CZ-PZ','P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8', 'T8-P8'};

% Specify folder location
myFolder = 'D:\chb-mit-scalp-eeg-database-1.0.0\chb-mit-scalp-eeg-database-1.0.0';

% Check if folder location exist. If not, warn user
if ~isfolder(myFolder)
    errorMessage = sprintf(['Error: The following folder does not exist:\n%s\nPlease specify a new folder.', myFolder]);
    uiwait(warndlg(errorMessage));
    myFolder = uigetdir();
    
    % Check if the folder returned false
    if myFolder == 0
        return;
    end
end

% Get a list of all desired file names within the folder
filePattern = fullfile(myFolder, '**/*.edf'); % '**/*.edf' Checks for all files in the folder and subfolders
desiredFiles = dir(filePattern);

% Retrieve all desired files through a for loop
for i = 1 : length(desiredFiles)
    % Read Files
    baseFileName = desiredFiles(i).name;
    fullFileName = fullfile(desiredFiles(i).folder, baseFileName);
    fprintf(1, 'Now reading %s\n', fullFileName);
    
    % Save File Data
    
    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create a new file
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% filename = '';
% masterEDF = edfwrite(filename,'SignalLabels.');











