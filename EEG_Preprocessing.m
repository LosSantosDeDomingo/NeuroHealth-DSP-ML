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
% Last Revision: 04/30/2025
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Clear Workspace, Command Window, and Figures 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; % Clear Workspace memory
clc; % Clear Command Window
close all; % Close all figures
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Check for Available Signals
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Node String Array
nodeList = {'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2','FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 'FZ-CZ', 'CZ-PZ','P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8', 'T8-P8'};

% Gather Document Details
documentInfo = edfinfo('chb01_01.edf');
availableSignals = documentInfo.SignalLabels;

% Check which requested channels are available
isAvailable = ismember(nodeList, availableSignals);

% Display Missing Channels
if ~all(isAvailable)
    missing = nodeList(~isAvailable);
    fprintf('Missing channels in %s:\n', 'chb01_01.edf');
    disp(missing);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Collect EEG Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
multiChannelMatrix = edfread('chb01_01.edf', 'SelectedSignals', nodeList);
multiChannelCellArr = table2array(multiChannelMatrix);
signalMatrix = cellfun(@vertcat, multiChannelCellArr, 'UniformOutput', false);
signalMatrix = cell2mat(signalMatrix');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Process EEG Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Basic Interval Information
samples = 256;
sampleRate = 1/samples;
beginRange = 1;
endRange = 257;
t = (0:255);

% Node File Size
numberOfWindows = floor(size(signalMatrix, 1) / samples);
numberOfChannels = size(signalMatrix, 2);

% Create a matrix
windowedEEG = zeros(samples,numberOfWindows,numberOfChannels);

% Loop though array
for i = 1:numberOfChannels
    for j = 1:numberOfWindows

        % Interval Information
        startIndex = (j-1) * samples + 1;
        endIndex = j * samples;
    
        % Insert data into the array
        windowedEEG(:,j,i) = signalMatrix(startIndex:endIndex, i);
    end
end

% Plot Test
channelIndex = 1;
windowIndex = 10;
plot(windowedEEG(:, windowIndex, channelIndex));
xlabel('Sample (1–256)');
ylabel('EEG Amplitude');
title(sprintf('Channel %d, Window %d', channelIndex, windowIndex));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
