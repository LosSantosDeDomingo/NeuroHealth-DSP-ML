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
% Collect EEG Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
documentInfo = edfinfo('chb01_01.edf');
documentData = edfread("chb01_01.edf");
fp1_f7_Data = edfread('chb01_01.edf', 'SelectedSignals', 'FP1-F7');
fp1_f7_cellArr = table2array(fp1_f7_Data);
fp1_f7_Arr = vertcat(fp1_f7_cellArr{:});
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
arraySize = length(fp1_f7_Arr);
fullFile = floor(arraySize) - 1; % Used to complete a full window, provides an integer
numberofWindows = floor(arraySize / samples);

% Create a matrix
matrixSize = samples * numberofWindows;
FP1_F7 = zeros(samples,numberofWindows);

% Loop though array
for i = 1 : numberofWindows

    % Interval Information
    startIndex = (i-1) * samples + 1;
    endIndex = i * samples;
    
    % Insert data into the array
    FP1_F7(:,i) = fp1_f7_Arr(startIndex:endIndex);

end

% Plot Test
for i=1:5
    figure
    plot(t, FP1_F7(:, i)); % Only plot the current window  
    title(sprintf('Window %d', i));
    xlabel('Samples (0–255)');
    ylabel('EEG Amplitude (\muV)'); % Assuming units are microvolts
    pause(0.1); % Optional: slow down loop so you can see the plot  
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
