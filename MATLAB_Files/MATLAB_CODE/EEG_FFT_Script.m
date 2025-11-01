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
% (2) https://www.ncbi.nlm.nih.gov/books/NBK539805/
% (3) https://www.mathworks.com/help/
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Purpose of Program
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The purpose of this program is to breakdown the recorded EEG data
% by employing the Fast Fourier Transform (FFT). The Data collected
% was previously processed in the EEG_Preprocessing program. All
% resulting data gathered from the program will be used to be broken
% down prior to being labeled and sent to SVM_Model program for
% training.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Improvement Status
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Immediate Improvements for Current Version:
% --------------------------------------------
% (1) Include graphs to show the signal data before and after
% (2) Save graphs to an external hardware
% 
% Possible Improvements for Later Version:
% -----------------------------------------
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Version Info
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Version: 1
% Data Created: 05/12/2025
% Last Revision: 06/27/2025
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Clear Workspace, Command Window, and Figures 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; % Clear Workspace memory
clc; % Clear Command Window
close all; % Close all figures
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
signalFolders = {'D:\ProcessedEEG', 'D:\ProcessedSeizureEEG'};
numberOfFolders = length(signalFolders);

% Validate Folders Exist
for inputFolder = 1:numberOfFolders
    currentFolder = signalFolders{inputFolder};
    if ~isfolder(currentFolder)
        errorMessage = sprintf('Error: The following folder does not exist:\n%s\nPlease specify a new folder.', currentFolder);
        uiwait(warndlg(errorMessage));
        selectedFolder = uigetdir();
        if selectedFolder == 0
            return;
        end
        signalFolders{inputFolder} = selectedFolder;
    end
end

% Desired Subfolders and Files (Base Date)
filePattern = fullfile(signalFolders{1}, '**/*.mat');
desiredFiles = dir(filePattern);
folderSize = length(desiredFiles);

% Desired Subfolders and Files (Seizure Date)
filePatternSeizure = fullfile(signalFolders{2}, '**/*.mat');
desiredFilesSeizure = dir(filePatternSeizure);
folderSizeSeizure = length(desiredFilesSeizure);

% Display Files
fprintf('Running EEG FFT Processing Script...\n\n');
fprintf('Locating Input Files...\n');
fprintf('Total number of .mat files located within the baseline folder: %d\n', folderSize);
fprintf('Total number of .mat files located within the seizure folder: %d\n\n', folderSizeSeizure);

% Output Folders
outputFolders = {'D:\processedSignalsFFT\', 'D:\processedSeizureSignalFFT\'};
numberOfOutputFolders = length(outputFolders);

fprintf('Locating Output Folders...\n');
for outputSeizureFolder = 1:numberOfOutputFolders
    currentFolder = outputFolders{outputSeizureFolder};
    if ~isfolder(currentFolder)
        mkdir(currentFolder);
    end
    fprintf('The %s folder was found...\n', currentFolder);
end

fprintf('\nProcessing All Baseline EEG Signals...\n');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stage 2: Folder Looping (Base Brain Signals)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loop Through Folder
for file = 1:folderSize % Playing around with the numbers for testing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Stage 2.1: File Retrieval
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    % Retrieve all desired files and folders
    fileNames = desiredFiles(file).name;
    folderNames = desiredFiles(file).folder;
    fullFileName = fullfile(folderNames, fileNames);
   
    % Validate File
    if ~isfile(fullFileName)
    errorMessage = sprintf('Error: The following folder does not exist:\n%s\nPlease specify a new file.', fullFileName);
    uiwait(warndlg(errorMessage));
    fullFileName = uigetdir();
        if fullFileName == 0
            return;
        end
    end

    % Display File Name and File Count
    fprintf('(%d / %d) ', file, folderSize);
    fprintf(1, 'Now processing %s\n', fileNames); 
    
    % Load File
    load(fullFileName); % Variable Name: windowedEEG (Channels X Samples X Windows)
    numberOfChannels = size(windowedEEG, 1);
    numberOfSamples = size(windowedEEG, 2); % Signal length after windowing
    numberOfWindows = size(windowedEEG, 3);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Stage 2.2: Filter Design
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     
    % Calculate Signal Details 
    samplingFrequency = 256; % Should be the same as the number of samples
    samplingPeriod = 1 / samplingFrequency;
    timeInterval = 0:samplingPeriod:1;
    frequency = (0:samplingFrequency - 1) * (samplingFrequency / numberOfSamples);

    % Bandpass Frequencies
    startFrequency = 0.5;
    stopFrequency = 70;

    % Normalize Frequencies by the Nyquist Rate
    nyquistFrequency = samplingFrequency / 2;
    normalizedStartFrequency = startFrequency / nyquistFrequency;
    normalizedStopFrequency = stopFrequency / nyquistFrequency;

    % Create Filter
    bandpassRange = [normalizedStartFrequency normalizedStopFrequency];
    filterOrder = 50;
    filterCoefficients = fir1(filterOrder,bandpassRange,"bandpass");
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Stage 2.3: Perform FFT
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
    % Perallocate Matrix
    numberOfBands = 5;
    totalRowVectors = numberOfBands * numberOfChannels;
    windowedEEG_relativePower = zeros(numberOfWindows, totalRowVectors);
    
    for window = 1:numberOfWindows
        for channel = 1:numberOfChannels
            % Apply Filter
            currentWindow = windowedEEG(channel, :, window)';
            appliedFilterCoefficients = fftfilt(filterCoefficients, currentWindow);

            % Run FFT
            fftCoefficients = fft(appliedFilterCoefficients);
            
            % Normalize Coefficients
            normalizedFFTCoefficients = samplingPeriod * fftCoefficients;

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Stage 2.4: Locating EEG Frequency Bands
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
            % Alpha waves (8-12 Hz): Bins 10 - 13
            % Beta waves (12-30 Hz): Bins 14 - 31
            % Theta waves (4-8 Hz): Bins 6 - 9
            % Delta waves (0.5-4 Hz): Bins 2 - 5
            % Gamma waves (30+ Hz): Bins 32 - 71
            % Desired Frequency Range 1 Hz - 70 Hz
            
            % Search Bins
            frequencyResolution = samplingFrequency / numberOfSamples;
            frequencyBin = (0:numberOfSamples - 1) * frequencyResolution;
            
            % Band Indices
            deltaIndex = find(frequencyBin >= 2 & frequencyBin <= 5);
            if isempty(deltaIndex)
                warning('No FFT bins found for Delta band in this configuration.');
            end

            thetaIndex = find(frequencyBin >= 6 & frequencyBin <= 9);
            if isempty(thetaIndex)
                warning('No FFT bins found for Theta band in this configuration.');
            end

            alphaIndex = find(frequencyBin >= 10 & frequencyBin <= 13);
            if isempty(alphaIndex)
                warning('No FFT bins found for Alpha band in this configuration.');
            end

            betaIndex = find(frequencyBin >= 14 & frequencyBin <= 31);
            if isempty(betaIndex)
                warning('No FFT bins found for Beta band in this configuration.');
            end

            gammaIndex = find(frequencyBin >= 32 & frequencyBin <= 71);
            if isempty(gammaIndex)
                warning('No FFT bins found for Gamma band in this configuration.');
            end

            % Associate Bins to Bands
            deltaBand = normalizedFFTCoefficients(deltaIndex);
            thetaBand = normalizedFFTCoefficients(thetaIndex);
            alphaBand = normalizedFFTCoefficients(alphaIndex);
            betaBand = normalizedFFTCoefficients(betaIndex);
            gammaBand = normalizedFFTCoefficients(gammaIndex);

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Stage 2.5: Power Calculations
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
            % Calculate Band Power
            avgPowerDeltaBand = mean(abs(deltaBand).^2);
            avgPowerThetaBand = mean(abs(thetaBand).^2);
            avgPowerAlphaBand = mean(abs(alphaBand).^2);
            avgPowerBetaBand = mean(abs(betaBand).^2);
            avgPowerGammaBand = mean(abs(gammaBand).^2);
            
            % Total Power
            totalPower = avgPowerDeltaBand + avgPowerThetaBand + avgPowerAlphaBand + avgPowerBetaBand + avgPowerGammaBand;

            % Relative Band Power
            if totalPower == 0
                relativePowerDelta = 0; 
                relativePowerTheta = 0;
                relativePowerAlpha = 0; 
                relativePowerBeta = 0;
                relativePowerGamma = 0;
            else
                relativePowerDelta = avgPowerDeltaBand / totalPower;
                relativePowerTheta = avgPowerThetaBand / totalPower;
                relativePowerAlpha = avgPowerAlphaBand / totalPower;
                relativePowerBeta = avgPowerBetaBand / totalPower;
                relativePowerGamma = avgPowerGammaBand / totalPower;
            end

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Stage 2.6: Machine Learning Preparations
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
            % Combine Relative Power
            relativePowerVector = [relativePowerDelta, relativePowerTheta, relativePowerAlpha,...
                                   relativePowerBeta, relativePowerGamma];
            
            % Prepare Matrix (window x (Bands x Channel))
            currentRowVectors = numberOfBands * channel;
            
            % Store Power Vector in Matrix
            startColumn = (channel - 1) * numberOfBands + 1;
            endColumn = currentRowVectors;
            columnInterval = startColumn:endColumn;
            windowedEEG_relativePower(window,columnInterval) = relativePowerVector;
        end
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Stage 2.7: Table Making
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    % Column Naming
    bandList = {'_Delta_RP', '_Theta_RP', '_Alpha_RP','_Beta_RP','_Gamma_RP'};
    variableNames = string(zeros(1, numberOfBands));
    removal = '.mat';
    channelCounter = 1;
    channelIndex = mod(channelCounter - 1, numberOfChannels) + 1;

    for tableColumn = 1:totalRowVectors
        bandIndex = mod(tableColumn, numberOfBands);
        if bandIndex ~= 0
            replacement = "_" + desiredNodes(channelIndex) + bandList(bandIndex);
            variableNames(tableColumn) = replace(fileNames, removal, replacement);
        else
            bandIndex = 5;
            replacement = "_" + desiredNodes(channelIndex) + bandList(bandIndex);
            variableNames(tableColumn) = replace(fileNames, removal, replacement);
            channelCounter = channelCounter + 1;
            channelIndex = mod(channelCounter, numberOfChannels);

            if channelIndex == 0
                channelIndex = 22;
                variableNames(tableColumn) = replace(fileNames, removal, replacement);
            end
        end
    end

    % Row Naming
    rowNames = "Window_" + (1:numberOfWindows)';

    % Table Naming
    windowedEEG_RP_Table = array2table(windowedEEG_relativePower, 'RowNames', rowNames, 'VariableNames', variableNames);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Stage 2.8 File Saving
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
    % Save Results as a .mat File
    adjustment = "_FFT.mat";
    fixedFileName = replace(fileNames, removal, adjustment);
    save([outputFolders{1} fixedFileName], "windowedEEG_RP_Table");
end

fprintf('\nAll baseline EEG data has been processed, moving on to seizure data...\n'); 
fprintf('\nProcessing All Seizure EEG Signals...\n');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stage 3: Folder Looping (Seizure Brain Signals)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loop Through Folder
for file = 1:folderSizeSeizure
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Stage 3.1: File Retrieval
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    % Retrieve all desired files and folders
    seizureFileNames = desiredFilesSeizure(file).name;
    folderNames = desiredFilesSeizure(file).folder;
    fullFileName = fullfile(folderNames, seizureFileNames);
    
    % Validate File
    if ~isfile(fullFileName)
    errorMessage = sprintf('Error: The following folder does not exist:\n%s\nPlease specify a new file.', fullFileName);
    uiwait(warndlg(errorMessage));
    fullFileName = uigetdir();
        if fullFileName == 0
            return;
        end
    end

    % Display File Name and File Count
    fprintf('(%d / %d) ', file, folderSizeSeizure);
    fprintf(1, 'Now processing %s\n', seizureFileNames); 
    
    % Load File
    load(fullFileName); % Variable Name: windowedEEG (Channels X Samples X Windows)
    numberOfChannels = size(windowedEEG, 1);
    numberOfSamples = size(windowedEEG, 2); % Signal length after windowing
    numberOfWindows = size(windowedEEG, 3);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Stage 3.2: Filter Design
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     
    % Calculate Signal Details 
    samplingFrequency = 256; % Should be the same as the number of samples
    samplingPeriod = 1 / samplingFrequency;
    timeInterval = 0:samplingPeriod:1;
    frequency = (0:samplingFrequency - 1) * (samplingFrequency / numberOfSamples);

    % Bandpass Frequencies
    startFrequency = 0.5;
    stopFrequency = 70;

    % Normalize Frequencies by the Nyquist Rate
    nyquistFrequency = samplingFrequency / 2;
    normalizedStartFrequency = startFrequency / nyquistFrequency;
    normalizedStopFrequency = stopFrequency / nyquistFrequency;

    % Create Filter
    bandpassRange = [normalizedStartFrequency normalizedStopFrequency];
    filterOrder = 50;
    filterCoefficients = fir1(filterOrder,bandpassRange,"bandpass");
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Stage 3.3: Perform FFT
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
    % Perallocate Matrix
    numberOfBands = 5;
    totalRowVectors = numberOfBands * numberOfChannels;
    windowedEEG_relativePower = zeros(numberOfWindows, totalRowVectors);
    
    for window = 1:numberOfWindows
        for channel = 1:numberOfChannels
            % Apply Filter
            currentWindow = windowedEEG(channel, :, window)';
            appliedFilterCoefficients = fftfilt(filterCoefficients, currentWindow);

            % Run FFT
            fftCoefficients = fft(appliedFilterCoefficients);
            
            % Normalize Coefficients
            normalizedFFTCoefficients = samplingPeriod * fftCoefficients;

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Stage 3.4: Locating EEG Frequency Bands
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
            % Alpha waves (8-12 Hz): Bins 10 - 13
            % Beta waves (12-30 Hz): Bins 14 - 31
            % Theta waves (4-8 Hz): Bins 6 - 9
            % Delta waves (0.5-4 Hz): Bins 2 - 5
            % Gamma waves (30+ Hz): Bins 32 - 71
            % Desired Frequency Range 1 Hz - 70 Hz

            % Search Bins
            frequencyResolution = samplingFrequency / numberOfSamples;
            frequencyBin = (0:numberOfSamples - 1) * frequencyResolution;
            
            % Band Indices
            deltaIndex = find(frequencyBin >= 2 & frequencyBin <= 5);
            if isempty(deltaIndex)
                warning('No FFT bins found for Delta band in this configuration.');
            end

            thetaIndex = find(frequencyBin >= 6 & frequencyBin <= 9);
            if isempty(thetaIndex)
                warning('No FFT bins found for Theta band in this configuration.');
            end

            alphaIndex = find(frequencyBin >= 10 & frequencyBin <= 13);
            if isempty(alphaIndex)
                warning('No FFT bins found for Alpha band in this configuration.');
            end

            betaIndex = find(frequencyBin >= 14 & frequencyBin <= 31);
            if isempty(betaIndex)
                warning('No FFT bins found for Beta band in this configuration.');
            end

            gammaIndex = find(frequencyBin >= 32 & frequencyBin <= 71);
            if isempty(gammaIndex)
                warning('No FFT bins found for Gamma band in this configuration.');
            end

            % Associate Bins to Bands
            deltaBand = normalizedFFTCoefficients(deltaIndex);
            thetaBand = normalizedFFTCoefficients(thetaIndex);
            alphaBand = normalizedFFTCoefficients(alphaIndex);
            betaBand = normalizedFFTCoefficients(betaIndex);
            gammaBand = normalizedFFTCoefficients(gammaIndex);

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Stage 3.5: Power Calculations
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
            % Calculate Band Power
            avgPowerDeltaBand = mean(abs(deltaBand).^2);
            avgPowerThetaBand = mean(abs(thetaBand).^2);
            avgPowerAlphaBand = mean(abs(alphaBand).^2);
            avgPowerBetaBand = mean(abs(betaBand).^2);
            avgPowerGammaBand = mean(abs(gammaBand).^2);
            
            % Total Power
            totalPower = avgPowerDeltaBand + avgPowerThetaBand + avgPowerAlphaBand + avgPowerBetaBand + avgPowerGammaBand;

            % Relative Band Power
            if totalPower == 0
                relativePowerDelta = 0; relativePowerTheta = 0;
                relativePowerAlpha = 0; relativePowerBeta = 0;
                relativePowerGamma = 0;
            else
                relativePowerDelta = avgPowerDeltaBand / totalPower;
                relativePowerTheta = avgPowerThetaBand / totalPower;
                relativePowerAlpha = avgPowerAlphaBand / totalPower;
                relativePowerBeta = avgPowerBetaBand / totalPower;
                relativePowerGamma = avgPowerGammaBand / totalPower;
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Stage 3.6: Machine Learning Preparations
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
            % Combine Relative Power
            relativePowerVector = [relativePowerDelta, relativePowerTheta, relativePowerAlpha,...
                                   relativePowerBeta, relativePowerGamma];
            
            % Prepare Matrix (window x (Bands x Channel))
            % numberOfBands = width(relativePowerVector);
            % totalRowVectors = numberOfBands * numberOfChannels;
            currentRowVectors = numberOfBands * channel;
            % windowedEEG_relativePower = zeros(numberOfWindows, totalRowVectors);
            
            % Store Power Vector in Matrix
            startColumn = (channel - 1) * numberOfBands + 1;
            endColumn = currentRowVectors;
            columnInterval = startColumn:endColumn;
            windowedEEG_relativePower(window,columnInterval) = relativePowerVector;
        end
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Stage 3.7: Table Making
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    % Column Naming
    bandList = {'_Delta_RP', '_Theta_RP', '_Alpha_RP','_Beta_RP','_Gamma_RP'};
    variableNames = string(zeros(1, numberOfBands));
    removal = '.mat';
    channelCounter = 1;
    channelIndex = mod(channelCounter - 1, numberOfChannels) + 1;

    for tableColumn = 1:totalRowVectors
        bandIndex = mod(tableColumn, numberOfBands);
        if bandIndex ~= 0
            replacement = "_" + desiredNodes(channelIndex) + bandList(bandIndex);
            variableNames(tableColumn) = replace(seizureFileNames, removal, replacement);
        else
            bandIndex = 5;
            replacement = "_" + desiredNodes(channelIndex) + bandList(bandIndex);
            variableNames(tableColumn) = replace(seizureFileNames, removal, replacement);
            channelCounter = channelCounter + 1;
            channelIndex = mod(channelCounter, numberOfChannels);

            if channelIndex == 0
                channelIndex = 22;
                variableNames(tableColumn) = replace(seizureFileNames, removal, replacement);
            end
        end
    end

    % Row Naming
    rowNames = "Window_" + (1:numberOfWindows)';

    % Table Naming
    windowedSeizureEEG_RP_Table = array2table(windowedEEG_relativePower, 'RowNames', rowNames, 'VariableNames', variableNames);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Stage 3.8 File Saving
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
    % Save Results as a .mat File
    adjustment = "_FFT_Seizure.mat";
    fixedSeizureFileName = replace(seizureFileNames, removal, adjustment);
    save([outputFolders{2} fixedSeizureFileName], "windowedSeizureEEG_RP_Table");
end
fprintf('\n\nAll seizure EEG data has been processed...\n');

% Display When FFT processing Has Been Completed
fprintf('All files have been processed...');
