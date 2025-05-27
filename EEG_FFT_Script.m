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
% Detection on FPGA" by Jiangwei He and Co.
% (2) https://www.ncbi.nlm.nih.gov/books/NBK539805/
%
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
% (1) Remove signal reconstruction (Not needed I lost track trying to teach myself)
% (2) Fix dominant frequency portion to check for band frequencies
% (3) Check the total power of all bands
% (4) Prepare final matrix and labels
% (5) Save to external hard drive files
% (6) Repeat but for seizure data
% 
% Possible Improvements for Later Version:
% -----------------------------------------
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Version Info
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Version: 1
% Data Created: 05/12/2025
% Last Revision: 05/27/2025
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Clear Workspace, Command Window, and Figures 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; % Clear Workspace memory
clc; % Clear Command Window
close all; % Close all figures
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stage 1: Locate Signal Folders (.mat files)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
folderSize = 1; %length(desiredFiles);

% Desired Subfolders and Files (Seizure Date)
filePatternSeizure = fullfile(signalFolders{2}, '**/*.mat');
desiredFilesSeizure = dir(filePatternSeizure);
folderSizeSeizure = 1; %length(desiredFiles);

% Display Files
fprintf('Running EEG FFT Processing Script...\n\n');
fprintf('Locating Input Files...\n');
fprintf('Total number of .mat files located within the baseline folder: %d\n', folderSize);
fprintf('Total number of .mat files located within the seizure folder: %d\n\n', folderSizeSeizure);

% Output Folders
outputFolders = {'D:\processedSignalsFFT', 'D:\ProcessedSeizureSignalFFT'};
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
for file = 1:1 %folderSize
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
    % Alpha waves (8-12 Hz):
    % Beta waves (12-30 Hz):
    % Theta waves (4-8 Hz):
    % Delta waves (0.5-4 Hz):
    % Gamma waves (30+ Hz):
    % Desired Frequency Range 0.5 Hz - 70 Hz

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
    for window = 1:1 %numberOfWindows
        for channel = 1:1 %numberOfChannels
            % Apply Filter
            currentWindow = windowedEEG(channel, :, window)';
            appliedFilterCoefficients = fftfilt(filterCoefficients, currentWindow);

            % Run FFT
            fftCoefficients = fft(appliedFilterCoefficients);
            
            % Normalize Coefficients
            normalizedFFTCoefficients = samplingPeriod * fftCoefficients;

            % Amplitude and Phase Spectrum
            amplitudeSpectrum = abs(normalizedFFTCoefficients);
            phaseAngleSpectrum = angle(normalizedFFTCoefficients);

            % Graphing (Made for Testing)
            lineWidth = 1;
            figure(1)
            subplot(2,2,1)
            stem(frequency', amplitudeSpectrum, '*', 'LineWidth', lineWidth), xlabel('Frequency (Hz)'), ylabel('Amplitude'), grid on
            subplot(2,2,2)
            stem(frequency', phaseAngleSpectrum, '*', 'LineWidth', lineWidth), xlabel('Frequency (Hz)'), ylabel('Phase Angle'), grid on            

            % FFT Table (Made for Testing)
            signalTable = table(normalizedFFTCoefficients, frequency', amplitudeSpectrum, phaseAngleSpectrum);
            signalTable.Properties.VariableNames = {'FFT Coefficients', 'Frequency', 'Amplitude', 'Phase'};
            disp(signalTable);

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Stage 2.4: Signal Reconstruction
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
            % Remove Complex Conjugates and the Noise Floor
            nyquistRange = ceil(numberOfSamples / 3.5) + 1;
            desiredAmplitudes = amplitudeSpectrum(1:nyquistRange);
            desiredLocations = frequency(1:nyquistRange)';
            desiredPhase = phaseAngleSpectrum(1:nyquistRange);
            
            % Plot (For Testing)
            subplot(2,2,3)
            stem(desiredLocations, desiredAmplitudes, '*', 'LineWidth', lineWidth), xlabel('Frequency (Hz)'), ylabel('Amplitude'), grid on

            % Find Local Maxima
            relativeThreshold = 0.01;
            prominenceThreshold = relativeThreshold * max(desiredAmplitudes);
            [peaks, locations, widths, prominence] = findpeaks(desiredAmplitudes, 'MinPeakProminence', prominenceThreshold);
            
            % Set Dominant Frequencies
            numberOfPeaks = length(peaks);
            k = 15;

            if numberOfPeaks < k
                % Create Padding
                paddedAmplitudes = zeros(k, 1);
                paddedFrequencies = zeros(k, 1);
                paddedPhase = zeros(k, 1);

                % Pad the remaining space with zeros
                paddedFrequencies(1:numberOfPeaks) = desiredLocations(locations);
                paddedAmplitudes(1:numberOfPeaks) = peaks;
                paddedPhase(1:numberOfPeaks) = desiredPhase(locations);

                % Maintain Values
                dominantFrequencies = paddedFrequencies;
                dominantAmplitudes = paddedAmplitudes;
                dominantPhases = paddedPhase;
            else     
                % Create score
                score = peaks .* prominence;
                [~, sortedIndex] = sort(score, 'descend');
                topK = sortedIndex(1:k);
                
                % Truncate signal
                truncatedFrequencies = desiredLocations(locations(topK));
                truncatedPhase = desiredPhase(locations(topK));
                truncatedAmplitudes = peaks(topK);

                [~, sortIndex] = sort(truncatedFrequencies, 'ascend');
                
                % Maintain Values
                dominantFrequencies = truncatedFrequencies(sortIndex);
                dominantAmplitudes = truncatedAmplitudes(sortIndex);
                dominantPhases = truncatedPhase(sortIndex);
            end

            % Graphing (Made for Testing)
            subplot(2,2,4)
            stem(dominantFrequencies, dominantAmplitudes, '*', 'LineWidth', lineWidth), xlabel('Frequency (Hz)'), ylabel('Amplitude'), grid on
            
            % Reconstruct FFT Coefficients
            deconstructedCoefficients = zeros(k,1);

            for index = 1:k
                deconstructedCoefficients(index) = dominantAmplitudes(index) * exp(dominantPhases(index)*i());
            end
            
            % Rebuild Full Spectrum Coefficients
            fullSpectrumCoefficients = zeros(numberOfSamples, 1);
            
            for sample = 1:numberOfSamples
                for index = 1:k
                    if sample == dominantFrequencies(index)
                        fullSpectrumCoefficients(sample) = deconstructedCoefficients(index);
                    end
               end
            end
            
            % Include Conjugates
            reverseIndex = sort(frequency(:)+1,'descend');
            reverseIndex = reverseIndex(1:nyquistRange);
            
            chosenReverse = zeros(k, 1);
            for reverse = 1:length(reverseIndex)
                for index = 1:k
                    if reverse == dominantFrequencies(index)
                        chosenReverse(index) = reverseIndex(reverse);
                    end
                end
            end
            chosenReverse = sort(chosenReverse, 'ascend');

            for coefficient = 1:length(normalizedFFTCoefficients)
                for index = 1:k
                    if coefficient == chosenReverse(index)
                        fullSpectrumCoefficients(coefficient) = normalizedFFTCoefficients(coefficient);
                    end
                end
            end


            % Signal Reconstruction
            signalRecontruction = ifft(fullSpectrumCoefficients, 'symmetric');

            % Graphing (Made for Testing)
            figure(2);
            plot(frequency', currentWindow, 'k-*', 'LineWidth', 2)
            hold on, grid on
            xlabel('Frequency (Hz)'), ylabel('Amplitude')

            figure(3);
            plot(frequency', signalRecontruction, 'b--', 'LineWidth', 2)
            hold on, grid on
            xlabel('Frequency (Hz)'), ylabel('Amplitude')            
            
        end
    end
 

    %if folderNames == "D:\ProcessedEEG"
     %   seizureStatus = 0;
    %end 
end

