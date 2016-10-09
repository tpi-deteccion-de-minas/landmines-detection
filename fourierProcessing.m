close all
clear all
clc

%% Some paths

mainOutputFolder = 'dataset';
rawOutputFolder = strcat(mainOutputFolder, '/', 'raw');

%% Some checks

if ~exist(mainOutputFolder, 'dir')
    error('Error: %s folder not found.', rawOutputFolder);
end

%% Gathering the data

files = dir(strcat(rawOutputFolder, '/*.mat'));
folder = {repmat({rawOutputFolder}, 1, size(files, 1))};
folder = [folder{:}];

%% Some variables

label0DCount = 0;       % Counting each column as a sample
label1DCount = 0;       % Counting each column as a sample
idx = 1;                % Samples so far

% (i,0): id, (i,1): class, (i,2): features
generatedDataset = cell(184147, 3);

for i = 1:size(files, 1)
    currentFile = strcat(folder{i}, '/', files(i).name);
    fprintf('\nProcessing file %s...', currentFile)
    load(currentFile);
    % 0: no mine, 1: mine
    if pInfo.label == 0
        label0DCount = label0DCount + size(signal.ch3, 2) + size(signal.ch4, 2);
    else
        label1DCount = label1DCount + size(signal.ch3, 2) + size(signal.ch4, 2);
    end
    
    % From here to the end of the loop, each file can be processed in order
    % to generate the final dataset. Each file has a struct object named
    % 'signal' which has two measurements, 'ch3' and 'ch4', corresponding
    % to each of the antennas that we were told to take into account.
    %
    % Each measurement is a matrix so, to extract a single sample or A-Scan
    % we use the following instruction:
    %
    %       one_sample = signal.ch3(:, i);
    %                   or
    %       one_sample = signal.ch4(:, i);
    %
    % depending on the channel we want to work on and the i-th sample we
    % want.
    %
    % The signal processing starts here (e.g. Fourier, Wigner-Ville, etc.)
    %
    % Some things to remember:
    %   - The resulting dataset will be saved into a .csv (comma-separated
    %     file).
    %   - The processing involving the Wigner-Ville transform should give
    %     a fixed size image for each sample in order to feed the convolu-
    %     tional neural network.
    
    sample = zeros(1, 25);
    signals = {signal.ch3 signal.ch4};
    for j = 1:2
        currX = 1:size(signals{j}, 1);
        for k = 1:size(signals{j}, 2)
            % Statistical moments
            currAScan = signals{j}(:, k);
            sample(1) = mean(currAScan);
            sample(2) = std(currAScan);
            sample(3) = skewness(currAScan);
            sample(4) = kurtosis(currAScan);
            sample(5) = max(currAScan);
            sample(6) = min(currAScan);
            
            % Fourier coefficients
            fourierFit = fit(currX', currAScan, 'fourier8');
            sample(7:24) = coeffvalues(fourierFit);
            sample(25) = (2*pi)/fourierFit.w; % Frequency of the obtained fourier signal
            
            % TODO: add more features.
            
            % Adding the sample to the generated dataset
            generatedDataset{idx, 1} = files(i).name;
            generatedDataset{idx, 2} = pInfo.label;
            generatedDataset{idx, 3} = sample;
            idx = idx + 1;
        end
    end
end

% Saving the dataset
save('dataset/fourier/fourier.mat', 'generatedDataset');

% Some numbers about the dataset
fprintf('\n\nNo. of samples labeled as %s: %i', '"No Mine"', label0DCount)
fprintf('\nNo. of samples labeled as %s: %i', '"Mine"', label1DCount)
fprintf('\nTotal of labeled samples: %i\n', label0DCount + label1DCount)