close all
clear all
clc

%% Some variables

% 0: Fourier features
% 1: Spectrogram
% 2: Spectrogram images (224,224)
% 3: Wavelets

datasetChoice = 3;

%% Some paths

mainFolder = 'dataset';
trainFolder = strcat(mainFolder, '/rawNoCrop/train');
testFolder = strcat(mainFolder, '/rawNoCrop/test');

% Path to save the dataset
switch datasetChoice
    case 0
        datasetName = 'csv_fourier';
    case 1
        datasetName = 'csv_spectrogram';
    case 2
        datasetName = 'img_spectrogram';
    case 3
        datasetName = 'h5_wavelets';
end

datasetPath = strcat('dataset/rawNoCrop/', datasetName);

%% Gathering the data

trainFiles = dir(strcat(trainFolder, '/*.mat'));
trainFolders = {repmat({trainFolder}, 1, size(trainFiles, 1))};
trainFolders = [trainFolders{:}];

testFiles = dir(strcat(testFolder, '/*.mat'));
testFolders = {repmat({testFolder}, 1, size(testFiles, 1))};
testFolders = [testFolders{:}];

%% Dataset generation
[X_train, Y_train, trainMean, trainStd] = generateDataset(trainFolders, trainFiles, datasetChoice, 1);
[X_test, Y_test] = generateDataset(testFolders, testFiles, datasetChoice, 0, trainMean, trainStd);

% This section is only used for the Fourier features dataset as it is small
% enough to keep in memory.
if datasetChoice == 0
    %% Saving
    dlmwrite(strcat(datasetPath, '/train.txt'),       X_train, 'delimiter', ' ', 'precision', 8);
    dlmwrite(strcat(datasetPath, '/trainLabels.txt'), Y_train, 'delimiter', ' ', 'precision', 8);
    dlmwrite(strcat(datasetPath, '/test.txt'),        X_test,  'delimiter', ' ', 'precision', 8);
    dlmwrite(strcat(datasetPath, '/testLabels.txt'),  Y_test,  'delimiter', ' ', 'precision', 8);
end