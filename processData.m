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

allFolders = [trainFolders testFolders];
allFiles = [trainFiles; testFiles];

%% Dataset generation
generateDataset(allFolders, allFiles, datasetChoice);