close all
clear all
clc

%% Some Variables

calibrationFolder = 'Calibration';
measure1Folder = 'Measure 1';
measure2Folder = 'Measure 2';
measure3Folder = 'Measure 3';

mainOutputFolder = 'dataset';

cropMode = 0; % 0: Do not delete "noise" area. 1: Delete it

if cropMode == 0
    rawOutputFolder = strcat(mainOutputFolder, '/', 'rawNoCrop');
else
    rawOutputFolder = strcat(mainOutputFolder, '/', 'raw');
end

%% Some checks before starting

if ~exist(calibrationFolder, 'dir')
    error('Error: %s folder not found.', calibrationFolder);
end

if ~exist(measure1Folder, 'dir')
    error('Error: %s folder not found.', measure1Folder);
end

if ~exist(measure2Folder, 'dir')
    error('Error: %s folder not found.', measure2Folder);
end

if ~exist(measure3Folder, 'dir')
    error('Error: %s folder not found.', measure3Folder);
end

if ~exist('cropSignal.m', 'file')
    error('cropSginal function is missing from the current folder.');
end

if ~exist(mainOutputFolder, 'dir')
    mkdir(mainOutputFolder);
end

if ~exist(rawOutputFolder, 'dir')
    mkdir(rawOutputFolder);
end

%% Grouping all measuring files

calibrationFiles = dir(strcat(calibrationFolder, '/*.mat'))';
measure1Files = dir(strcat(measure1Folder, '/*.mat'))';
measure2Files = dir(strcat(measure2Folder, '/*.mat'))';
measure3Files = dir(strcat(measure3Folder, '/*.mat'))';

files = [calibrationFiles measure1Files measure2Files measure3Files];
folders = [{repmat({calibrationFolder}, 1, size(calibrationFiles, 2))}
    {repmat({measure1Folder}, 1, size(measure1Files, 2))}
    {repmat({measure2Folder}, 1, size(measure2Files, 2))}
    {repmat({measure3Folder}, 1, size(measure3Files, 2))}];
folders = [folders{:}];

%% Reading labels

labelFileID = fopen('LabelsFiltered.csv');
labelFile = textscan(labelFileID, '%s %s %s', 'Delimiter', ',');
fclose(labelFileID);
labelFileNames = strcat(labelFile{2}, '/', labelFile{3});

%% Processing files

% Some useful statitics
problematicFiles = cell(1, 1);

status1Files = cell(1, 1);
status2Files = cell(1, 1);
statusCount = 0;

label0Count = 0;
label1Count = 0;
label0DCount = 0;       % Counting each column as a sample
label1DCount = 0;       % Counting each column as a sample
filesWithoutLabel = cell(1, 1);

for i = 1:size(files, 2)
    currentFile = strcat(folders{i}, '/', files(i).name);
    fprintf('\nProcessing file %s...', currentFile)
    load(currentFile);
    pInfo.cropLimits = zeros(1, 2);
    pInfo.cropStatus = zeros(1, 2);
    try
        if strcmp(files(i).name, 'IED4HH.mat') % This specific file has only two channels
            [signal.ch3, pInfo.cropLimits(1), pInfo.cropStatus(1)] = cropSignal(subData(:, :, 1), cropMode);
            [signal.ch4, pInfo.cropLimits(2), pInfo.cropStatus(2)] = cropSignal(subData(:, :, 2), cropMode);
        else
            [signal.ch3, pInfo.cropLimits(1), pInfo.cropStatus(1)] = cropSignal(subData(:, :, 3), cropMode);
            [signal.ch4, pInfo.cropLimits(2), pInfo.cropStatus(2)] = cropSignal(subData(:, :, 4), cropMode);
        end
        currentLabel = labelFile{1}(strcmp(currentFile, labelFileNames));
        if size(currentLabel, 1) == 0
            % No label found
            filesWithoutLabel{end+1} = currentFile;
        else
            switch currentLabel{1}
                case 'No'
                    pInfo.label = 0;
                    label0Count = label0Count + 1;
                    label0DCount = label0DCount + size(signal.ch3, 2) + size(signal.ch4, 2);
                case 'Yes'
                    pInfo.label = 1;
                    label1Count = label1Count + 1;
                    label1DCount = label1DCount + size(signal.ch3, 2) + size(signal.ch4, 2);
            end
        end
        if any(pInfo.cropStatus == 1)
            statusCount = statusCount + 1;
            status1Files{end+1} = currentFile;
        end
        if any(pInfo.cropStatus == 2)
            statusCount = statusCount + 1;
            status2Files{end+1} = currentFile;
        end
        switch folders{i}
            case 'Calibration'
                outputFilePath = strcat(rawOutputFolder, '/c_', files(i).name);                
            case 'Measure 1'
                outputFilePath = strcat(rawOutputFolder, '/m1_', files(i).name);
            case 'Measure 2'
                outputFilePath = strcat(rawOutputFolder, '/m2_', files(i).name);
            case 'Measure 3'
                outputFilePath = strcat(rawOutputFolder, '/m3_', files(i).name);
        end
        fprintf(' Output: %s', outputFilePath)
        save(outputFilePath, 'pInfo', 'signal');
    catch ME
        problematicFiles{end+1} = currentFile;
        fprintf(' Error while cropping.')
    end
end

%% Reporting stats

fprintf('\nNo. of .mat files labeled as %s: %i', '"No Mine"', label0Count)
fprintf('\nNo. of .mat files labeled as %s: %i', '"Mine"', label1Count)
fprintf('\nTotal of labeled files: %i', label0Count + label1Count)
fprintf('\nNo. of samples labeled as %s: %i', '"No Mine"', label0DCount)
fprintf('\nNo. of samples labeled as %s: %i', '"Mine"', label1DCount)
fprintf('\nTotal of labeled samples: %i', label0DCount + label1DCount)
fprintf('\nFiles with no label:\n')
disp(filesWithoutLabel)

fprintf('\nFiles which raised unknown errors (e.g. empty file):\n')
disp(problematicFiles)

fprintf('\nMeasurements with crop status 1:\n')
disp(status1Files)
fprintf('\nMeasurements with crop status 2:\n')
disp(status2Files)