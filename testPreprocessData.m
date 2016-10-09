close all
clear all
clc

%% Control variables

plotMaxValues = 0;
plotCroppedSignal = 0;
omitReflections = 1;

%% Extracting a revolution

% load 'IED2HH';
% load 'IED2HV';
% load 'Calibration/CylinderVV';
% load 'Calibration/CornerVH.mat';
% load 'Calibration/CylinderHH.mat';
% load 'Measure 1/CAN1PosHPolH.mat';     % Significant reflections at the beginning
load 'Measure 2/IED17HV.mat';          % High energy at the beginning
% load 'Measure 3/IED4HH.mat';           % WARNING: this sample has only two channels
% load 'Measure 2/CANHH.mat';            % WARNING: badly taken measurements

meanData = mean(subData(:, :, 4), 2) * ones(1, size(subData, 2));
subData(:, :, 4) = subData(:, :, 4) - meanData;

figure; imagesc(subData(:, :, 4));
imagePack = subData(:, :, 4);
channel4Max = sum(subData(:, :, 4).^2, 2);
[~, indexMax] = max(channel4Max);
% figure; plot(channel4Max);

[autocor, lags] = xcorr(imagePack(indexMax, :));
autocor = autocor/abs(max(autocor));
figure; plot(lags, autocor);

% [peaksL, locationsL] = findpeaks(autocor);
% short = mean(diff(locationsL)) * 50;

[peaksS, locationsS] = findpeaks(autocor, 'MinPeakDistance', 500);
[~, maxPeakIndex] = max(peaksS);
peaksS(maxPeakIndex) = NaN;
[~, secondMax] = max(peaksS);
% peaksS(maxPeakIndex) = [];
% locationsS(maxPeakIndex) = [];
% peaksS
% locationsS
lags(locationsS(secondMax))
% diff(locationsS)
period = mean(diff(locationsS));

start = size(subData, 2) - period;
croppedSignal = subData(:, start:end, 4);

ch3CroppedSignal = cropSignal(subData(:, :, 3));

if plotMaxValues
   figure;
   plot(channel4Max);
   
   figure;
   plot(lags, autocor);
   
   hold on;
   % plot(lags(locationsL), peaksL, 'or');
   plot(lags(locationsS), peaksS, 'vk');
   hold off;
end

if plotCroppedSignal
    figure; imagesc(croppedSignal);
    figure; imagesc(ch3CroppedSignal);
end

%% Omitting no reflection measurements

rowsBiases = var(subData(:, start:end, 4), 0, 2) .^ 0.5;
cropLimit = find(rowsBiases > mean(rowsBiases), 1, 'first');

energyRows = sum(subData(:, :, 4).^2, 2);
threshold = mean(energyRows) + var(energyRows) .^ 0.5;
cropLimit = find(energyRows > threshold, 1, 'first') - 10;
i = 2;
while cropLimit <= 0
    threshold = threshold + (var(energyRows) .^ 0.5)/i;
    cropLimit = find(energyRows > threshold, 1, 'first') - 10;
    i = i + 1;
end

if omitReflections
    croppedSignal = croppedSignal(cropLimit:end, :);
    figure; imagesc(croppedSignal);
end