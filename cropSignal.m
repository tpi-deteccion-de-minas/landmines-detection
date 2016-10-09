function [croppedSignal, cropLimit, status] = cropSignal(signal, cropMode)
    % cropMode: decides wether the "noise" area is deleted or not.
    %
    % Cropping exit status code
    %  0: successful and no problems.
    %  1: some significant reflections at the beginning.
    %  2: no-reflection area could not be determined.
    
    status = 0;
    
    % Finding the period
    energyRows = sum(signal.^2, 2);
    [~, indexMax] = max(energyRows);
    [autocor, lags] = xcorr(signal(indexMax, :));
    autocor = autocor/abs(max(autocor));
    [peaks, locs] = findpeaks(autocor, 'MinPeakDistance', 500);
    [~, maxPeakIndex] = max(peaks);
    peaks(maxPeakIndex) = NaN;
    [~, secondMax] = max(peaks);
    period = abs(lags(locs(secondMax)));
    start = ceil(size(signal, 2) - period);
    croppedSignal = signal(:, start:end);
    
    if cropMode ~= 0
        % Deleting the no-reflection area
        threshold = mean(energyRows) + var(energyRows) .^ 0.5;
        cropLimit = find(energyRows > threshold, 1, 'first') - 10;
        i = 2;
        while cropLimit <= 0
            status = 1;
            threshold = threshold + (var(energyRows) .^ 0.5)/i;
            cropLimit = find(energyRows > threshold, 1, 'first') - 10;
            i = i + 1;
            if i == 11 % Data cannot be cropped
                status = 2;
                cropLimit = 1;
                break;
            end
        end
        croppedSignal = croppedSignal(cropLimit:end, :);
    else
        cropLimit = 0;
        status = 0;
    end    
end