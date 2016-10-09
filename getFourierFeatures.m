function sample = getFourierFeatures(currAScan)
      sample = zeros(1, 23);

%     % Finding peaks
%     [maxPeaks, maxLocs] = findpeaks(currAScan);
%     [maxValues, maxIdxs] = sort(maxPeaks, 'descend');
%     maxIdx = maxLocs(maxIdxs(1));
% 
%     % Finding valleys
%     [minPeaks, minLocs] = findpeaks(-currAScan);
%     [minValues, minIdxs] = sort(minPeaks, 'descend');
%     minIdx = minLocs(minIdxs(1));

    % Statistical moments
    sample(1) = mean(currAScan);
    sample(2) = std(currAScan);
    sample(3) = skewness(currAScan);
    sample(4) = kurtosis(currAScan);
    [sample(5), maxIdx] = max(currAScan);
    [sample(6), minIdx] = min(currAScan);
    %                 sample(5) = maxValues(1);
    %                 sample(6) = minValues(1);

    % Properties from Signal Processing toolbox
    sample(7) = sum(currAScan.^2);              % Energy
    sample(8) = find(currAScan > sample(1)+sample(2)*1.75, 1, 'first'); % Where most energy concentrates
    sample(9) = wentropy(currAScan, 'shannon'); % Entropy
    sample(10) = sample(5) - sample(6);         % Peak to valley range
    sample(11) = abs(maxIdx - minIdx);          % Peak to valley distance
    sample(12) = rms(currAScan);
    sample(13) = rssq(currAScan);
    sample(14) = peak2rms(currAScan);

    % Properties from Fourier (FIX)
    fftCurrAScan = fft(currAScan - sample(1));
    [maxValue, indexMax] = max(abs(fftCurrAScan));
    sample(15) = maxValue;
    sample(16) = indexMax;

    phases = angle(fft(currAScan - mean(currAScan)));
    sample(17) = phases(indexMax);
    sample(18) = mean(phases);

    % TODO: add more features.
    sample(19) = maxIdx;
    sample(20) = minIdx;
    sample(21) = meanfreq(currAScan);
    sample(22) = medfreq(currAScan);
    sample(23) = median(currAScan);

%     sample(24) = mean(diff(maxLocs(maxIdxs(1:3))));
%     sample(25) = mean(diff(minLocs(minIdxs(1:3))));
%     sample(26) = maxValues(2);
%     sample(27) = minValues(2);
end