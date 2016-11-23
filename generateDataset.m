function [X, Y] = generateDataset(folder, files, datasetChoice)
    % datasetChoice:
    %   0: Fourier features
    %   1: Spectrogram

    label0DCount = 0;       % Counting each column as a sample
    label1DCount = 0;       % Counting each column as a sample
    idx = 1;                % Example counter
    
    % "No mine" is represented by a 0.
    % "Mine" is represented by a 1.

    labels = uint8(zeros(170000,1));
    
    % Object IDs extracted from file
    objectIds = uint8(zeros(170000,1));
    
    % IMPORTANT: change the matrix dimensions according to the
    % modifications made.
    switch datasetChoice
        case 0 % Fourier            
            features = single(zeros(170000, 23));
            datasetPath = strcat('dataset/rawNoCrop/csv_fourier');
            datasetName = 'descriptors';
        case 1 % Spectrogram
            features = single(zeros(170000, 2304));
            datasetPath = strcat('dataset/rawNoCrop/csv_spectrogram');
            datasetName = 'spectrogram_48x48';
        case 2 % Spectrogram images
            datasetPath = strcat('dataset/rawNoCrop/img_spectrogram/images');           
        case 3
            features = single(zeros(170000,1024));
            datasetPath = strcat('dataset/rawNoCrop/h5_wavelets');
            datasetName = 'wavelets_32x32';
        otherwise
            error('Choose a valid dataset type.')
    end
    
    if ~exist(datasetPath, 'dir')
        mkdir(datasetPath);
    end
    
    for i = 1:size(files, 1)
        currentFile = strcat(folder{i}, '/', files(i).name);
        load(currentFile);
        
        if exist('subData')
            fprintf('\nProcessing file %s', currentFile)
            for j = 1:size(subData, 3)
                signal.(strcat('ch', num2str(j))) = subData(:, :, j);
            end
            clear('subData');
        else            
            numSamplesCh3 = size(signal.ch3, 2);
            numSamplesCh4 = size(signal.ch4, 2);
            objectNumberIdx = regexp(files(i).name, '[0-9]');
            fprintf('\nProcessing file %s %i %i ID: %i...', currentFile, numSamplesCh3, numSamplesCh4, ...
                uint8(str2double(files(i).name(objectNumberIdx(2:end)))))

            % 0: no mine, 1: mine
            if pInfo.label == 0
                label0DCount = label0DCount + numSamplesCh3 + numSamplesCh4;
            else
                label1DCount = label1DCount + numSamplesCh3 + numSamplesCh4;
            end
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
        %   - The processing involving the Time-Frequency transform should give
        %     a fixed size image for each sample in order to feed the convolu-
        %     tional neural network.

        % The variable 'sample' stores the features or characteristics of the
        % signal being currently processed. The second dimension size must
        % match the number of features being calculated. For this initial setup
        % we have 18 features, hence, a 1-by-18 vector is created to store
        % those features.
        
        channels = fieldnames(signal);
        for j = 1:length(channels)
            currChannel = signal.(channels{j});
            currSignalSize = size(currChannel, 2);
            for k = 1:currSignalSize
                currAScan = currChannel(:, k);
                
                % Adding the sample to the generated dataset
                if (isfield(pInfo, 'label'))
                    labels(idx) = uint8(pInfo.label);
                else
                    labels(idx) = 0;
                end
                
                % Extracting the object ID
                objectNumberIdx = regexp(files(i).name, '[0-9]');
                objectIds(idx) = uint8(str2double(files(i).name(objectNumberIdx(2:end))));
                
                switch datasetChoice
                    case 0 % Fourier features
                        features(idx,:) = getFourierFeatures(currAScan);
                    case 1 % Spectrogram 48x48
                        s = single(abs(spectrogram(currAScan, 128, 120, 94))');
                        features(idx,:) = s(:)';
                    case 2 % Spectrogram 224x224
                        s = abs(spectrogram(currAScan, 288, 287, 446));
                        % Normalazing between [0,1] to save image
                        s = s - min(s(:));
                        s = s / max(s(:));
                        filename = strcat(datasetPath, '/', num2str(labels(idx)), '_', ...
                                num2str(j), '-', num2str(k), '-', pInfo.mMeasDesc, '.jpg');                    
                        imwrite(s, filename);
                    case 3 % Wavelets 32x32
                        features(idx,:) = getWaveletsv2(currAScan)';
                end
                idx = idx + 1;
            end
        end
    end

    % Some stats about the dataset
    totalSamples = label0DCount + label1DCount;
    fprintf('\n\nNo. of samples labeled as %s: %i', '"No Mine"', label0DCount)
    fprintf('\nNo. of samples labeled as %s: %i', '"Mine"', label1DCount)
    fprintf('\nTotal of labeled samples: %i', totalSamples)
    
    switch datasetChoice
        case 0
            sprintf('\nChosen dataset: Fourier features.\n');
        case 1
            imageShape = size(s);
            fprintf('\nChosen dataset: Spectrogram. Image shape: (%i,%i)\n', imageShape(1), imageShape(2));
        case 2
            imageShape = size(s);
            fprintf('\nChosen dataset: Wavelets. Image shape: (%i,%i)\n', imageShape(1), imageShape(2));
    end

    %% The generated dataset is converted from a cell array to matrix.
    %
    % X stores the generated features of each sample.
    % Y stores the label or class associated to each sample.

    if datasetChoice ~= 2
        X = features(1:totalSamples, :);
        Y = labels(1:totalSamples, :);
        objectIds = objectIds(1:totalSamples, :);
    else
        return
    end
    
    datasetPath = strcat(datasetPath, '/', datasetName, '.h5');
    
    if ~exist(datasetPath, 'file')
        h5create(datasetPath, '/data', size(X));
        h5create(datasetPath, '/labels', size(Y));
        h5create(datasetPath, '/objectIds', size(objectIds));
    end
    
    h5write(datasetPath, '/data', X)
    h5write(datasetPath, '/labels', Y)
    h5write(datasetPath, '/objectIds', objectIds)
end