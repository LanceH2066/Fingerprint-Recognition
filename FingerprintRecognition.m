%% Initialization
clc; clear; close all;

% Paths to data
rawDataPath = './data';
processedDataPath = './data/processed';

% Create processed folder if it doesn't exist
if ~exist(processedDataPath, 'dir')
    mkdir(processedDataPath);
end

% Parameters
numClasses = 100; % Number of fingerprint classes
numSamplesPerClass = 8; % Samples per class

%% Data Preparation
% Initialize data containers
labels = [];
featuresMinutiae = [];
featuresTexture = [];

for classIdx = 1:numClasses
    for sampleIdx = 1:numSamplesPerClass
        % Construct file name
        rawFileName = sprintf('%d_%d.tif', classIdx, sampleIdx);
        processedFileName = sprintf('%d_%d_processed.tif', classIdx, sampleIdx);
        rawFilePath = fullfile(rawDataPath, rawFileName);
        processedFilePath = fullfile(processedDataPath, processedFileName);

        % Check if the file is already processed
        if ~exist(processedFilePath, 'file')
            % Read raw image
            img = imread(rawFilePath);

            % Preprocess image
            preprocessedImg = preprocessFingerprint(img);

            % Save the processed image
            imwrite(preprocessedImg, processedFilePath);
        else
            % Load the preprocessed image
            preprocessedImg = imread(processedFilePath);
        end

        % Feature extraction
        minutiaeFeatures = extractMinutiaeFeatures(preprocessedImg);
        textureFeatures = extractTextureFeatures(preprocessedImg);

        % Store features and labels
        featuresMinutiae = [featuresMinutiae; minutiaeFeatures];
        featuresTexture = [featuresTexture; textureFeatures];
        labels = [labels; classIdx];
    end
end

%% Split Data into Training and Testing Sets
% Use 70% for training and 30% for testing
[trainIdx, testIdx] = crossvalind('HoldOut', labels, 0.3);

trainLabels = labels(trainIdx);
testLabels = labels(testIdx);

trainMinutiae = featuresMinutiae(trainIdx, :);
testMinutiae = featuresMinutiae(testIdx, :);

trainTexture = featuresTexture(trainIdx, :);
testTexture = featuresTexture(testIdx, :);

%% Classification and Performance Evaluation
% Initialize storage for results
results = struct();

% Classifiers
classifiers = {'SVM', 'kNN'};
methods = {'Minutiae', 'Texture'};

figure; hold on; grid on;
colors = {'r', 'b', 'g', 'k'}; % Colors for each ROC curve
legendEntries = {};

for methodIdx = 1:length(methods)
    for classifierIdx = 1:length(classifiers)
        % Select features and classifier
        if methodIdx == 1
            trainData = trainMinutiae;
            testData = testMinutiae;
        else
            trainData = trainTexture;
            testData = testTexture;
        end

        trainData = zscore(trainData);
        testData = zscore(testData);

        if classifierIdx == 1
            % Train SVM (using fitcecoc for multiclass support)
            t = templateSVM('KernelFunction', 'rbf', 'KernelScale', 'auto', 'BoxConstraint', 1);
            model = fitcecoc(trainData, trainLabels, 'Learners', t, 'Coding', 'onevsall');
            [predictions, scores] = predict(model, testData);
        else
            % Train k-NN
            model = fitcknn(trainData, trainLabels, 'NumNeighbors', 5, ...
                'Distance', 'euclidean', 'DistanceWeight', 'inverse', 'Standardize', true);
            [predictions, scores] = predict(model, testData);
        end
        
        % Evaluate performance
        [X, Y, T, AUC] = perfcurve(testLabels, scores(:, 2), 1);

        % Store results
        results.(methods{methodIdx}).(classifiers{classifierIdx}) = struct(...
        'X', X, 'Y', Y, 'AUC', AUC);

        % Get ROC data
        X = results.(methods{methodIdx}).(classifiers{classifierIdx}).X;
        Y = results.(methods{methodIdx}).(classifiers{classifierIdx}).Y;
        AUC = results.(methods{methodIdx}).(classifiers{classifierIdx}).AUC;

        % Plot ROC curve
        plot(X, Y, 'LineWidth', 1.5, 'Color', colors{(methodIdx - 1) * 2 + classifierIdx});
        
        % Add to legend
        legendEntries{end + 1} = sprintf('%s + %s (AUC: %.2f)', methods{methodIdx}, classifiers{classifierIdx}, AUC);
    end
end

% Customize the plot
title('Comparison of ROC Curves');
xlabel('False Positive Rate');
ylabel('True Positive Rate');
legend(legendEntries, 'Location', 'SouthEast');

%% Helper Functions
function features = extractMinutiaeFeatures(img)

    if ~islogical(img)
        binary_image = imbinarize(img); % Only binarize if the image is not binary
    else
        binary_image = img; % Keep the image as-is if it's already binary
    end

    % Get image size
    [rows, cols] = size(binary_image);
    
    % Define cropping region
    startRow = min(120, rows); % Ensure startRow doesn't exceed the number of rows
    endRow = min(400, rows);  % Ensure endRow doesn't exceed the number of rows
    startCol = min(20, cols); % Ensure startCol doesn't exceed the number of columns
    endCol = min(250, cols);  % Ensure endCol doesn't exceed the number of columns
    
    % Adjust region dynamically based on image size
    binary_image = binary_image(startRow:endRow, startCol:endCol);
    
    % Thinning
    thin_image = ~bwmorph(binary_image, 'thin', Inf);
    
    % Minutiae extraction
    s = size(thin_image);
    N = 3; % Window size
    n = (N - 1) / 2;
    r = s(1) + 2 * n;
    c = s(2) + 2 * n;
    temp = zeros(r, c);
    ridge = zeros(r, c);
    bifurcation = zeros(r, c);
    temp((n + 1):(end - n), (n + 1):(end - n)) = thin_image;
    
    for x = (n + 1 + 10):(s(1) + n - 10)
        for y = (n + 1 + 10):(s(2) + n - 10)
            e = 1;
            for k = x - n:x + n
                f = 1;
                for l = y - n:y + n
                    mat(e, f) = temp(k, l);
                    f = f + 1;
                end
                e = e + 1;
            end
            if mat(2, 2) == 0
                ridge(x, y) = sum(sum(~mat));
                bifurcation(x, y) = sum(sum(~mat));
            end
        end
    end
    
    % Ridge and bifurcation counts
    ridge_count = length(find(ridge == 2));
    bifurcation_count = length(find(bifurcation == 4));
    
    % Combine features
    features = [ridge_count, bifurcation_count];
end


function features = extractTextureFeatures(img)
    % Convert image to grayscale if needed
    if size(img, 3) == 3
        img = rgb2gray(img);
    end

    % Extract LBP features
    features = extractLBPFeatures(img, 'Upright', true, 'CellSize', [16 16]);
end

function preprocessedImg = preprocessFingerprint(img)
    % Normalize image intensity
    normalizedImg = mat2gray(img);

    % Reduce noise
    filteredImg = medfilt2(normalizedImg, [3 3]);

    % Enhance contrast
    enhancedImg = adapthisteq(filteredImg);

    % Binarize image
    binaryImg = imbinarize(enhancedImg, adaptthresh(enhancedImg, 0.5));

    % Thin ridges
    preprocessedImg = bwmorph(binaryImg, 'thin', Inf);
end