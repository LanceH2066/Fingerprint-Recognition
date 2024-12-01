clc, clear, close all;

% Define the path to the Data folder and output folder
dataFolder = fullfile(pwd, 'Data'); % Path to raw data
outputFolder = fullfile(dataFolder, 'Processed'); % Path to processed data

% Create the output folder if it doesn't exist
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Get a list of all fingerprint image files in the folder
imageFiles = dir(fullfile(dataFolder, '*.tif'));

% Initialize arrays to store feature vectors and labels
minutiaeFeatures = [];
lbpFeatures = [];
labels = []; % Ground truth labels (to be defined based on your dataset)

% Loop through each image to extract features and generate labels
for i = 1:length(imageFiles)
    % Load the processed image
    sampleImagePath = fullfile(outputFolder, [imageFiles(i).name(1:end-4) 'processed.tif']);
    if ~exist(sampleImagePath, 'file')
        continue;
    end
    
    sampleImage = imread(sampleImagePath);
    
    % Convert to grayscale if not already
    if size(sampleImage, 3) == 3
        sampleImage = rgb2gray(sampleImage);
    end
    
    % 1. **Minutiae-Based Feature Extraction**
    binarySample = imbinarize(sampleImage);
    thinnedSample = bwmorph(binarySample, 'thin', Inf);
    [minutiaePoints, minutiaeType] = detectMinutiae(thinnedSample);
    minutiaeHistogram = createMinutiaeHistogram(minutiaePoints, size(sampleImage));
    minutiaeFeatures = [minutiaeFeatures; minutiaeHistogram];
    
    % 2. **Texture-Based Feature Extraction (LBP)**
    radius = 1; % Radius for LBP
    numNeighbors = 8; % Number of neighbors for LBP
    lbpFeaturesVector = extractLBPFeatures(sampleImage, 'Radius', radius, 'NumNeighbors', numNeighbors);
    lbpFeatures = [lbpFeatures; lbpFeaturesVector];
    
    % Add label (1 for true, 0 for false or according to your dataset)
    labels = [labels; 1]; % Adjust based on your dataset's ground truth
end

% Split data into training and testing sets
cv = cvpartition(labels, 'HoldOut', 0.3); % 70% training, 30% testing
trainIdx = cv.training();
testIdx = cv.test();

% Train and test the classifiers for minutiae-based features
minutiaeTrainData = minutiaeFeatures(trainIdx, :);
minutiaeTestData = minutiaeFeatures(testIdx, :);
minutiaeTrainLabels = labels(trainIdx);
minutiaeTestLabels = labels(testIdx);

% Train SVM for minutiae-based features
svmMinutiae = fitcsvm(minutiaeTrainData, minutiaeTrainLabels, 'KernelFunction', 'linear', 'Probability', true);
[~, scoreMinutiae] = predict(svmMinutiae, minutiaeTestData);

% Evaluate performance for minutiae-based method
if size(confusionmat(minutiaeTestLabels, predict(svmMinutiae, minutiaeTestData)), 1) > 1
    minutiaeConfMatrix = confusionmat(minutiaeTestLabels, predict(svmMinutiae, minutiaeTestData));
    minutiaeAccuracy = sum(diag(minutiaeConfMatrix)) / sum(minutiaeConfMatrix(:));
    minutiaeTPR = minutiaeConfMatrix(2, 2) / sum(minutiaeConfMatrix(2, :)); % True Positive Rate
    minutiaeFPR = minutiaeConfMatrix(1, 2) / sum(minutiaeConfMatrix(1, :)); % False Positive Rate
else
    minutiaeTPR = NaN;
    minutiaeFPR = NaN;
    fprintf('Confusion matrix for minutiae-based method is single-class. TPR and FPR not calculated.\n');
end

% Plot ROC curve for minutiae-based method
[Xminutiae, Yminutiae, ~, AUCminutiae] = perfcurve(minutiaeTestLabels, scoreMinutiae(:, 2), 'trueclass', 1);

% Train and test the classifiers for texture-based features (LBP)
lbpTrainData = lbpFeatures(trainIdx, :);
lbpTestData = lbpFeatures(testIdx, :);
lbpTrainLabels = labels(trainIdx);
lbpTestLabels = labels(testIdx);

% Train SVM for LBP features
svmLBP = fitcsvm(lbpTrainData, lbpTrainLabels, 'KernelFunction', 'linear', 'Probability', true);
[~, scoreLBP] = predict(svmLBP, lbpTestData);

% Evaluate performance for LBP-based method
if size(confusionmat(lbpTestLabels, predict(svmLBP, lbpTestData)), 1) > 1
    lbpConfMatrix = confusionmat(lbpTestLabels, predict(svmLBP, lbpTestData));
    lbpAccuracy = sum(diag(lbpConfMatrix)) / sum(lbpConfMatrix(:));
    lbpTPR = lbpConfMatrix(2, 2) / sum(lbpConfMatrix(2, :)); % True Positive Rate
    lbpFPR = lbpConfMatrix(1, 2) / sum(lbpConfMatrix(1, :)); % False Positive Rate
else
    lbpTPR = NaN;
    lbpFPR = NaN;
    fprintf('Confusion matrix for LBP-based method is single-class. TPR and FPR not calculated.\n');
end

% Plot ROC curve for LBP-based method
[Xlbp, Ylbp, ~, AUClbp] = perfcurve(lbpTestLabels, scoreLBP(:, 2), 'trueclass', 1);

% Display results
fprintf('Minutiae-based method accuracy: %.2f%%\n', minutiaeAccuracy * 100);
fprintf('Minutiae-based method TPR: %.2f\n', minutiaeTPR);
fprintf('Minutiae-based method FPR: %.2f\n', minutiaeFPR);
fprintf('LBP-based method accuracy: %.2f%%\n', lbpAccuracy * 100);
fprintf('LBP-based method TPR: %.2f\n', lbpTPR);
fprintf('LBP-based method FPR: %.2f\n', lbpFPR);

% Plot ROC curves
figure;
plot(Xminutiae, Yminutiae, 'b', 'DisplayName', 'Minutiae-based ROC');
hold on;
plot(Xlbp, Ylbp, 'r', 'DisplayName', 'LBP-based ROC');
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve Comparison');
legend('show');
grid on;

% Function to detect minutiae points (ridge endings and bifurcations)
function [minutiaePoints, minutiaeType] = detectMinutiae(binaryImg)
    [rows, cols] = size(binaryImg);
    minutiaePoints = [];
    minutiaeType = [];

    % Loop through each pixel to detect minutiae
    for r = 2:rows-1
        for c = 2:cols-1
            if binaryImg(r, c) == 1
                % Analyze the 3x3 neighborhood
                neighborhood = binaryImg(r-1:r+1, c-1:c+1);
                numOnPixels = sum(neighborhood(:));

                % Check for ridge endings (1 on pixel and 3 on neighbors)
                if numOnPixels == 2
                    minutiaePoints = [minutiaePoints; r, c];
                    minutiaeType = [minutiaeType; 'E']; % 'E' for ridge ending
                % Check for bifurcations (1 on pixel and 4 on neighbors)
                elseif numOnPixels == 3
                    minutiaePoints = [minutiaePoints; r, c];
                    minutiaeType = [minutiaeType; 'B']; % 'B' for bifurcation
                end
            end
        end
    end
end

% Function to create a histogram of minutiae points
function minutiaeHistogram = createMinutiaeHistogram(minutiaePoints, imageSize)
    % Initialize a histogram vector (e.g., 10 bins for simplicity)
    numBins = 10; % Adjust as needed for your application
    minutiaeHistogram = zeros(1, numBins);

    % Define the grid size for partitioning the image
    binWidth = floor(imageSize(2) / numBins);
    binHeight = floor(imageSize(1) / numBins);

    % Loop through each minutiae point and count it in the appropriate bin
    for i = 1:size(minutiaePoints, 1)
        r = minutiaePoints(i, 1);
        c = minutiaePoints(i, 2);
        
        % Calculate which bin the point falls into based on its coordinates
        binIndex = min(floor(c / binWidth) + 1, numBins);
        % Optionally, add logic for vertical bins if needed
        % binIndex = min(floor(r / binHeight) + 1, numBins);

        % Increment the count in the corresponding bin
        minutiaeHistogram(binIndex) = minutiaeHistogram(binIndex) + 1;
    end
end
