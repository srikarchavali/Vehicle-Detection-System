close all;
clear all;
clc;
%%
doTraining = false;
if ~doTraining && ~exist('yolov2ResNet50VehicleExample_19b.mat','file')    
    disp('Downloading pretrained detector (98 MB)...');
    pretrainedURL = 'https://www.mathworks.com/supportfiles/vision/data/yolov2ResNet50VehicleExample_19b.mat';
    websave('yolov2ResNet50VehicleExample_19b.mat',pretrainedURL);
end
%%
% Load Dataset
unzip vehicleDatasetImages.zip
data = load('vehicleDatasetGroundTruth.mat');
vehicleDataset = data.vehicleDataset;
%%
% Display first few rows of the data set.
vehicleDataset(1:4,:)
%%
% Add the fullpath to the local vehicle data folder.
vehicleDataset.imageFilename = fullfile(pwd,vehicleDataset.imageFilename);
%%
% Split the dataset into training, validation, and test sets. 
% Select 60% of the data for training, 10% for validation,...
...and the rest for testing the trained detector.
% rng(0);
shuffledIndices = randperm(height(vehicleDataset));
idx = floor(0.6 * length(shuffledIndices) );

trainingIdx = 1:idx;
trainingDataTbl = vehicleDataset(shuffledIndices(trainingIdx),:);

validationIdx = idx+1 : idx + 1 + floor(0.1 * length(shuffledIndices) );
validationDataTbl = vehicleDataset(shuffledIndices(validationIdx),:);

testIdx = validationIdx(end)+1 : length(shuffledIndices);
testDataTbl = vehicleDataset(shuffledIndices(testIdx),:);

%%
% Creating data stores
imdsTrain = imageDatastore(trainingDataTbl{:,'imageFilename'});
bldsTrain = boxLabelDatastore(trainingDataTbl(:,'vehicle'));

imdsValidation = imageDatastore(validationDataTbl{:,'imageFilename'});
bldsValidation = boxLabelDatastore(validationDataTbl(:,'vehicle'));

imdsTest = imageDatastore(testDataTbl{:,'imageFilename'});
bldsTest = boxLabelDatastore(testDataTbl(:,'vehicle'));
%%
% Combine image and box label datastores.
trainingData = combine(imdsTrain,bldsTrain);
validationData = combine(imdsValidation,bldsValidation);
testData = combine(imdsTest,bldsTest);
%%
% Display one of the training images and box labels
Data = read(trainingData);
I = Data{1};
bbox = Data{2};
annotatedImage = insertShape(I,'Rectangle',bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)
%%
% specify the network input size and the number of classes.
inputSize = [224 224 3];
numClasses = width(vehicleDataset)-1;
%%
trainingDataForEstimation = transform(trainingData,@(data)preprocessData(data,inputSize));
numAnchors = 7;
[anchorBoxes, meanIoU] = estimateAnchorBoxes(trainingDataForEstimation, numAnchors);

% use resnet50 to load a pretrained ResNet-50 model
featureExtractionNetwork = resnet50;

% Select 'activation_40_relu' as the feature extraction layer to replace ...
...the layers after 'activation_40_relu' with the detection subnetwork.
featureLayer = 'activation_40_relu';

% Creating the YOLO v2 object detection network.
lgraph = yolov2Layers(inputSize,numClasses,anchorBoxes,featureExtractionNetwork,featureLayer);

%%
% Data Augmentation

% Using transform to augment the training data by randomly flipping the...
...image and associated box labels horizontally.
augmentedTrainingData = transform(trainingData,@augmentData);

% Visualize the augmented images.
augmentedData = cell(4,1);
for k = 1:4
    data = read(augmentedTrainingData);
    augmentedData{k} = insertShape(data{1},'Rectangle',data{2});
    reset(augmentedTrainingData);
end
figure
montage(augmentedData,'BorderSize',10)

%%
% Preprocess Training Data
% Preprocess the augmented training data, and the validation data...
...to prepare for training.
preprocessedTrainingData = transform(augmentedTrainingData,@(data)preprocessData(data,inputSize));
preprocessedValidationData = transform(validationData,@(data)preprocessData(data,inputSize));

% Read the preprocessed training data.
data = read(preprocessedTrainingData);

% Display the image and bounding boxes.
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'Rectangle',bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)

if doTraining       
    % Train the YOLO v2 detector.
    [detector,info] = trainYOLOv2ObjectDetector(preprocessedTrainingData,lgraph,options);
else
    % Load pretrained detector.
    pretrained = load('yolov2ResNet50VehicleExample_19b.mat');
    detector = pretrained.detector;
    detector2 = detector;
end
%%
% To do a quick test, running the detector on a test image.

I = imread('highway.png');
I = imresize(I,inputSize(1:2));
[bboxes,scores] = detect(detector,I);
% Display the results.
I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
figure
imshow(I)
%%
% Apply the same preprocessing transform to the test data as for the training data.

preprocessedTestData = transform(testData,@(data)preprocessData(data,inputSize));

% Run the detector on all the test images.
detectionResults = detect(detector, preprocessedTestData);

% Evaluate the object detector using average precision metric.
[ap,recall,precision] = evaluateDetectionPrecision(detectionResults, preprocessedTestData);

% The precision/recall (PR) curve highlights how precise a detector is at varying levels of recall.
figure
plot(recall,precision)
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %.2f',ap))

%%
% Read a frame of interest from a video.
vidObj   = VideoReader('05_highway_lanechange_25s.mp4');
vidObj.CurrentTime = 0.1;
I = readFrame(vidObj);

% Load the monoCamera object.
data = load('FCWDemoMonoCameraSensor.mat', 'sensor');
sensor = data.sensor;

% Load the pretrained detector for vehicles.
detector = vehicleDetectorACF();

% Width of a common vehicle is between 1.5 to 2.5 meters. 
vehicleWidth = [1.5, 2.5];

% Configure the detector to take into account configuration of the camera
% and expected vehicle width
detector = configureDetectorMonoCamera(detector, sensor, vehicleWidth);

% Detect vehicles and show the bounding boxes.
[bboxes, ~] = detect(detector, I);
Iout = insertShape(I, 'rectangle', bboxes);
figure;
imshow(Iout)
title('Detected Vehicles')

%Estimate Distances to Detected Vehicles
% Find the midpoint for each bounding box in image coordinates.
midPtsImg = [bboxes(:,1)+bboxes(:,3)/2  bboxes(:,2)+bboxes(:,4)./2];
midPtsWorld = imageToVehicle(sensor, midPtsImg);
x = midPtsWorld(:,1);
y = midPtsWorld(:,2);
distance  = sqrt(x.^2 + y.^2);

% Display vehicle bounding boxes and annotate them with distance in meters.
distanceStr = cellstr([num2str(distance) repmat(' m',[length(distance) 1])]);
Iout = insertObjectAnnotation(I, 'rectangle', bboxes, distanceStr);
imshow(Iout)
title('Distances of Vehicles from Camera')

% Find the midpoint for each bounding box in image coordinates.
midPtsImg = [bboxes(:,1)+bboxes(:,3)/2  bboxes(:,2)+bboxes(:,4)./2];
midPtsWorld = imageToVehicle(sensor, midPtsImg);
x = midPtsWorld(:,1);
y = midPtsWorld(:,2);
distance  = sqrt(x.^2 + y.^2);

% Display vehicle bounding boxes and annotate them with distance in meters.
distanceStr = cellstr([num2str(distance) repmat(' m',[length(distance) 1])]);
Iout = insertObjectAnnotation(I, 'rectangle', bboxes, distanceStr);
imshow(Iout)
title('Distances of Vehicles from Camera')

% text to speech conversion
mindist = min(distance);
speech = ('The distance to the nearest vehicle is'...
    + string(mindist)+'meters');
caUserInput = char(speech); % Convert from cell to string.
NET.addAssembly('System.Speech');
obj = System.Speech.Synthesis.SpeechSynthesizer;
obj.Volume = 100;
Speak(obj, caUserInput);
%%
vidReader = VideoReader('05_highway_lanechange_25s.mp4');
vidPlayer = vision.DeployableVideoPlayer;
i = 1;
results = struct('Boxes',[],'Scores',[]);
while(hasFrame(vidReader))    
    % GET DATA
    I = readFrame(vidReader);    
     fps = 0;
     avgfps = [];
     tic;
    % PROCESS
    [bboxes, scores,label] = detect(detector2,I,'Threshold',0.4);
     newt = toc;
% fps
fps = .9*fps + .1*(1/newt);
avgfps = [avgfps, fps]; %#ok<AGROW>
    
    % Select strongest detection 
    T = 0.0; % Define threshold here
idx = scores >= T;
% Retrieve those scores that surpassed the threshold
s = scores(idx);
% Do the same for the labels as well
lbl = label(idx);
bboxes = bboxes(idx, :); % This logic doesn't change
for ii = 1 : size(bboxes, 1)
    annotation = sprintf('%s: (Confidence = %f)', lbl(ii), s(ii)); % Change    
    I = insertObjectAnnotation(I, 'rectangle', bboxes(ii,:), annotation); % New - Choose the right box
    I  = insertText(I , [1, 1],  sprintf('FPS %2.2f', fps));    
end
step(vidPlayer,I);
 i = i+1;
end
 results = struct2table(results);
 release(vidPlayer);
