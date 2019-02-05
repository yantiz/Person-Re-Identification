function [train_data,label] = AlexNet_ExtractFeatureReid(images, imsize, class)
    
    if nargin <= 4
        class = 'None';
        label = [];
    end
    
    dir = pwd();
    num = length(images);
    
    % Use AlexNet to extract features:
    net = alexnet;
    layer = 'fc7';
    
    images = augmentedImageDatastore(net.Layers(1).InputSize(1:2),images);
    
    train_data = activations(net,images,layer,'OutputAs','rows');
    
    num = size(train_data,1);
    if strcmp(class, 'pos')
        label = ones(num,1);
    end
    if strcmp(class, 'neg')
        label = zeros(num,1);
    end