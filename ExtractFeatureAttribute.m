function [train_data,label] = ExtractFeatureAttribute(images, imsize, class)

    if nargin <= 2
        class = 'None';
        label = [];
    end
    
    dir = pwd();
    num = length(images);
    train_data = []; % to hold all the HoG for training
    
    for i = 1:num
        img = images{i};
        img = rgb2gray(img);
        img=imresize(img,imsize,'bilinear');
        tmp = extractHOGFeatures(img,'CellSize',[32 32]);
        train_data = [train_data;tmp]; 
    end 
    
    num = size(train_data,1);
    if strcmp(class, 'pos')
        label = ones(num,1);
    end
    if strcmp(class, 'neg')
        label = zeros(num,1);
    end