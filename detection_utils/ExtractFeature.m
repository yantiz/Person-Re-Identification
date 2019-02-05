function [train_data,label] = ExtractFeature(images, imsize, class)
    dir = pwd();
    num = length(images);
    train_data = []; % to hold all the HoG for training
    
    for i = 1:num
        img = images{i};
        img = rgb2gray(img);
        img=imresize(img,imsize,'bilinear');
        
        tmp = extractHOGFeatures(img,'CellSize',[16 16]);
        train_data = [train_data;tmp]; 
    end 
    
    num = size(train_data,1);
    if class == 'pos'
        label = ones(num,1);
    end
    if class == 'neg'
        label = zeros(num,1);
    end