function [train_data,label] = ExtractFeatureReid(images, imsize, num_bins, hog_weight, class)
    
    if nargin <= 4
        class = 'None';
        label = [];
    end
    
    dir = pwd();
    num = length(images);
    train_data = []; % to hold all training feature representations
    
    for i = 1:num
        img_rgb = images{i};
        img = rgb2gray(img_rgb);
        img = imresize(img,imsize,'bilinear');
        features = [];
        
        % Rescaled Color Histograms for the upper and lower half images:
        h = size(img_rgb, 1);
        h = h / 2;
        
        for j = 1:2
            img_rgb_half = img_rgb((j-1)*h+1:j*h,:,:);

            if j == 1
                % Extract the upper body for a person for clothes texture:
                img_upperhalf = img(1:h,:);
                img_upperhalf_smoothed = imgaussfilt(img_upperhalf, 0.8);
                
                %hog = extractHOGFeatures(img_upperhalf,'CellSize',[16 16]);
                hog = extractHOGFeatures(img_upperhalf_smoothed,'CellSize',[16 16],'NumBins',10);
                
                features = [features, hog * hog_weight];
            end

            histR = imhist(img_rgb_half(:,:,1), 13).';
            histG = imhist(img_rgb_half(:,:,2), 13).';
            histB = imhist(img_rgb_half(:,:,3), 13).';
        
            histR = rescale(histR);
            histG = rescale(histG);
            histB = rescale(histB);
            
            features = [features, [histR, histG, histB] * (1 - hog_weight)];
        end
        
        % Combine features:
        train_data = [train_data; features]; 
    end 
    
    num = size(train_data,1);
    if strcmp(class, 'pos')
        label = ones(num,1);
    end
    if strcmp(class, 'neg')
        label = zeros(num,1);
    end