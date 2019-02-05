function [bbox,scores] = window(window_size,image,svm_model, threshold)
    test_data = [];
    [x,y,~]= size(image);
    position_all = [];
    for i = 1:15:(x-window_size(2))
        for j = 1:15:(y-window_size(1))
            rangex = i+window_size(2)-1;
            rangey = j+window_size(1)-1;
            input_img = image(i:rangex,j:rangey,:);
            input_img = rgb2gray(input_img);
            %feature = extractHOGFeatures(input_img)
            feature = extractHOGFeatures(input_img,'CellSize',[16 16]);
            test_data = [test_data;feature];
            position_all =[position_all;[i,j]];
%             fprintf('\nthe size of the input window: %g %g\n',size(input_img))
%             fprintf('\nthe location \n')
        end
    end
    [predict_label,predict_score] = predict(svm_model,test_data);
    %index = find(predict_label == 1);
    index = find(predict_score(:,2)> threshold);
    bbox = position_all(index,:);
    bbox = [bbox repmat(window_size, size(bbox,1),1)];
    scores = predict_score(index);
    %scores = predict_score;
        
    