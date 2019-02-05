% Image and Visual Computing Assignment 1: Person Re-identification
%==========================================================================
%   In this assignment, you are expected to use the previous learned method
%   to cope with person re-identification problem. The vl_feat, 
%   libsvm, liblinear and any other classification and feature extraction 
%   library are allowed to use in this assignment. The built-in matlab 
%   object-detection functionis not allowed. Good luck and have fun!
%
%                                               Released Date:   7/11/2018
%==========================================================================

%% Initialisation
%==========================================================================
% Add the path of used library.
% - The function of adding path of liblinear and vlfeat is included.
%==========================================================================
clear all
clc
run ICV_setup

% Hyperparameter of experiments
resize_size=[128 64];


%% Part 1: Person Re-identification (re-id): 
%==========================================================================
% The aim of this task is to verify whether the two given people in the
% images are the same person. We train a binary classifier to predict
% whether these two people are actually the same person or not.
% - Extract the features
% - Get a data representation for training
% - Train the re-identifier and evaluate its performance
%==========================================================================


disp('Person Re-id:Extracting features..')

load('./data/person_re-identification/person_re-id_train.mat')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loading the training data
% - train, a 1 x N struct array with fields image1, id1, image2 and id2
% - image1 and image2 are two image while id1 and id2 are the corresponding
% label. Here, we generate a label vector (i.e. Ytr) which indicates image1 
% is the same person as the image2 or not, where, Ytr = 1 indicates 'same' 
% and Ytr = -1 represents 'different'
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% You should construct the features in here. (read, resize, extract)
image1 = {train(:).image1}';
image2 = {train(:).image2}';
image1 = cat(4, image1{:});
image2 = cat(4, image2{:});

[X1, ~] = AlexNet_ExtractFeatureReid(image1, resize_size);
[X2, ~] = AlexNet_ExtractFeatureReid(image2, resize_size);
id1 = [train(:).id1]';
id2 = [train(:).id2]';
Ytr = ones(length(id1),1);
Ytr(id1 ~= id2) = -1;

best_mAP = 0;

% Parameters to tune:
best_C = 1e-4;

rng(0);

%for C = logspace(-5,5,10)
for C = 1e-4
        % Cross Validation for mAP measurement
        k_fold = 5;
        
        indices = crossvalind('Kfold', size(image1, 4), k_fold);
        mAP_list = zeros(k_fold, 1);

        for c = 1:k_fold
valid_index = (indices == c); 
train_index = ~valid_index;

Xtr1 = X1(train_index,:);
Xva1 = X1(valid_index,:);
Xtr2 = X2(train_index,:);
Xva2 = X2(valid_index,:);
id1_train = id1(train_index); 
id1_valid = id1(valid_index);
id2_train = id2(train_index);
id2_valid = id2(valid_index);

%% You need to use your own feature extractor by modifying the ExtractFeatureReid()
%  function or implement your own feature extraction function.
%  For example, use the BoW visual representation (Or any other better representation)

Xtr = abs(Xtr1 - Xtr2);
Xtr = double(Xtr);

% Train the re-identifier and evaluate the performance
%==========================================================================
% Try to train a better classifier.
%==========================================================================#
model = fitcsvm(Xtr, Ytr(train_index),'BoxConstraint',C);

% Constructing query-gallery pairs and label vector for validation set
num_va1 = size(Xva1, 1);
num_va2 = size(Xva2, 1);
Xva = [];
Yva = [];

for i = 1:num_va1
    Xva1_ = Xva1(i,:);
    temp_Xva = abs(Xva1_ - Xva2);
    Xva = [Xva; temp_Xva];
    temp_Yva = ones(num_va2,1);
    temp_Yva(id2_valid ~= id1_valid(i)) = -1;
    Yva = [Yva; temp_Yva];
end

Xva = double(Xva);

[l,prob] = predict(model, Xva);

% Compute the mean Average Precision
ap = zeros(num_va1, 1);
for i = 1:num_va1
    prob_i = prob((i - 1) * num_va2 + 1: i * num_va2,2);
    [~, sorted_index] = sort(prob_i, 'descend');
    temp_index = 1:num_va2;
    same_index = temp_index(id2_valid == id1_valid(i));
    [ap(i), ~] = compute_AP(same_index, sorted_index);
end
mAP_list(c) = nanmean(ap);
        end

        mAP = mean(mAP_list);
        
        if best_mAP < mAP
            best_mAP = mAP;
            
            best_C = C;
        end
        
        fprintf('The mAP of person re-identification on validation set for C=%.2e is:%.2f \n', best_C, mAP * 100)
end

Xtr = abs(X1 - X2);
Xtr = double(Xtr);

model = fitcsvm(Xtr, Ytr,'BoxConstraint',best_C);

% model = fitcknn(Xtr,Ytr,'NumNeighbors',3);
save('alexnet_person_reid_model.mat','model');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loading the testing data
% - query, a 1 x N struct array with fields image and id
% - gallery, a 1 x N struct array with fiedls image and id
% - In the testing, we aim at re-identifying the query person in the
% gallery images.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('./data/person_re-identification/person_re-id_test.mat')

image_query = {query(:).image}';
image_query = cat(4, image_query{:});
id_query = [query(:).id]';
image_gallery = {gallery(:).image}';
image_gallery = cat(4, image_gallery{:});
id_gallery = [gallery(:).id]';
num_query = size(image_query, 4);
num_gallery = size(image_gallery, 4);

% Extracting query features
[Xq, ~] = AlexNet_ExtractFeatureReid(image_query, resize_size);

% Extracting gallery features
[Xg, ~] = AlexNet_ExtractFeatureReid(image_gallery, resize_size);

% Constructing query-gallery pairs and label vector
Xte = [];
Yte = [];
for i = 1:num_query
    Xq_ = Xq(i,:);
    temp_Xte = abs(Xq_ - Xg);
    Xte = [Xte; temp_Xte];
    temp_Yte = ones(num_gallery,1);
    temp_Yte(id_gallery ~= id_query(i)) = -1;
    Yte = [Yte; temp_Yte];
end

Xte = double(Xte);

[l,prob] = predict(model,Xte);

% Compute the mean Average Precision
ap = zeros(num_query, 1);
for i = 1:num_query
    prob_i = prob((i - 1) * num_gallery + 1: i * num_gallery,2);
    [~, sorted_index] = sort(prob_i, 'descend');
    temp_index = 1:num_gallery;
    same_index = temp_index(id_gallery == id_query(i));
    [ap(i), ~] = compute_AP(same_index, sorted_index);
end
mAP = mean(ap);
fprintf('The mean Average Precision of person re-identification on test set is:%.2f \n', mAP * 100)

% fprintf('The accuracy of face recognition is:%.2f \n', acc)



%% Visualization the result of person re-id
image_query = {query(:).image}';
image_gallery = {gallery(:).image}';

query_idx = [2,15,26]; 
gallery_idx = [2, 1, 45];
l_ = reshape(l(:), [90, 30])';
Yte_ = reshape(Yte, [90, 30])';
nPairs = 3; % number of visualize data. maximum is 3
% nPairs = length(data_idx); 
visualise_reid(image_query, image_gallery, query_idx, gallery_idx, l_, Yte_, nPairs )