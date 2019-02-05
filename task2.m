% Image and Visual Computing Assignment 2: Person Attribute Recognition
%==========================================================================
%   In this assignment, you are expected to use the previous learned method
%   to cope with person attribute recognition problem. The vl_feat, 
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

%% Part II: Person Attribute Recognition: 
%==========================================================================
% The aim of this task is to recognize the attribute of persons (e.g. gender) 
% in the images. We train several binary classifiers to predict 
% whether the person has a certain attribute (e.g. gender) or not.
% - Extract the features
% - Get a data representation for training
% - Train the recognizer and evaluate its performance
%==========================================================================


disp('Person Attribute:Extracting features..')


Xtr = [];
Xte = [];
load('./data/person_attribute_recognition/person_attribute_tr.mat')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loading the training data
% -tr_img/te_img:
% The data is store in a N-by-1 cell array. Each cell is a person image.
% -Ytr/Yte: is a struct where Ytr.bag is a N-by-1 vector of 'Yes' or 'No'
% In this assignment, there are six types of attributes to be recognized: 
% backpack, bag, gender, hat, shoes, upred
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% You need to use your own feature extractor by modifying the ExtractFeatureReid()
%  function or implement your own feature extraction function.
%  For example, use the BoW visual representation (Or any other better representation)
[Xtr, ~] = ExtractFeatureAttribute(tr_img, resize_size);


% BoW visual representation (Or any other better representation)


load('./data/person_attribute_recognition/person_attribute_te.mat')

[Xte, ~] = ExtractFeatureAttribute(te_img, resize_size);

Xtr = double(Xtr);
Xte = double(Xte);


% Train the recognizer and evaluate the performance
%% backpack
model.backpack = fitcsvm(Xtr,Ytr.backpack);
[l.backpack,prob.backpack] = predict(model.backpack,Xte);

% Compute the accuracy
acc.backpack = mean(l.backpack==Yte.backpack)*100;

fprintf('The accuracy of backpack recognition is:%.2f \n', acc.backpack)

%% bag

model.bag = fitcsvm(Xtr,Ytr.bag);
[l.bag,prob.bag] = predict(model.bag,Xte);

% Compute the accuracy
acc.bag = mean(l.bag==Yte.bag)*100;

fprintf('The accuracy of bag recognition is:%.2f \n', acc.bag)

%% gender
model.gender = fitcsvm(Xtr,Ytr.gender);
[l.gender,prob.gender] = predict(model.gender,Xte);

% Compute the accuracy
acc.gender = mean(l.gender==Yte.gender)*100;

fprintf('The accuracy of gender recognition is:%.2f \n', acc.gender)

%% hat

model.hat = fitcsvm(Xtr,Ytr.hat);
[l.hat,prob.hat] = predict(model.hat,Xte);

% Compute the accuracy
acc.hat = mean(l.hat==Yte.hat)*100;

fprintf('The accuracy of hat recognition is:%.2f \n', acc.hat)

%% shoes

model.shoes = fitcsvm(Xtr,Ytr.shoes);
[l.shoes,prob.shoes] = predict(model.shoes,Xte);

% Compute the accuracy
acc.shoes = mean(l.shoes==Yte.shoes)*100;

fprintf('The accuracy of shoes recognition is:%.2f \n', acc.shoes)

%% upred

model.upred = fitcsvm(Xtr,Ytr.upred);
[l.upred,prob.upred] = predict(model.upred,Xte);

% Compute the accuracy
acc.upred = mean(l.upred==Yte.upred)*100;

fprintf('The accuracy of upred recognition is:%.2f \n', acc.upred)

ave_acc = (acc.backpack + acc.bag + acc.gender + acc.hat + acc.shoes + acc.upred) / 6;

fprintf('The average accuracy of attribute recognition is:%.2f \n', ave_acc)


%% Compute the AP
AP = zeros(6,1);
%% backpack

% Compute the AP of searching the people with backpack
index = 1:length(Yte.backpack);
same_index = index(Yte.backpack==1);
[~, index] = sort(prob.backpack(:,2), 'descend');
[AP(1), ~] = compute_AP(same_index, index);

fprintf('The Average Precision of backpack retrieval is:%.2f \n', AP(1))

%% bag

% Compute the AP of searching the people with bag
index = 1:length(Yte.bag);
same_index = index(Yte.bag==1);
[~, index] = sort(prob.bag(:,2), 'descend');
[AP(2), ~] = compute_AP(same_index, index);

fprintf('The Average Precision of bag retrieval is:%.2f \n', AP(2))

%% gender

% Compute the AP of female people retrieval
index = 1:length(Yte.gender);
same_index = index(Yte.gender==1);
[~, index] = sort(prob.gender(:,2), 'descend');
[AP(3), ~] = compute_AP(same_index, index);

fprintf('The Average Precision of female retrieval is:%.2f \n', AP(3))

%% hat

% Compute the AP of hat retrieval
index = 1:length(Yte.hat);
same_index = index(Yte.hat==1);
[~, index] = sort(prob.hat(:,2), 'descend');
[AP(4), ~] = compute_AP(same_index, index);

fprintf('The Average Precision of hat retrieval is:%.2f \n', AP(4))

%% shoes

% Compute the AP of shoes retrieval
index = 1:length(Yte.shoes);
same_index = index(Yte.shoes==1);
[~, index] = sort(prob.shoes(:,2), 'descend');
[AP(5), ~] = compute_AP(same_index, index);

fprintf('The Average Precision of shoes retrieval is:%.2f \n', AP(5))

%% upred

% Compute the AP of upred retrieval
index = 1:length(Yte.upred);
same_index = index(Yte.upred==1);
[~, index] = sort(prob.upred(:,2), 'descend');
[AP(6), ~] = compute_AP(same_index, index);

fprintf('The Average Precision of upred people retrieval is:%.2f \n', AP(6))

mAP = mean(AP);

fprintf('The average accuracy of attribute recognition is:%.2f \n', mAP)



%% Visualization the result of person re-id

data_idx = [12,34,213]; % The index of image in validation set
nPairs = 3; % number of visualize data. maximum is 3
idx_attribute = 3;
% nPairs = length(data_idx); 
visualise_attribute(te_img,prob,Yte,data_idx,nPairs, idx_attribute )