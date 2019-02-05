function [image] = crop_bbox(img,bbox)
%CROP_BBOX Summary of this function goes here
%   Detailed explanation goes here
num_bbox = size(bbox,1);
image = cell(num_bbox,1);
for i = 1:num_bbox
    bbox_ = bbox(i,:);
    x = bbox_(1);
    y = bbox_(2);
    w = bbox_(3);
    h = bbox_(4);
    image{i} = img(x:x+h,y:y+w,:);
end
end

