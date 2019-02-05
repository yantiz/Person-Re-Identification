function [AP] = evaluate(gt_bbox, bbox, score)
%EVALUATE Summary of this function goes here
%   Detailed explanation goes here
[~, index] = sort(score, 'descend');
bbox = bbox(index,:);
ratio = bboxOverlapRatio(bbox,gt_bbox);
ratio = max(ratio, [], 2);

same_index = index(ratio > 0.3);
if length(same_index) > 0
    AP = compute_AP(same_index, index, size(gt_bbox,1));
else
    AP = 0;
end
% AP = compute_AP(same_index, index, size(gt_bbox,1));


end

