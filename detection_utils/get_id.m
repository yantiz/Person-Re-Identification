function [id] = get_id(bbox,gt_bbox, g_id)
%GET_ID Summary of this function goes here
%   Detailed explanation goes here
id = -1 * ones(size(bbox,1),1);
for i = 1:size(gt_bbox)
    ratio = bboxOverlapRatio(bbox, gt_bbox(i,:));
    id(ratio > 0.3) = g_id(i);
end

end

