function [bbox] = adjust_bbox(bbox,W,H,W_,H_)
%ADJUST_BBOX Summary of this function goes here
%   Detailed explanation goes here
bbox = [bbox(:,2) * W_ / W bbox(:,1) * H_ / H bbox(:,3) * W_ / W bbox(:,3) * H_ / H];

end

