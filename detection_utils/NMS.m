function [reserve_bbox, reserve_score] = NMS(bbox, score, threshold)

num_b = size(bbox,1);
reserve_bbox = [];
[score, index] = sort(score, 'descend');
bbox = bbox(index,:);
reserve_bbox = [reserve_bbox; bbox(1,:)];
reserve_score = score(1);
for i = 2:num_b
    flag = 1;
    for j = 1:size(reserve_bbox,1)
        if bboxOverlapRatio(bbox(i,:),reserve_bbox(j,:)) > threshold
            flag = 0;
        end
    end
    if flag == 1
        reserve_bbox = [reserve_bbox; bbox(i,:)];
        reserve_score = [reserve_score; score(i)];
    end
end

end