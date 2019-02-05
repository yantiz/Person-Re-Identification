function temp_draw_box(input,location)
% Xmin, Ymin, Xmax, Ymax 31, 326, 209, 712
%location = [31,209;326,712]; % x ; y in annotation
num = size(location,1);
figure; clf;
imshow(input);hold on;
for i = 1:num
    tmp = location(i,:);
    boundingbox = [tmp(2), tmp(1), tmp(3), tmp(4)];
    rectangle('Position',boundingbox,'EdgeColor','r','LineWidth',3);hold on;% x, y consistently
end 
%plot(location(1,1), location(2,1),'-g*');
