function visualise_reid(query_image, gallery_image, query_idx, gallery_idx,label,Yte,nPairs  )
%VISUALISE_VERIFICATION 
%==========================================================================
% Input:
%   query_image/gallery_image: the raw validation cell array
%   query_idx/gallery_idx:   index of query image and gallery image
%   label      : the predicted lable
%   Yte        : The ground truth of testing set
%   nPairs     : The number of visualise pair
%==========================================================================


figure;
hold on

count = 0;
% uniq_Y = unique(Yte);
label_str = {'Diffferent', 'Same'};
true_false_str = {'False','True'};

for i = 1:length(query_idx)
    count=count+1;
    subplot(nPairs,4,count);
    imshow(query_image{query_idx(i)});
    title('Img 1')

    count=count+1;
    subplot(nPairs,4,count);
    imshow(gallery_image{gallery_idx(i)});
    title('Img 2')

    count=count+1;
    ax = subplot(nPairs,4,count);
    pred_prob = label(query_idx(i), gallery_idx(i));
    gt_prob = zeros(1,2); % Ground Truth
    if Yte(query_idx(i), gallery_idx(i)) == -1;
        gt_prob(1)=1;
        gt_idx = 1;
    else
        gt_prob(2)=1;
        gt_idx = 2;
    end
    
%     [~,pred_l] = max(pred_prob);
    if pred_prob == 1
        pred_l = 2;
    else
        pred_l = 1;
    end
    idx = (pred_l == gt_idx)+1;
     
    text(0,0.6, 'f(x)' , 'FontSize',14);
    text(0.5,0.6,label_str{pred_l} , 'FontSize',14);
    text(0,0.3, 'y:' , 'FontSize',14);
    text(0.5,0.3,label_str{gt_idx} , 'FontSize',14);
    set ( ax, 'visible', 'off')
    

    count=count+1;
    ax = subplot(nPairs,4,count);
    
    
    text(0.3,0.9, 'Is f(x)=y ?', 'FontSize',16);
    text(0.5,0.5,true_false_str{idx} , 'FontSize',14);
    set ( ax, 'visible', 'off')
end

saveas(gcf, 'visualise_re-id.png')

end

