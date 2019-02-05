function visualise_attribute(te_img_sample,prob,Yte,data_idx,nSample, idx_Attri  )
%VISUALISE_VERIFICATION 
%==========================================================================
% Input:
%   va_img_sample: the raw validation cell array
%   prob:        the probability of negative and positive samples which has
%                dimension Nx2.
%   Yva          : The ground truth of validation set
%   data_idx     : The index of data
%   nSample      : The number of visualise pair
%==========================================================================


figure;
hold on

count = 0;
% uniq_Y = unique(Yte);
label_str = {'no', 'yes', 'male','female'};
Attribute = {'backpack','bag','gender','hat','shoes','upred'};
true_false_str = {'False','True'};
if idx_Attri == 1
    prob = prob.backpack;
    Yte = Yte.backpack;
end
if idx_Attri == 2
    prob = prob.bag;
    Yte = Yte.bag;
end
if idx_Attri == 3
    prob = prob.gender;
    Yte = Yte.gender;
end
if idx_Attri == 4
    prob = prob.hat;
    Yte = Yte.hat;
end
if idx_Attri == 5
    prob = prob.shoes;
    Yte = Yte.shoes;
end
if idx_Attri == 6
    prob = prob.upred;
    Yte = Yte.upred;
end
uniq_Y = unique(Yte);
title_str = Attribute{idx_Attri};


for i = 1:length(data_idx)
    count=count+1;
    subplot(nSample,3,count);
    imshow(te_img_sample{data_idx(i),1});
    title('Img')

    count=count+1;
    ax = subplot(nSample,3,count);
    pred_prob = prob(data_idx(i),:);
    
    [~,pred_l] = max(pred_prob);
    if idx_Attri == 3
        pred_label_str = label_str{pred_l+2};
        if Yte(data_idx(i))==1
            Y_str = label_str(4);
        else
            Y_str = label_str(3);
        end
    else
        pred_label_str = label_str{pred_l};
        if Yte(data_idx(i))==1
            Y_str = label_str(2);
        else
            Y_str = label_str(1);
        end
    end
%     pred_label_idx = find(Yte==pred_l);
%     pred_label_str = te_img_sample{pred_label_idx(1),2}(1:9);
%     pred_label_str = strrep(pred_label_str,'_', ' ');
%     label_str = te_img_sample{data_idx(i),2}(1:9);
%     label_str = strrep(label_str,'_', ' ');
    text(0,0.6, 'f(x)' , 'FontSize',14);
    text(0.3,0.6,pred_label_str , 'FontSize',14);
    text(0,0.3, 'y:' , 'FontSize',14);
    text(0.3,0.3,Y_str , 'FontSize',14);
    title(title_str);
    set( ax, 'visible', 'off')
    

    count=count+1;
    ax = subplot(nSample,3,count);
    idx = sum(uniq_Y(pred_l) == Yte(data_idx(i)))+1;
    text(0.3,0.9, 'Is f(x)=y ?', 'FontSize',16);
    text(0.5,0.5,true_false_str{idx} , 'FontSize',14);
    title(title_str);
    set ( ax, 'visible', 'off')
end

saveas(gcf, 'visualise_attribute.png')

end

