function [left,right]=padding_operation(pos_img,target_scale)
if pos_img-111>0
    left=pos_img-111;
else
    left=1;
end
if pos_img+112<=target_scale
    right=pos_img+112;
else
    right=target_scale;
end
