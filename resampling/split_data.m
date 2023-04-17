% Split data for the execution of the SDMN-LT experiments
function [Xtr,Ytr,Xvd,Yvd] = split_data(X,Y,p)
    [tr,vd] = crossvalind('HoldOut',Y,p);
    Xtr = X(tr,:);
    Ytr = Y(tr,:);
    Xvd = X(vd,:);
    Yvd = Y(vd,:);
end