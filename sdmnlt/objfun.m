% Objective function of micro-GA
function [f,nd] = objfun(Xtr,Ytr,Xvd,Yvd,params,x)
    % Train of the SDMN-LT model
    dendrite = sdmnlt_train_mga(Xtr,Ytr,params,x);
    if isempty(dendrite)
        f = 1;
        nd = NaN; 
        return; 
    end
    % Predictions on the validation set
    Ypp = sdmnlt_predict(Xvd,params,dendrite);
    Ypp = Ypp'; 
    % Number of dendrites 
    nd = sum(cat(2,dendrite.number));
    % Weighted sum function 
    rr  = (nd-params.c)./(sum(params.ni)-params.c);
    err = mean(Yvd~=Ypp);
    w = params.w;
    f = w*err + (1-w)*rr;
end 