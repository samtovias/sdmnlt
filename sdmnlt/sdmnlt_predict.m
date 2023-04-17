% SDMN-LT prediction function
function [Yp,Pr] = sdmnlt_predict(X,params,dendrite)
    % Definitions of distance, softmax, minmax and smax functions
    distance = params.distance;
    softmax = params.softmax;
    minmax  = params.minmax; 
    smax = params.smax;
    % Parameters 
    c = params.c;
    n = size(X,1);
    R  = zeros(n,c);
    % Compute dendrite responses 
    for i = 1:c
        D = bsxfun(@minus,dendrite(i).radii,pdist2(X,dendrite(i).C,distance));
        R(:,i) = smax(D,dendrite(i).beta);
    end
    R = [ones(n,1) R]*dendrite(c+1).weights;
    % Get predictions
    Pr = softmax(minmax(R));
    [~,Yp] = max(Pr,[],2);
    Yp = Yp';
    Pr = Pr';
end