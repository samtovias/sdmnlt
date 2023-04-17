% SDMN-LT training
function [dendrite,params] = sdmnlt_train(X,Y,x)
    % Set the minimum number of instance per dendrite 
    mpd = 3; 
    % Set params and verify hyperparameters 
    params = setparams(X,Y,x);
    % Get params
    LT = params.LT; 
    distance = params.distance; 
    smax = params.smax; 
    T = params.Ttr; 
    % Initialize model struct
    dendrite = struct([]);
    while isempty(dendrite)
        cts = [];
        mrk = [];
        nd  = zeros(1,params.c);
        % Get centroids for each dendrite
        for i = 1:params.c
            Ti = LT{i};
            th = Ti(x(i),3);
            Xi = X(Y==i,:);
            Ci = cluster(Ti,'Cutoff',th,'Criterion','distance'); 
            nd(i) = max(Ci);
            Ct = zeros(nd(i),params.d);
            flag = false(nd(i),1);
            for j = 1:nd(i)
                Xj = Xi(Ci==j,:);
                if size(Xj,1) >= mpd
                    Ct(j,:) = mean(Xj,1); % Compute centroid
                else
                    flag(j) = true;
                    continue;
                end
            end
            if sum(flag==1) == nd(i)
                dendrite = []; 
                % Reduce the MPD
                if mpd > 1
                    mpd = mpd - 1;
                end
                break; 
            end
            Ct(flag,:) = [];
            nd(i) = size(Ct,1);
            dendrite(i).C = Ct;
            cts = cat(1,cts,Ct);
            mrk = cat(1,mrk,i*ones(nd(i),1));
        end
    end
    % Compute radii 
    D = pdist2(cts,cts,distance,'Smallest',3);
    if size(D,2) > 2
        s = mean(D(2:3,:),1);
    else
        s = D(2,:);
    end
    % Responses of the dendrite cluster
    n = params.n;
    R  = zeros(n,params.c);
    for i = 1:params.c
        dendrite(i).radii = s(mrk==i);
        dendrite(i).number = nd(i);
        dendrite(i).beta = x(params.c+1);
        a = dendrite(i).radii;
        b = pdist2(X,dendrite(i).C,distance);
        D = bsxfun(@minus,a,b);
        R(:,i) = smax(D,dendrite(i).beta);
    end
    % Gradient descend cross entropy and softmax
    yj = [ones(params.n,1) R];
    a = 1/sqrt(params.c);
    W = -a + 2*a*rand(params.c+1,params.c);
    B = zeros(params.c+1,params.c);
    J = zeros(1,params.maxepoch);
    for t = 1:params.maxepoch
        z = params.softmax(params.minmax(yj*W));
        D = params.eta*yj'*(T-z);
        V = params.alpha*(params.alpha*B+D)+D; % Nesterov momentum
        W = W + V;
        B = V;
        J(t) = mean(-sum(T.*log(z+eps),2));
    end
    dendrite(params.c+1).weights = W;
    dendrite(params.c+2).loss = J;
end

%--------------------------------
% Functions 

% Setup parameters 
function params = setparams(Xtr,Ytr,x) 
    % Base
    n = numel(Ytr); 
    d = size(Xtr,2); 
    c = max(Ytr); 
    ni = accumarray(Ytr,ones(n,1),[c 1],@sum,0)';
    % Verify hyperparameters
    for i = 1:c 
        if x(i) < 0 || x(i) > ni(i)-1
            error('ith cut-off point is outside range [0,ni-1].'); 
        end
    end
    if x(i+1) < 0
        error('Smoothness factor must be equal or greater than 0.'); 
    end
    distance = 'squaredeuclidean';
    params.distance = distance; 
    softmax = @(x)(bsxfun(@rdivide,exp(x),sum(exp(x),2))); 
    minmax = @(x)(bsxfun(@minus,x,max(x,[],2)));
    smax = @(u,a)(sum(bsxfun(@times,bsxfun(@rdivide,exp(a*u),sum(exp(a*u),2)),u),2));
    Ttr = bsxfun(@eq,Ytr,meshgrid(1:c,1:numel(Ytr)));
    LT = buildlt(Xtr,Ytr,c,params.distance);       
    % Dataset 
    params.n = n;                  % Size of the dataset 
    params.d  = d;          	   % Dimensionality of the dataset 
    params.c  = c;                 % Number of classes
    params.ni = ni;                % Number of patterns per class               
    params.Ttr = Ttr;              % Training binary targets 
    % Optimization problem 
    params.LT = LT;                % Linkage trees      
    % Stochastic gradient descend  
    params.softmax = softmax;      % Softmax function 
    params.minmax = minmax;        % Minmax function  
    params.alpha = 0.9;            % Alpha of Nesterv momentum of sgd 
    params.eta = 0.001;            % Learning rate of sgd 
    params.maxepoch = 1000;        % Max epochs of sgd 
    % Responses                    
    params.smax = smax;            % Smooth max function 
end

%--------------------------------
% Create linkage trees per class 
function LT = buildlt(Xtr,Ytr,c,distance)
    LT = cell(1,c);
    for i = 1:c
        Xi = Xtr(Ytr==i,:);
        Zi = linkage(Xi,'complete',distance);
        id = Zi(:,3) == 0;
        % Avoid zero cut-off
        Zi(id,3) = 0.1*min(Zi(~id,3)); 
        LT{i} = Zi;
    end
end
