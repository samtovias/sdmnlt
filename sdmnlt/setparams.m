% Setup parameters of SDMN-LT micro-GA algorithm
function params = setparams(Xtr,Ytr,params)
    % Base
    n = numel(Ytr); 
    d = size(Xtr,2); 
    c = max(Ytr); 
    ni = accumarray(Ytr,ones(n,1),[c 1],@sum,0)';
    lmin = ones(1,c); 
    lmax = ni-1;
    beta = [params.beta(1) params.beta(2)];
    prec = [params.precision(1) params.precision(2)];
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
    params.lmin = lmin;            % Inferior linkage trees levels    
    params.lmax = lmax;            % Superior linkage trees levels 
    params.beta_min = beta(1);     % Inferior smoothness factor limit
    params.beta_max = beta(2);     % Superior smoothness factor limit 
    params.rmin = [lmin beta(1)];  % Inferior limits problem 
    params.rmax = [lmax beta(2)];  % Superior limits problem    
    for i = 1:c                    % Precision of the variables
        params.prec(i) = prec(1);  
    end                            
    params.prec(c+1) = prec(2);     
    % Genetic algorithm 
    nbin = c+1;             
    nbits = getnbits(nbin,params);          
    bitlength = sum(nbits); 
    params.nbin = nbin;            % Number of binary variables
    params.nbits = nbits;          % Bits per binary variable
    params.bitlength = bitlength;  % Length of the chromosome  
    % Stochastic gradient descend 
    params.softmax = softmax;      % Softmax function 
    params.minmax = minmax;        % Minmax function  
    % Responses 
    params.smax = smax;            % Smooth max function 
end

%--------------------------------
% Functions 

%--------------------------------
% Get number of bits per variable
function nbits = getnbits(nbin,params)
    nbits = zeros(1,nbin);
    for i = 1:nbin 
        nbits(i) = floor(log2((params.rmax(i)-params.rmin(i))*(10^params.prec(i))))+1;
    end
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
