% SDMN-LT training to be used in micro-GA
function dendrite = sdmnlt_train_mga(X,Y,params,x)
    % Get params
    LT = params.LT; 
    distance = params.distance; 
    smax = params.smax; 
    T = params.Ttr; 
    dendrite = struct([]);
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
            if size(Xj,1) >= params.mpd
                Ct(j,:) = mean(Xj,1); % Compute centroid
            else
                flag(j) = true;
                continue;
            end
        end
        if sum(flag==1) == nd(i)
            dendrite = []; 
            return;
        end
        Ct(flag,:) = [];
        nd(i) = size(Ct,1);
        dendrite(i).C = Ct;
        cts = cat(1,cts,Ct);
        mrk = cat(1,mrk,i*ones(nd(i),1));
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
        D = bsxfun(@minus,dendrite(i).radii,pdist2(X,dendrite(i).C,distance));
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