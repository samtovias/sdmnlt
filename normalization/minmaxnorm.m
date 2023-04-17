% Minmax normalization 1    
function [vect,dmn,dmx] = minmaxnorm(X,stats)
    if nargin == 1
        dmx = max(X,[],1);
        dmn = min(X,[],1);
    elseif nargin == 2
        dmn = stats(1,:);
        dmx = stats(2,:);
    end
    N = size(X,1);
    ind = dmx==dmn;
    mx = repmat(dmx,N,1);
    mn = repmat(dmn,N,1);
    vect = (2.*(X - mn)./((mx-mn)+1e-6))-1; % [-1,1]
    % vect = (X - mn)./(mx-mn); % [0,1]
    % Avoid NaN problem when variables have no range
    vect(:,ind) = 0;   
end 