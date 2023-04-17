% Decode binary population of micro-GA
function xpop = decode(bpop,params)
    np = params.np; 
    rmin = params.rmin; 
    rmax = params.rmax;
    nbits = params.nbits;
    nbin = params.nbin;
    xpop = zeros(np,numel(nbits));
    for i=1:np
        stop = 0;
        for j=1:nbin
            start = stop+1; stop = start + nbits(j) - 1;
            total = sum(bpop(i,start:stop) .* 2.^(nbits(j)-1 : -1: 0));
            xpop(i,j) = rmin(j) + (rmax(j)-rmin(j)) / (2^nbits(j)-1) * total;
        end
    end
    xpop = round(xpop);
end