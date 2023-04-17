% Mean Hamming distance for check convergence in micro-GA
function mh = meanhamming(bpop)
    bpop2 = double(bpop);
    np = size(bpop,1); 
    sumhamming = sum(sum(pdist2(bpop2,bpop2,'hamming'))) / 2 ;
    mh =  sumhamming / (np*(np-1)/2);
end 