% Selection with binary tournament for micro-GA
function selected = selection(bpop,fpop)
    np = size(bpop,1);
    ind = (1:np)';
    vs = [Shuffle(ind) Shuffle(ind)];  
    while sum(vs(:,1)==vs(:,2)) > 0
        vs = [Shuffle(ind) Shuffle(ind)];
    end
    winners = fpop(vs(:,1)) <= fpop(vs(:,2)); 
    index = [vs(winners,1);vs(~winners,2)];
    selected = bpop(index,:);
end 