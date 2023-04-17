% Two-point crossover for micro-GA
function bpop = crossover(bpop,params)
    childs = [];
    np = params.np; 
    pc = params.pc; 
    bitlength = params.bitlength; 
    for i = 1:np/2
        % Avoid two copies of the same string mating for the next generation
        [parent1,parent2] = avoidrepetition(bpop,params); 
        child1 = parent1;
        child2 = parent2;
        if rand < pc
            point1 = randi([1 bitlength-2],1);
            point2 = randi([point1+1 bitlength-1],1);
            child1(point1:point2) = parent2(point1:point2);
            child2(point1:point2) = parent1(point1:point2);
        end
        childs = cat(1,childs,child1,child2);
    end    
    bpop = childs; 
end 

%--------------------------------
% Functions 

%--------------------------------
% Avoid two copies of the same string mating for the next generation
function [parent1,parent2] = avoidrepetition(bpop,params) 
    np = params.np;
    ind = Shuffle(1:np);
    ind1 = ind(1);
    ind2 = ind(2);
    parent1 = bpop(ind1,:);
    parent2 = bpop(ind2,:);
    cont = 0; 
    while isequal(parent1,parent2) 
        ind = Shuffle(1:np);
        ind1 = ind(1);
        ind2 = ind(2);
        parent1 = bpop(ind1,:); 
        parent2 = bpop(ind2,:);
        cont = cont + 1; 
        if cont > 2*np 
            % Sometimes it is not posible for few individuals  
            break; 
        end
    end
end 