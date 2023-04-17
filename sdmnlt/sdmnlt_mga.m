% Learning smooth dendrite morphological neurons for pattern classification using linkage
% ... trees and evolutionary-based hyperparameter tuning (SDMN-LT)
% Evolutionary algorithm: Micro Genetic Algorithm (micro-GA, mga)
function [dendrite,out] = sdmnlt_mga(Xtr,Ytr,Xvd,Yvd,params,dataset,nt,threshold)    
    % Initialization of binary population  
    bpop = logical(randi([0 1],[params.np params.bitlength]));
    % Decode init population
    xpop = decode(bpop,params);
    % Evaluate objective function
    fpop = zeros(params.np,1);
    nd = zeros(params.np,1);
    parfor i=1:params.np
        x = xpop(i,:);
        [fpop(i),nd(i)] = objfun(Xtr,Ytr,Xvd,Yvd,params,x);
    end  
    fpop = round(fpop,4);
    % Current best solution  
    [fbest,ibest] = min(fpop);
    bbest = bpop(ibest,:); 
    xbest = xpop(ibest,:);
    ndbest = nd(ibest);
    % Curves
    curves = zeros(2,params.gen+1);
    curves(1,1) = fbest;
    curves(2,1) = mean(fpop);     
    % Generations 
    for g = 1:params.gen  
       % Take time of each generation
       tStart = tic;  
       % Selection  
       bpop = selection(bpop,fpop);
       % Crossover 
       bpop = crossover(bpop,params);
       % Decode current population
       xpop = decode(bpop,params);
       % Evaluate objective function
       fpop = zeros(params.np,1);
       nd = zeros(params.np,1);
       parfor i=1:params.np
           x = xpop(i,:);
           [fpop(i),nd(i)] = objfun(Xtr,Ytr,Xvd,Yvd,params,x);
       end  
       fpop = round(fpop,4);
       % Best solution of the offspring  
       [f,ind] = min(fpop);
       ibeta = xpop(ind,params.c+1);  
       % Update the best solution   
       if ((f < fbest) || ...
          ((f == fbest) && (nd(ind) < ndbest) && (ibeta <= xbest(params.c+1))) || ...
          ((f == fbest) && (nd(ind) <= ndbest) && (ibeta < xbest(params.c+1))))     
            fbest = f; 
            bbest = bpop(ind,:);
            xbest = xpop(ind,:);
            ndbest = nd(ind);
       else
            % Elitism  
            ind = randi([1,params.np],1,1);
            fpop(ind) = fbest; 
            bpop(ind,:) = bbest;
            xpop(ind,:) = xbest;
       end
       % Check for convergence of the population
       mh = meanhamming(bpop); 
       if mh < threshold 
            % Display announcement of convergence
            disp('Convergence!');
            % Reset population  
            bpop = logical(randi([0 1],[params.np params.bitlength]));
            % Decode reset population 
            xpop = decode(bpop,params);
            % Evaluate the objective function
            fpop = zeros(params.np,1);
            nd = zeros(params.np,1);
            parfor i=1:params.np
                x = xpop(i,:);
                [fpop(i),nd(i)] = objfun(Xtr,Ytr,Xvd,Yvd,params,x);
            end  
            fpop = round(fpop,4);
            % Best solution of the reset population   
            [f,ind] = min(fpop);
            ibeta = xpop(ind,params.c+1); 
            % Update the best solution   
           if ((f < fbest) || ...
              ((f == fbest) && (nd(ind) < ndbest) && (ibeta <= xbest(params.c+1))) || ...
              ((f == fbest) && (nd(ind) <= ndbest) && (ibeta < xbest(params.c+1))))     
                fbest = f; 
                bbest = bpop(ind,:);
                xbest = xpop(ind,:);
                ndbest = nd(ind);
           else
                % Elitism  
                ind = randi([1,params.np],1,1);
                fpop(ind) = fbest; 
                bpop(ind,:) = bbest;
                xpop(ind,:) = xbest;
           end
       end
       tEnd = toc(tStart);
       % Save curves
       curves(1,g+1) = fbest;
       curves(2,g+1) = mean(fpop);
       % Display status of the evolutionary optimization process 
       fprintf('SDMN-LT | gen: %d - %s - ntest: %d - fbest: %.4f - w: %.1f - time %.2f\n',...
                g,dataset,nt,fbest,params.w,tEnd); 
       % If the best solution has one dendrite per class, break the loop 
       if ndbest == params.c 
           break; 
       end
    end
    % Save final results
    out.bbest = bbest; 
    out.xbest = xbest; 
    out.bpop = bpop;
    out.xpop = xpop; 
    out.fpop = fpop;
    out.fbest = fbest; 
    out.curves = curves;
    % Train a SDMN-LT model with the best solution 
    dendrite = sdmnlt_train_mga(Xtr,Ytr,params,xbest);
    if isempty(dendrite)
        warning('Cannot create a DMN model with %d or more instances per dendrite\n',params.mpd);
    end
end 