% Run experiments of SDMN-LT model

% Learning smooth dendrite morphological neurons for pattern classification using linkage
% ... trees and evolutionary-based hyperparameter tuning (SDMN-LT)
% Evolutionary algorithm: Micro Genetic Algorithm (micro-GA, mga)
clearvars; clc; close all; 

% Method name 
method = {'SDMN-LT with micro-GA'};

% Datasets:
sets = {'imbalanced','large','noisy','real'};

% Experiments for each dataset type
for ii = 1:numel(sets)
    
    % Datasets type
    datasets = sets{ii};
    
    % Paths 
    % Path datasets 
    path_datasets = strcat('datasets/',datasets,'/');
    % Save results
    results = strcat('results/sdmnlt/',datasets); 
    if ~isfolder(results); mkdir(results); end
    
    % Get names of the datasets 
    names = dir(fullfile(path_datasets));
    num = numel(names)-2; 
    datasets_names = cell(num,1);
    for i = 1:num
        datasets_names{i} = names(i+2).name;
    end
    
    % For each dataset 
    for i = 1:num
        
        % Load dataset
        dataset = datasets_names{i}; 
        dataset_file = fullfile(path_datasets,dataset);
        data = load(dataset_file);    
        Xtr = data.Xtr;
        Ytr = data.Ytr; 
        Xtt = data.Xtt; 
        Ytt = data.Ytt; 
        
        % Dataset information 
        n = size(Xtr,2);
        d = size(Xtr,1);
        c = max(Ytr); 
        
        % Training and test sets 
        X = Xtr';
        Y = Ytr';
        Xtt = Xtt';
        Ytt = Ytt';
        
        % Minmax normalization in range [-1,1]
        [X,mn,mx] = minmaxnorm(X);
        Xtt = minmaxnorm(Xtt,[mn;mx]);
        
        % Training and validation sets
        [Xtr,Ytr,Xvd,Yvd] = split_data(X,Y,0.2);
    
        % Distance for linkage tree and SDMN-LT training
        distance = 'squaredeuclidean'; 
    
        % Setting preliminar parameters 
        paramsc.gen = 1000;          % Number of generations 
        paramsc.w = 0.5;             % Weight of the objective function 
        paramsc.distance = distance; % Distance function 
        paramsc.beta = [1 30];       % Smoothness factor [0.1 30] 
        paramsc.precision = [0 0];   % Precision of the cutoff levels and beta
        paramsc.alpha = 0.9;         % Alpha of Nesterv momentum of sgd 
        paramsc.eta = 0.001;         % Learning rate of sgd 
        paramsc.maxepoch = 2000;     % Max epochs of sgd 
        
        paramsc.np = 4;              % Population size
        paramsc.pc = 1.0;            % Crossover probability 
        threshold = 0.05;            % For convergence 
        paramsc.mpd = 3;             % Minimum number of instance per dendrite

        % Setting parameters 
        params = setparams(Xtr,Ytr,paramsc);
        
        % SDMN-LT hyperparameters tunning with micro genetic algorithm 
        sdmn = []; 
        while isempty(sdmn)
            tic;
            [sdmn,out] = sdmnlt_mga(Xtr,Ytr,Xvd,Yvd,params,dataset,1,threshold);    
            tunning_time = toc; 
            if params.mpd > 1 
                params.mpd = params.mpd - 1;
            end
        end
        
        % Get results 
        [Yp1,Pr1] = sdmnlt_predict(X,params,sdmn);   % Predictions on the training set
        [Yp2,Pr2] = sdmnlt_predict(Xtt,params,sdmn); % Predictions on the test set
        hpars = out.xbest;                           % Hyperparameters (best solution decoded)
        bf = out.fbest;                              % Fitness of the best solution 
        model = sdmn;                                % Save SDMN-LT model   
        Yptr = Yp1;                                  % Predicted labels of the training set   
        Yptt = Yp2;                                  % Predicted labels of the test set      
        nd = sum(cat(1,sdmn.number));                % Number of dendrites                              
        weights = numel(cat(1,sdmn.weights));        % Number of weights 
        beta = 1;                                    % One beta parameter    
        np = (d + 1)*nd + weights + beta;            % Number of learnable parameters for spherical dendrites 
        mc = (np/2) * log2(n);                       % Model complexity    

        % Get performance metrics: Accuracy, Matthews Correlation Coefficient and F1-Score
        [ACCtr,~,MCCtr,~,~,~,F1tr,~,~] = mulclassperf(Y',Yptr,c);   
        [ACCtt,~,MCCtt,~,~,~,F1tt,~,~] = mulclassperf(Ytt',Yptt,c);   
        
        % Compute training time of the SDMN-LT model
        tic;
        sdmnlt_train(Xtr,Ytr,hpars); 
        training_time = toc; 
    
        % Display status
        str1 = strcat('Dataset ',{' '},num2str(i),' - ',{' '},'Method',{' '},method{1});
        fprintf('%s\n',str1{1});
            
        % Save results  
        file = fullfile(results,dataset);
        save(file,'dataset','method','model','np','mc','tunning_time','training_time','hpars',...
                  'out','Yptr','Yptt','ACCtr','ACCtt','MCCtr','MCCtt','F1tr','F1tt');
    
    end
    
end 