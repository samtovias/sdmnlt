% Examples with synthetic datasets

% Learning smooth dendrite morphological neurons for pattern classification using linkage
% ... trees and evolutionary-based hyperparameter tuning (SDMN-LT)
% Evolutionary algorithm: Micro Genetic Algorithm (micro-GA, mga)
clearvars; clc; close all; 

% Method name 
method = {'SDMN-LT with micro-GA'};

% Path datasets 
path_datasets = strcat('datasets/synthetic/');

% Datasets
dataset = 'Concentric.mat';
% dataset = 'Horseshoes.mat';
% dataset = 'Moons.mat';
% dataset = 'Ripley_Dataset.mat';
% dataset = 'Three_Gaussians.mat';
% dataset = 'Three_Spirals.mat';
% dataset = 'Two_Gaussians.mat';
% dataset = 'Two_Spirals.mat';
% dataset = 'XOR_Problem.mat';

% Load dataset
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
paramsc.gen = 100;           % Number of generations 
paramsc.w = 0.5;             % Weight of the objective function 
paramsc.distance = distance; % Distance function 
paramsc.beta = [1 30];       % Smoothness factor [0.1 30] 
paramsc.precision = [0 0];   % Precision of the cutoff levels and beta
paramsc.alpha = 0.9;         % Alpha of Nesterv momentum of sgd 
paramsc.eta = 0.001;         % Learning rate of sgd 
paramsc.maxepoch = 500;      % Max epochs of sgd 

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

% Plot SDMN-LT dendrites and decision regions 
sdmnlt_plot(Xtt,Ytt,params,sdmn);