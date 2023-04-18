% FIRST RUN THIS FILE TO LOAD ALL FUNCTIONS IN DIRECTORY 
clearvars; close all; clc;
addpath(fullfile(pwd,'C-functions'),...
        fullfile(pwd,'C-functions','Source-C-codes'));
addpath(genpath(fullfile(pwd,'datasets')));
addpath(fullfile(pwd,'normalization'));
addpath(fullfile(pwd,'performance_metrics'));
addpath(fullfile(pwd,'resampling'));
addpath(fullfile(pwd,'sdmnlt'));
