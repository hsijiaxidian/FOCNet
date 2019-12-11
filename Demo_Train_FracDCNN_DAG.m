
clear;
rng('default')
addpath('utilities');
%-------------------------------------------------------------------------
% Configuration
%-------------------------------------------------------------------------
opts.learningRate     = [logspace(-3,-3,45) logspace(-3.5,-4,45)];% you can change the learning rate
opts.batchSize        = 8; % 
opts.gpus             = [1]; 
opts.numSubBatches    = 1;

% solver
opts.solver           = 'Adam'; % global
opts.derOutputs       = {'objective',1} ;

opts.backPropDepth    = Inf;
%-------------------------------------------------------------------------
%   Initialize model
%-------------------------------------------------------------------------
global CurTask;
CurTask = 'Denoising'; %% 'Deblocking' and 'SISR'
opts.sigma = 50;
% CurTask = 'SISR'; opts.sigma = 2;
opts.modelName        = ['FracDCNN' CurTask num2str(opts.sigma)];% model name
% net  = feval(['FracDCNN','_Init']);
net  = feval(['FracDCNN']);

%-------------------------------------------------------------------------
%   Train
%-------------------------------------------------------------------------

[net] = FracDCNN_train_dag(net,  ...
    'learningRate',opts.learningRate, ...
    'derOutputs',opts.derOutputs, ...
    'numSubBatches',opts.numSubBatches, ...
    'backPropDepth',opts.backPropDepth, ...
    'solver',opts.solver, ...
    'batchSize', opts.batchSize, ...
    'modelname', opts.modelName, ...
    'sigma', opts.sigma, ...
    'gpus',opts.gpus) ;






