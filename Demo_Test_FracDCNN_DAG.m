clear; clc;

%% testing set
addpath(fullfile('utilities'));

folderModel = 'model';
folderTest  = 'testsets';
folderResult= 'results';
imageSets   = {'BSD68','Set12','Set14','Urban100'}; % testing datasets
setTestCur  = imageSets{2};      % current testing dataset


showresult  = 1;
gpu         = 1;


noiseSigma  = 50;
CurTask = 'Denoising';
% load model
epoch       = 30;

modelName   = ['FracDCNN' CurTask num2str(noiseSigma)];

% case one: for the model in 'data/model'
%load(fullfile('data',folderModel,[modelName,'-epoch-',num2str(epoch),'.mat']));

% case two: for the model in 'utilities'
load(fullfile('data\FracDCNNDenoising50\',[modelName,'-epoch-',num2str(epoch),'.mat']));




net = dagnn.DagNN.loadobj(net) ;

net.removeLayer('loss') ;
out1 = net.getVarIndex('prediction') ;
net.vars(net.getVarIndex('prediction')).precious = 1 ;

net.mode = 'test';

if gpu
    net.move('gpu');
end

% read images
ext         =  {'*.jpg','*.png','*.bmp'};
filePaths   =  [];
for i = 1 : length(ext)
    filePaths = cat(1,filePaths, dir(fullfile(folderTest,setTestCur,ext{i})));
end

folderResultCur       =  fullfile(folderResult, [setTestCur,'_',int2str(noiseSigma)]);
if ~isdir(folderResultCur)
    mkdir(folderResultCur)
end


% PSNR and SSIM
PSNRs = zeros(1,length(filePaths));
SSIMs = zeros(1,length(filePaths));


for i = 1 : 2*length(filePaths)
    
    % read image
    ii = ceil(i/2);
    if mod(i,2) == 1
        label = zeros(128);
    else
        label = imread(fullfile(folderTest,setTestCur,filePaths(ii).name));
    end
    [~,nameCur,extCur] = fileparts(filePaths(ii).name);
    [w,h,c]=size(label);
    if c==3
        label = rgb2gray(label);
    end
    % pad image to correlated with the down sample
    label = modcrop(label,8);
    
    % add additive Gaussian noise
    randn('seed',0);
    noise = noiseSigma/255.*randn(size(label));
    input = im2single(label) + single(noise);
    
    if gpu
        input = gpuArray(input);
    end
    tic
    net.eval({'input', input}) ;
    toc
    % output (single)
    output = gather(squeeze(gather(net.vars(out1).value)));

    
    
    % calculate PSNR and SSIM
    [PSNRCur, SSIMCur] = Cal_PSNRSSIM(label,im2uint8(output),0,0);
    if showresult
        imshow(cat(2,im2uint8(input),im2uint8(label),im2uint8(output)));
        title([filePaths(ii).name,'    ',num2str(PSNRCur,'%2.2f'),'dB','    ',num2str(SSIMCur,'%2.4f')])
        imwrite(im2uint8(output), fullfile(folderResultCur, [nameCur, '_' int2str(noiseSigma),'_PSNR_',num2str(PSNRCur*100,'%4.0f'), extCur] ));
        drawnow;
       pause(1)
    end
    PSNRs(ii) = PSNRCur;
    SSIMs(ii) = SSIMCur;
end


disp([mean(PSNRs),mean(SSIMs)]);




