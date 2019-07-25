function net = FracDCNN()

% Create DAGNN object
net = dagnn.DagNN();

% conv + relu
blockNum = 1;
inVar = 'input';
stride = [1,1]; 
lr     = [1,0];

%% start
% Initialize x,y,z,s
% original scale
[net, inVar, blockNum] = addConv(net, blockNum, inVar, [3,3,1,128], [1,1], stride, [1,1]);
[net, inVar, blockNum] = addReLU(net, blockNum, inVar);
[net, inVar, blockNum] = addConv(net, blockNum, inVar, [3,3,128,128], [1,1], stride, lr);  % Conv
[net, inVar, blockNum] = addBnorm(net, blockNum, inVar, 128);
[net, inVar_x1, blockNum] = addReLU(net, blockNum, inVar); % initialize x1

% down 1-mid  1/2 scale
[net, inVar, blockNum] = addPooling(net, blockNum, inVar_x1);
[net, inVar, blockNum] = addConv(net, blockNum, inVar, [3,3,128,128], [1,1], stride, lr);  % Conv
[net, inVar, blockNum] = addBnorm(net, blockNum, inVar, 128);
[net, inVar_y1, blockNum] = addReLU(net, blockNum, inVar); % initialize y1

% down 1-bottom 1/4 scale
[net, inVar, blockNum] = addPooling(net, blockNum, inVar_y1);
[net, inVar, blockNum] = addConv(net, blockNum, inVar, [3,3,128,128], [1,1], stride, lr);  % Conv
[net, inVar, blockNum] = addBnorm(net, blockNum, inVar, 128);
[net, inVar_z1, blockNum] = addReLU(net, blockNum, inVar); % initialize z1

% down 1-bottom 1/8 scale
[net, inVar, blockNum] = addPooling(net, blockNum, inVar_z1);
[net, inVar, blockNum] = addConv(net, blockNum, inVar, [3,3,128,128], [1,1], stride, lr);  % Conv
[net, inVar, blockNum] = addBnorm(net, blockNum, inVar, 128);
[net, inVar_s1, blockNum] = addReLU(net, blockNum, inVar); % initialize s1

% up 1-mid 1/4 scale
[net, inVar, blockNum] = addConvt(net, blockNum, inVar_s1, [2,2,128,128],2, lr);%upsampling
[net, inVar, blockNum] = addGate(net,blockNum,{inVar_z1, inVar},lr);
[net, inVar, blockNum] = addConv(net, blockNum, inVar, [3,3,128,128], [1,1], stride, lr);  % Conv
[net, inVar, blockNum] = addBnorm(net, blockNum, inVar, 128);
[net, inVar, blockNum] = addReLU(net, blockNum, inVar);
[net, inVar_z2, blockNum] = addPLM(net,blockNum,{inVar_z1, inVar});% --------- z1->z2;

% up 1-top 1/2 scale
[net, inVar, blockNum] = addConvt(net, blockNum, inVar_z2, [2,2,128,128],2, lr);%upsampling
[net, inVar, blockNum] = addGate(net,blockNum,{inVar_y1, inVar},lr);
[net, inVar, blockNum] = addConv(net, blockNum, inVar, [3,3,128,128], [1,1], stride, lr);  % Conv
[net, inVar, blockNum] = addBnorm(net, blockNum, inVar, 128);
[net, inVar, blockNum] = addReLU(net, blockNum, inVar);
[net, inVar_y2, blockNum] = addPLM(net,blockNum,{inVar_y1, inVar});% --------- y1->y2

% up 1-top original scale
[net, inVar, blockNum] = addConvt(net, blockNum, inVar_y2, [2,2,128,128],2, lr);%upsampling
[net, inVar, blockNum] = addGate(net,blockNum,{inVar_x1, inVar},lr);
[net, inVar, blockNum] = addConv(net, blockNum, inVar, [3,3,128,128], [1,1], stride, lr);  % Conv
[net, inVar, blockNum] = addBnorm(net, blockNum, inVar, 128);
[net, inVar, blockNum] = addReLU(net, blockNum, inVar);
[net, inVar_x2, blockNum] = addPLM(net,blockNum,{inVar_x1, inVar});% --------- x1->x2

% down 2-mid 1/2 scale
[net, inVar, blockNum] = addPooling(net, blockNum, inVar_x2);
[net, inVar, blockNum] = addGate(net,blockNum,{inVar_y2,inVar},lr);
[net, inVar, blockNum] = addConv(net, blockNum, inVar, [3,3,128,128], [1,1], stride, lr);  % Conv
[net, inVar, blockNum] = addBnorm(net, blockNum, inVar, 128);
[net, inVar, blockNum] = addReLU(net, blockNum, inVar);
[net, inVar_y3, blockNum] = addPLM(net,blockNum,{inVar_y1,inVar_y2,inVar});% ------ y1->y2->y3

% down 2-bottom 1/4 scale
[net, inVar, blockNum] = addPooling(net, blockNum, inVar_y3);
[net, inVar, blockNum] = addGate(net,blockNum,{inVar_z2,inVar},lr);
[net, inVar, blockNum] = addConv(net, blockNum, inVar, [3,3,128,128], [1,1], stride, lr);  % Conv
[net, inVar, blockNum] = addBnorm(net, blockNum, inVar, 128);
[net, inVar, blockNum] = addReLU(net, blockNum, inVar);
[net, inVar_z3, blockNum] = addPLM(net,blockNum,{inVar_z1, inVar_z2,inVar});% ------ z1->z2->z3

% down 2-bottom 1/8 scale
[net, inVar, blockNum] = addPooling(net, blockNum, inVar_z3);
[net, inVar, blockNum] = addGate(net,blockNum,{inVar_s1, inVar},lr);
[net, inVar, blockNum] = addConv(net, blockNum, inVar, [3,3,128,128], [1,1], stride, lr);  % Conv
[net, inVar, blockNum] = addBnorm(net, blockNum, inVar, 128);
[net, inVar, blockNum] = addReLU(net, blockNum, inVar);
[net, inVar_s2, blockNum] = addPLM(net,blockNum,{inVar_s1, inVar}); % -------- s1->s2

% up 2-mid 1/4 scale
[net, inVar, blockNum] = addConvt(net, blockNum, inVar_s2, [2,2,128,128], 2, lr);% upsampling
[net, inVar, blockNum] = addGate(net,blockNum,{inVar_z3, inVar},lr);
[net, inVar, blockNum] = addConv(net, blockNum, inVar, [3,3,128,128], [1,1], stride, lr);  % Conv
[net, inVar, blockNum] = addBnorm(net, blockNum, inVar, 128);
[net, inVar, blockNum] = addReLU(net, blockNum, inVar);
[net, inVar_z4, blockNum] = addPLM(net,blockNum,{inVar_z1,inVar_z2, inVar_z3, inVar});% ----- z1->z2->z3->z4

% up 2-mid 1/2 scale
[net, inVar, blockNum] = addConvt(net, blockNum, inVar_z4, [2,2,128,128], 2, lr);% upsampling
[net, inVar, blockNum] = addGate(net,blockNum,{inVar_y3, inVar},lr);
[net, inVar, blockNum] = addConv(net, blockNum, inVar, [3,3,128,128], [1,1], stride, lr);  % Conv
[net, inVar, blockNum] = addBnorm(net, blockNum, inVar, 128);
[net, inVar, blockNum] = addReLU(net, blockNum, inVar);
[net, inVar_y4, blockNum] = addPLM(net,blockNum,{inVar_y1,inVar_y2, inVar_y3, inVar});%  ------ y1->y2->y3->y4

% up 2-top scale
[net, inVar, blockNum] = addConvt(net, blockNum, inVar_y4, [2,2,128,128], 2, lr);% upsampling
[net, inVar, blockNum] = addGate(net,blockNum,{inVar_x2,inVar},lr);
[net, inVar, blockNum] = addConv(net, blockNum, inVar, [3,3,128,128], [1,1], stride, lr);
[net, inVar, blockNum] = addBnorm(net, blockNum, inVar, 128);
[net, inVar, blockNum] = addReLU(net, blockNum, inVar);
[net, inVar_x3, blockNum] = addPLM(net,blockNum,{inVar_x1,inVar_x2,inVar}); % ------ x1->x2->x3

% down 3-mid 1/2 scale
[net, inVar, blockNum] = addPooling(net, blockNum, inVar_x3);
[net, inVar, blockNum] = addGate(net,blockNum,{inVar_y4,inVar},lr);
[net, inVar, blockNum] = addConv(net, blockNum, inVar, [3,3,128,128], [1,1], stride, lr);
[net, inVar, blockNum] = addBnorm(net, blockNum, inVar, 128);
[net, inVar, blockNum] = addReLU(net, blockNum, inVar);
[net, inVar_y5, blockNum] = addPLM(net,blockNum,{inVar_y1,inVar_y2, inVar_y3, inVar_y4,inVar});% y1--->y5

% down 3-bottom 1/4 scale
[net, inVar, blockNum] = addPooling(net, blockNum, inVar_y5);
[net, inVar, blockNum] = addGate(net,blockNum,{inVar_z4,inVar},lr);
[net, inVar, blockNum] = addConv(net, blockNum, inVar, [3,3,128,128], [1,1], stride, lr);  % Conv
[net, inVar, blockNum] = addBnorm(net, blockNum, inVar, 128);
[net, inVar, blockNum] = addReLU(net, blockNum, inVar);
[net, inVar_z5, blockNum] = addPLM(net,blockNum,{inVar_z1,inVar_z2, inVar_z3, inVar_z4, inVar});% z1---->z5

% down 3-bottom 1/8 scale
[net, inVar, blockNum] = addPooling(net, blockNum, inVar_z5);
[net, inVar, blockNum] = addGate(net,blockNum,{inVar_s2,inVar},lr);
[net, inVar, blockNum] = addConv(net, blockNum, inVar, [3,3,128,128], [1,1], stride, lr);  % Conv
[net, inVar, blockNum] = addBnorm(net, blockNum, inVar, 128);
[net, inVar, blockNum] = addReLU(net, blockNum, inVar);
[net, inVar_s3, blockNum] = addPLM(net,blockNum,{inVar_s1,inVar_s2,inVar});% s1--->s3

% up 3-bottom 1/4 scale
[net, inVar, blockNum] = addConvt(net, blockNum, inVar_s3, [2,2,128,128], 2,lr);%upsampling
[net, inVar, blockNum] = addGate(net,blockNum,{inVar_z5,inVar},lr);
[net, inVar, blockNum] = addConv(net, blockNum, inVar, [3,3,128,128], [1,1], stride, lr);  % Conv
[net, inVar, blockNum] = addBnorm(net, blockNum, inVar, 128);
[net, inVar, blockNum] = addReLU(net, blockNum, inVar);
[net, inVar_z6, blockNum] = addPLM(net,blockNum,{inVar_z1,inVar_z2, inVar_z3,inVar_z4,inVar_z5, inVar});% z1--->z6

% up 3-mid 1/2 scale
[net, inVar, blockNum] = addConvt(net, blockNum, inVar_z6, [2,2,128,128],2,lr);%upsampling
[net, inVar, blockNum] = addGate(net,blockNum,{inVar_y5, inVar},lr);
[net, inVar, blockNum] = addConv(net, blockNum, inVar, [3,3,128,128], [1,1], stride, lr);  % Conv
[net, inVar, blockNum] = addBnorm(net, blockNum, inVar, 128);
[net, inVar, blockNum] = addReLU(net, blockNum, inVar);
[net, inVar_y6, blockNum] = addPLM(net,blockNum,{inVar_y1,inVar_y2,inVar_y3, inVar_y4, inVar_y5, inVar});% y1--->y6

% up 3-top scale
[net, inVar, blockNum] = addConvt(net, blockNum, inVar_y6, [2,2,128,128],2,lr);%upsampling
[net, inVar, blockNum] = addGate(net,blockNum,{inVar_x3,inVar},lr);
[net, inVar, blockNum] = addConv(net, blockNum, inVar, [3,3,128,128], [1,1], stride, lr);
[net, inVar, blockNum] = addBnorm(net, blockNum, inVar, 128);
[net, inVar, blockNum] = addReLU(net, blockNum, inVar);
[net, inVar_x4, blockNum] = addPLM(net,blockNum,{inVar_x1,inVar_x2,inVar_x3,inVar}); % x1--->x4

% down to 4-mid 1/2 scale
[net, inVar, blockNum] = addPooling(net, blockNum, inVar_x4);
[net, inVar, blockNum] = addGate(net,blockNum,{inVar_y6,inVar},lr);
[net, inVar, blockNum] = addConv(net, blockNum, inVar, [3,3,128,128], [1,1], stride, lr);  % Conv
[net, inVar, blockNum] = addBnorm(net, blockNum, inVar, 128);
[net, inVar, blockNum] = addReLU(net, blockNum, inVar);
[net, inVar_y7, blockNum] = addPLM(net,blockNum,{inVar_y1,inVar_y2,inVar_y3, inVar_y4,inVar_y5,inVar_y6,inVar});% y1--->y7

% down 4-bottom 1/4 scale
[net, inVar, blockNum] = addPooling(net, blockNum, inVar_y7);
[net, inVar, blockNum] = addGate(net,blockNum,{inVar_z6,inVar},lr);
[net, inVar, blockNum] = addConv(net, blockNum, inVar, [3,3,128,128], [1,1], stride, lr);  % Conv
[net, inVar, blockNum] = addBnorm(net, blockNum, inVar, 128);
[net, inVar, blockNum] = addReLU(net, blockNum, inVar);
[net, inVar_z7, blockNum] = addPLM(net,blockNum,{inVar_z1,inVar_z2, inVar_z3, inVar_z4,inVar_z5, inVar_z6,inVar});% z1--->z7

% down 4-bottom 1/8 scale
[net, inVar, blockNum] = addPooling(net, blockNum, inVar_z7);
[net, inVar, blockNum] = addGate(net,blockNum,{inVar_s3,inVar},lr);
[net, inVar, blockNum] = addConv(net, blockNum, inVar, [3,3,128,128], [1,1], stride, lr);  % Conv
[net, inVar, blockNum] = addBnorm(net, blockNum, inVar, 128);
[net, inVar, blockNum] = addReLU(net, blockNum, inVar);
[net, inVar_s4, blockNum] = addPLM(net,blockNum,{inVar_s1,inVar_s2,inVar_s3,inVar});% s1--->s4

% up 4-bottom 1/4 scale
[net, inVar, blockNum] = addConvt(net, blockNum, inVar_s4, [2,2,128,128], 2,lr);%upsampling
[net, inVar, blockNum] = addGate(net,blockNum,{inVar_z7,inVar},lr);
[net, inVar, blockNum] = addConv(net, blockNum, inVar, [3,3,128,128], [1,1], stride, lr);  % Conv
[net, inVar, blockNum] = addBnorm(net, blockNum, inVar, 128);
[net, inVar, blockNum] = addReLU(net, blockNum, inVar);
[net, inVar_z8, blockNum] = addPLM(net,blockNum,{inVar_z1,inVar_z2, inVar_z3,inVar_z4,inVar_z5,inVar_z6,inVar_z7,inVar});% z1--->z8

% up 4-mid 1/2 scale
[net, inVar, blockNum] = addConvt(net, blockNum, inVar_z8, [2,2,128,128],2,lr);%upsampling
[net, inVar, blockNum] = addGate(net,blockNum,{inVar_y7,inVar},lr);
[net, inVar, blockNum] = addConv(net, blockNum, inVar, [3,3,128,128], [1,1], stride, lr);  % Conv
[net, inVar, blockNum] = addBnorm(net, blockNum, inVar, 128);
[net, inVar, blockNum] = addReLU(net, blockNum, inVar);
[net, inVar_y8, blockNum] = addPLM(net,blockNum,{inVar_y1,inVar_y2,inVar_y3,inVar_y4,inVar_y5,inVar_y6,inVar_y7,inVar});% y1--->y8

% up 4-top scale
[net, inVar, blockNum] = addConvt(net, blockNum, inVar_y8, [2,2,128,128],2,lr);%upsampling
[net, inVar, blockNum] = addGate(net,blockNum,{inVar_x4,inVar},lr);
[net, inVar, blockNum] = addConv(net, blockNum, inVar, [3,3,128,128], [1,1], stride, lr);
[net, inVar, blockNum] = addBnorm(net, blockNum, inVar, 128);
[net, inVar, blockNum] = addReLU(net, blockNum, inVar);
[net, inVar, blockNum] = addPLM(net,blockNum,{inVar_x1,inVar_x2,inVar_x3, inVar_x4,inVar});% x1--->x5

[net, inVar, blockNum] = addConv(net, blockNum, inVar, [3,3,128,128], [1,1], stride, lr);
[net, inVar, blockNum] = addBnorm(net, blockNum, inVar, 128);
[net, inVar, blockNum] = addReLU(net, blockNum, inVar);
[net, inVar, blockNum] = addConv(net, blockNum, inVar, [3,3,128,1], [1,1], stride, lr);

% sum
inVar = {'input',inVar};
[net, inVar, blockNum] = addSum(net,blockNum,inVar);
% [net, inVar, blockNum] = addConv(net, blockNum, inVar, [1,1,2,1],[0,0],stride,lr);
outputName = 'prediction';
net.renameVar(inVar,outputName)

% loss
net.addLayer('loss', dagnn.Loss('loss','L2'), {'prediction','label'}, {'objective'},{});
net.vars(net.getVarIndex('prediction')).precious = 1;


end




% Add a Concat layer
function [net, inVar, blockNum] = addConcat(net, blockNum, inVar)

outVar   = sprintf('concat%d', blockNum);
layerCur = sprintf('concat%d', blockNum);

block = dagnn.Concat('dim',3);
net.addLayer(layerCur, block, inVar, {outVar},{});

inVar = outVar;
blockNum = blockNum + 1;
end

% Add a pooling layer
function [net, inVar, blockNum] = addPooling(net, blockNum, inVar)

outVar   = sprintf('concat%d', blockNum);
layerCur = sprintf('concat%d', blockNum);

block = dagnn.Pooling('poolSize',[2,2], 'stride', 2);
net.addLayer(layerCur, block, inVar, {outVar},{});

inVar = outVar;
blockNum = blockNum + 1;
end

% Add a loss layer
function [net, inVar, blockNum] = addLoss(net, blockNum, inVar)

outVar   = 'objective';
layerCur = sprintf('loss%d', blockNum);

block    = dagnn.Loss('loss','L2');
net.addLayer(layerCur, block, inVar, {outVar},{});

inVar    = outVar;
blockNum = blockNum + 1;
end


% Add a sum layer
function [net, inVar, blockNum] = addSum(net, blockNum, inVar)

outVar   = sprintf('sum%d', blockNum);
layerCur = sprintf('sum%d', blockNum);

block    = dagnn.Sum();
net.addLayer(layerCur, block, inVar, {outVar},{});

inVar    = outVar;
blockNum = blockNum + 1;
end


function [net, inVar, blockNum] = addPLM(net, blockNum, inVar)

outVar   = sprintf('plm%d', blockNum);
layerCur = sprintf('plm%d', blockNum);

block    = dagnn.PLM();
net.addLayer(layerCur, block, inVar, {outVar},{});

inVar    = outVar;
blockNum = blockNum + 1;
end

% Add a relu layer
function [net, inVar, blockNum] = addReLU(net, blockNum, inVar)

outVar   = sprintf('relu%d', blockNum);
layerCur = sprintf('relu%d', blockNum);

block    = dagnn.ReLU('leak',0);
net.addLayer(layerCur, block, {inVar}, {outVar},{});

inVar    = outVar;
blockNum = blockNum + 1;
end

function [net, inVar, blockNum] = addWsum(net, blockNum, inVar,lr)

outVar   = sprintf('wsum%d', blockNum);
layerCur = sprintf('wsum%d', blockNum);
trainMethod = 'adam';

block    = dagnn.Wsum();
params = {[layerCur '_weight']};
net.addLayer(layerCur, block, inVar, {outVar}, params);

weight  = net.getParamIndex({[layerCur '_weight']});

sc = ones(1,numel(inVar),'single') ; %improved Xavier
net.params(weight).value        = sc;
net.params(weight).learningRate = lr(1);
net.params(weight).weightDecay  = 1;
net.params(weight).trainMethod  = trainMethod;

inVar    = outVar;
blockNum = blockNum + 1;
end

function [net, inVar, blockNum] = addGate(net, blockNum, inVar,lr)

outVar   = sprintf('gate%d', blockNum);
layerCur = sprintf('gate%d', blockNum);
trainMethod = 'adam';

block    = dagnn.Gate();
params = {[layerCur '_weight']};
net.addLayer(layerCur, block, inVar, {outVar},params);

weight  = net.getParamIndex({[layerCur '_weight']});

sc = single(1) ; %improved Xavier
net.params(weight).value        = sc;
net.params(weight).learningRate = lr(1);
net.params(weight).weightDecay  = 1;
net.params(weight).trainMethod  = trainMethod;

inVar    = outVar;
blockNum = blockNum + 1;
end

% Add a bnorm layer
function [net, inVar, blockNum] = addBnorm(net, blockNum, inVar, n_ch)

trainMethod = 'adam';
outVar   = sprintf('bnorm%d', blockNum);
layerCur = sprintf('bnorm%d', blockNum);

params={[layerCur '_g'], [layerCur '_b'], [layerCur '_m']};
net.addLayer(layerCur, dagnn.BatchNorm('numChannels', n_ch), {inVar}, {outVar},params) ;

pidx = net.getParamIndex({[layerCur '_g'], [layerCur '_b'], [layerCur '_m']});
b_min                           = 0.025;
net.params(pidx(1)).value       = clipping(sqrt(2/(9*n_ch))*randn(n_ch,1,'single'),b_min);
net.params(pidx(1)).learningRate= 1;
net.params(pidx(1)).weightDecay = 0;
net.params(pidx(1)).trainMethod = trainMethod;

net.params(pidx(2)).value       = zeros(n_ch, 1, 'single');
net.params(pidx(2)).learningRate= 1;
net.params(pidx(2)).weightDecay = 0;
net.params(pidx(2)).trainMethod = trainMethod;

net.params(pidx(3)).value       = [zeros(n_ch,1,'single'), 0.01*ones(n_ch,1,'single')];
net.params(pidx(3)).learningRate= 1;
net.params(pidx(3)).weightDecay = 0;
net.params(pidx(3)).trainMethod = 'average';

inVar    = outVar;
blockNum = blockNum + 1;
end


% add a ConvTranspose layer
function [net, inVar, blockNum] = addConvt(net, blockNum, inVar, dims, upsample, lr)
opts.cudnnWorkspaceLimit = 1024*1024*1024*4; % 2GB
convOpts = {'CudnnWorkspaceLimit', opts.cudnnWorkspaceLimit} ;
trainMethod = 'adam';

outVar      = sprintf('convt%d', blockNum);

layerCur    = sprintf('convt%d', blockNum);

convBlock = dagnn.ConvTranspose('size', dims,'upsample', upsample, ...
    'hasBias', true, 'opts', convOpts);

net.addLayer(layerCur, convBlock, {inVar}, {outVar},{[layerCur '_f'], [layerCur '_b']});

f  = net.getParamIndex([layerCur '_f']) ;
sc = sqrt(2/(dims(1)*dims(2)*dims(4))) ; %improved Xavier
weight = sc*randn(dims, 'single');
net.params(f).value        = orthrize(weight);
net.params(f).learningRate = lr(1);
net.params(f).weightDecay  = 1;
net.params(f).trainMethod  = trainMethod;

f = net.getParamIndex([layerCur '_b']) ;
net.params(f).value        = zeros(dims(3), 1, 'single');
net.params(f).learningRate = lr(2);
net.params(f).weightDecay  = 1;
net.params(f).trainMethod  = trainMethod;

inVar    = outVar;
blockNum = blockNum + 1;
end


% add a Conv layer
function [net, inVar, blockNum] = addConv(net, blockNum, inVar, dims, pad, stride, lr)
opts.cudnnWorkspaceLimit = +inf; % 2GB
convOpts    = {'CudnnWorkspaceLimit', opts.cudnnWorkspaceLimit} ;
trainMethod = 'adam';

outVar      = sprintf('conv%d', blockNum);
layerCur    = sprintf('conv%d', blockNum);

convBlock   = dagnn.Conv('size', dims, 'pad', pad,'stride', stride, ...
    'hasBias', true, 'opts', convOpts);

net.addLayer(layerCur, convBlock, {inVar}, {outVar},{[layerCur '_f'], [layerCur '_b']});

f = net.getParamIndex([layerCur '_f']) ;
sc = sqrt(2/(dims(1)*dims(2)*max(dims(3), dims(4)))) ; %improved Xavier
weight = sc*randn(dims, 'single') ;
net.params(f).value        = orthrize(weight);
net.params(f).learningRate = lr(1);
net.params(f).weightDecay  = 1;
net.params(f).trainMethod  = trainMethod;

f = net.getParamIndex([layerCur '_b']) ;
net.params(f).value        = zeros(dims(4), 1, 'single');
net.params(f).learningRate = lr(2);
net.params(f).weightDecay  = 1;
net.params(f).trainMethod  = trainMethod;

inVar    = outVar;
blockNum = blockNum + 1;
end

function A = orthrize(A)
B = A;

A = reshape(A,[size(A,1)*size(A,2)*size(A,3),size(A,4),1,1]);
if size(A,1)> size(A,2)
    [U,S,V] = svd(A,0);
else
    [U,S,V] = svd(A,'econ');
end

S1 =ones(size(diag(S)));
A = U*diag(S1)*V';
A = reshape(A,size(B));

end

function A = clipping(A,b)
A(A>=0&A<b) = b;
A(A<0&A>-b) = -b;
end


