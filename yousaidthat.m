clear all

run matconvnet_gen/matlab/vl_setupnn
addpath utils

% ===== PARAMS =====

inputfile   = 'data/transcript.mp3';
tmpwavfile  = 'tmp.wav';
writepath   = 'data/output';

asf = 1; % Audio start frame
opt.frame_rate = 25;
val = 640;

opt.audio.window   = [0 1];
opt.audio.fs       = 16000;
opt.audio.Tw       = 25;
opt.audio.Ts       = 10;            % analysis frame shift (ms)
opt.audio.alpha    = 0.97;          % preemphasis coefficient
opt.audio.R        = [ 300 3700 ];  % frequency range to consider
opt.audio.M        = 40;            % number of filterbank channels 
opt.audio.C        = 13;            % number of cepstral coefficients
opt.audio.L        = 22;            % cepstral sine lifter parameter


% ===== LOAD ORIGINAL NET =====

netStructv201 = load('model/v201.mat'); 
global net
net = dagnn.DagNN.loadobj(netStructv201.net);
net.mode = 'test';
net.move('gpu')

names = {'loss1a','loss1b','loss2a','loss2b','loss1','loss2','loss_SR'} ;
for i = 1:numel(names)
  try
    layer = net.layers(net.getLayerIndex(names{i})) ;
    net.removeLayer(names{i}) ;
    net.renameVar(layer.outputs{1}, layer.inputs{1}, 'quiet', true) ;
  end
end

% ===== LOAD DEBLUR NET =====

netStructv114 = load('model/v114.mat'); 
global netR

netR = dagnn.DagNN.loadobj(netStructv114.net);
netR.mode = 'test';
netR.move('gpu')

names = {'loss'} ;
for i = 1:numel(names)
  layer = netR.layers(netR.getLayerIndex(names{i})) ;
  netR.removeLayer(names{i}) ;
  netR.renameVar(layer.outputs{1}, layer.inputs{1}, 'quiet', true) ;
end

fprintf('Net loaded.\n');

%=========csv to data =====%
fid = fopen('/u/lchen63/data/mat/test_yousaidthat.csv');
tline = fgetl(fid);
while ischar(tline)
  cells = strsplit(tline,',')
  faceimg= imread(cells{1});
  faceimg = permute(faceimg,[3 2 1]);
  for i = 2:5
    faceimg = cat(1,faceimg,permute(imread(cells{i}),[3 2 1]));
  end
  faceY   = gpuArray( reshape(faceimg,[112,112,15,1]));

  audio_path  = cells{6}
  Zpad              = zeros(16000,1);
  [Zo1,fs]          = audioread(audio_path);

  Z = [Zpad; Zo1; Zpad];
  [ C, F, ~ ]     = runmfcc( Z, opt.audio );
  forwardpass(cells{7}, C,F, faceY);
end

%% ===== FORWARD PASS =====
function forwardpass(img_path, C,F, faceY)
  padn    = 2;
  Y       = cell(0);
  cn = 1

  for j = 1:4: size(C,2)-34

      mfcc = gpuArray(repmat(single(C (2:13,j:j+34)),1,1,1,1));

      net.eval({'input_audio',mfcc,'input_face',faceY});

      im = gather(net.getVar('prediction').value);
      netR.eval({'input_lip',gpuArray(im)});
      imb         = gather(netR.vars(end).value);
      netR.eval({'input_lip',gpuArray(im)});
      gg = im +imb
      cc = gg(:,:,:,1)
      name = [img_path 'fake_'  num2str(cn, '%03d') '.jpg']

      imwrite(cc,filename)
      cn = cn + 1
      fprintf('name')

  end
end
