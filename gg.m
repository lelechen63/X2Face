

% SyncNet for all angles - v0.1 - 3 Apr 2017 
% Y - grayscale image frames, stacked along the 3rd dimension
% Z - audio signal -- 16KHz mono

clear all
run  ~/matconvnet-1.0-beta25/matlab/vl_setupnn.m 	% Path to MatConvNet Vers 23
addpath tools

modelPath   = 'model/syncnet_v7.mat';
dataPath    = 'data/ex1.mat';

%% Fixed parameters

shiftw  = 24;
shift   = -shiftw:shiftw;             % Audio-video offsets to search
Yframes = 5;
Aframes	= 20;

opt.fs       = 16000;
opt.Tw       = 25;
opt.Ts       = 10;            % analysis frame shift (ms)
opt.alpha    = 0.97;          % preemphasis coefficient
opt.R        = [ 300 3700 ];  % frequency range to consider
opt.M        = 40;            % number of filterbank channels 
opt.C        = 13;            % number of cepstral coefficients
opt.L        = 22;            % cepstral sine lifter parameter

%% Load net

netStruct = load(modelPath); 

net = dagnn.DagNN.loadobj(netStruct.net);
net.move('gpu')
net.conserveMemory = 0;
net.mode = 'test' ;

% names = {'dist','loss'};
% for i = 1:numel(names)
%   layer = net.layers(net.getLayerIndex(names{i})) ;
%   net.removeLayer(names{i}) ;
%   net.renameVar(layer.outputs{1}, layer.inputs{1}, 'quiet', true) ;
% end
% save(modelPath,'net');

aud_id = structfind(net.vars,'name','x24_audio');
lip_id = structfind(net.vars,'name','x24_lip');

fprintf('Network loaded. \n');

%% Load and prepare data
fid = fopen('/u/lchen63/data/mat/test.csv');
tline = fgetl(fid);
while ischar(tline)
	cells = strsplit(tline,',')
	audio_path  = cells{2}
	frame = str2num(cells{3})
	export_audio(audio_path, frame , opt, net, aud_id);
end


function export_audio(audio_path, id, opt, net, aud_id)
	audio_file = [audio_path];
	imdb = struct('frame', [], 'audio_ft', []);
	data = audioread(audio_file);
	offset = id * 16000 / 25;
	pad = (21 / 100) * opt.fs;
	Z = data(offset-pad:offset+pad);
	[ C, ~, ~ ] = runmfcc( Z, opt );
	C = bsxfun(@minus, single(C), net.meta.normalization_a.averageImage);
	C = C(2:13,:);
	for j = 1:3
			c = C(:,((j-1)*10+1:(j-1)*10+20));
			net.eval({'input_audio',gpuArray(c)});
			Cf{j}   = squeeze(gather(net.vars(aud_id).value));
	end
	fprintf('Audio features extracted. \n');
	cf  = cat(2,Cf{:});
	save([strrep(audio_path, '.wav', '_') num2str(id)] '.mat', 'cf');
end 


function export_audio(track_name, opt, net, aud_id)
	audio_file = ['/u/lchen63/data/audio/'  track_name  '.wav'];
	frames = dir(['/u/lchen63/data/faces/' track_name   '/*.jpg']);

	imdb = struct('frame', [], 'audio_ft', []);

	%if exist(['/u/lchen63/data/audio/' track_name  '/'], 'dir') 
	%	return
	%end

	data = audioread(audio_file);
	for f=65:85
		fprintf(frames(f).name)
        fprintf(frames(f).name(end-8:end-4))
		offset = str2num(frames(f).name(end-8:end-4)) * 16000 / 25;

		%track_name, track_id
		pad = (21 / 100) * opt.fs;
		offset, pad, frames(f).name(end-8:end-4)
        size(data)
		Z = data(offset-pad:offset+pad);
		[ C, ~, ~ ] = runmfcc( Z, opt );
		C = bsxfun(@minus, single(C), net.meta.normalization_a.averageImage) ;
		C = C(2:13,:);

		for j = 1:3
    			c = C(:,((j-1)*10+1:(j-1)*10+20));
   			net.eval({'input_audio',gpuArray(c)});
    			Cf{j}   = squeeze(gather(net.vars(aud_id).value));
		end
		Cf
		fprintf('Audio features extracted. \n');
		cf  = cat(2,Cf{:});
        cf_size = size(cf)
% 		gg = struct('frame', frames(f).name, 'audio_ft', cf);
        name = ['/u/lchen63/data/audio/' track_name  '/' track_name '_' num2str(f) '.mat']
        save(name, 'cf')
	end
	if ~exist(['/u/lchen63/data/audio/' track_name  '/'], 'dir')
		mkdir(['/u/lchen63/data/audio/' track_name  '/']);
	end
	save(['/u/lchen63/data/audio/' track_name  '/audio.mat'], 'imdb');
end 