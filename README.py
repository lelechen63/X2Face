
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
names = dir('/scratch/local/ssd/koepke/voxceleb/faces/');



for i=1:size(names,1)
	if names(i).name(1) ~= '.'
		tracks = dir([names(i).folder '/' names(i).name '/1.6/']);
		for t=1:size(tracks,1)
			if tracks(t).name(1) ~= '.'
				export_audio(names(i).name,  tracks(t).name, opt, net, aud_id);
			end
		end
			
	end
	

end

function export_audio(track_name, track_id, opt, net, aud_id)
	audio_file = ['/datasets/voxceleb1/wav/' track_name '/' track_id '/audio.wav'];
	frames = dir(['/scratch/local/ssd/koepke/voxceleb/faces/' track_name '/1.6/' track_id  '/*.jpg']);

	imdb = struct('frame', [], 'audio_ft', []);

	%if exist(['/scratch/local/ssd/ow/voxceleb/audio/' track_name '/' track_id '/'], 'dir') 
	%	return
	%end

	data = audioread(audio_file);
	for f=1:numel(frames)
		offset = str2num(frames(f).name(1:end-4)) * 16000 / 25;
		track_name, track_id
		pad = (21 / 100) * opt.fs;
		offset, pad, frames(f).name(1:end-4)	
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
		imdb(end+1) = struct('frame', frames(f).name, 'audio_ft', cf);
	end
	if ~exist(['/scratch/local/ssd/ow/voxceleb/audio/' track_name '/' track_id '/'], 'dir')
		mkdir(['/scratch/local/ssd/ow/voxceleb/audio/' track_name '/' track_id '/']);
	end
	save(['/scratch/local/ssd/ow/voxceleb/audio/' track_name '/' track_id '/audio.mat'], 'imdb');
end
