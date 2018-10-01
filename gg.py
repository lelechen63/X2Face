import numpy as np 
a = '/u/lchen63/project/X2Face/UnwrapMosaic/examples/audio_features/Cristin_Milioti/1.6/IblJpk1GDZA/0004575.npz'
a = np.load(a)['audio_feat']
print a 
print a.shape