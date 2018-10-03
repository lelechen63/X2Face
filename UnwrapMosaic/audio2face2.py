# import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import os
import torch
from PIL import Image
from torch.autograd import Variable
from UnwrappedFace import UnwrappedFaceWeightedAverage, BottleneckFromNet
from sklearn.externals import joblib
from torchvision.transforms import Compose, Scale, ToTensor
from scipy.misc import imsave
os.environ["CUDA_VISIBLE_DEVICES"] = ''
from scipy.io import loadmat
import sys
from tempfile import TemporaryFile
print ('1')
def mat2npy(path):

    matx = loadmat(path)
    print (type(matx))
    print (matx.keys())
    imdb = matx['cf']
    print (imdb.shape)
    audio = imdb[:,1]
    outfile = path.replace('mat', 'npy')
    np.save(outfile, audio )
    # for i in range(1,11):
    #     tt = imdb[0,i][1]        
    #     audio = tt[:,0]
    #     outfile = 'examples/audio_features/test1/EzraMiller_%05d.npy'%(i + 10)
    #     np.save(outfile, audio )



    # d = np.load('examples/audio_features/Cristin_Milioti/1.6/IblJpk1GDZA/0004575.npz')['audio_feat']
    # print (d)
    # print (d.shape)
mat2npy('/u/lchen63/data/audio/EzraMiller/EzraMiller_75.mat')
#


# sys.exit("Error message")

def load_img_and_audio(file_path):
    transform = Compose([Scale((256,256)), ToTensor()])
    img = Image.open(file_path).convert('RGB')
    img = transform(img)
    audio_label_path = str(file_path).replace('audio_faces', 'audio_features').replace('jpg','npz')
    audio_feature = torch.Tensor(np.load(audio_label_path)['audio_feat'])
    return {'image' : img, 'audio' : audio_feature}


def load_img_and_audio1(file_path):
    transform = Compose([Scale((256,256)), ToTensor()])
    print (file_path)
    img_p = file_path[0]
    audio_p = file_path[1]
    print (img_p)
    print (audio_p)
    img = Image.open(img_p).convert('RGB')
    img = transform(img)
    # audio_label_path = str(file_path).replace('audio_faces', 'audio_features').replace('jpg','npy')
    audio_feature = torch.Tensor(np.load(audio_p))
    return {'image' : img, 'audio' : audio_feature}
   # paths to source frames

sourcepaths= [['/u/lchen63/data/faces/EzraMiller/EzraMiller_00075.jpg', '/u/lchen63/data/audio/EzraMiller/EzraMiller_75.npy']]

# path to frames corresponding to driving audio features
audio_path = 'examples/audio_faces/Peter_Capaldi/1.6/uAgUjSqIj7U'
imgpaths = os.listdir(audio_path)

# loading models
BASE_MODEL = '/mnt/ssd0/dat/lchen63/release_models/' # Change to your path
model_path = BASE_MODEL + 'x2face_model.pth'
model = UnwrappedFaceWeightedAverage(output_num_channels=2, input_num_channels=3,inner_nc=128)
model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage)['state_dict'])
s_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
modelfortargetpose = BottleneckFromNet()
state = modelfortargetpose.state_dict()
s_dict = {k: v for k, v in s_dict['state_dict'].items() if k in state.keys()}
state.update(s_dict)
modelfortargetpose.load_state_dict(state)

posemodel = nn.Sequential(nn.Linear(128, 3))
p_dict_pre = torch.load(BASE_MODEL + '/posereg.pth', map_location=lambda storage, loc: storage)['state_dict']
posemodel._modules['0'].weight.data = p_dict_pre['posefrombottle.weight'].cpu()
posemodel._modules['0'].bias.data = p_dict_pre['posefrombottle.bias'].cpu()

bottleneckmodel = nn.Sequential(nn.Linear(3, 128, bias=False), nn.BatchNorm1d(128))
b_dict_pre = torch.load(BASE_MODEL + '/posetobottle.pth', map_location=lambda storage, loc: storage)['state_dict']
bottleneckmodel.load_state_dict(b_dict_pre)
model = model
modelfortargetpose = modelfortargetpose
posemodel = posemodel
bottleneckmodel = bottleneckmodel

model.eval()
modelfortargetpose.eval()
posemodel.eval()
bottleneckmodel.eval()
# load linear regression from audio features to driving vector space
linearregression = joblib.load(BASE_MODEL + '/linearregression_scaledTrue_7000.pkl')
scalar = joblib.load(BASE_MODEL + '/scaler_7000.pkl')
scalar = None

# Drive 3 different identities with same audio
img_gt_gen = np.empty((0,2560,3))
cc = 0
for sourcepath in sourcepaths:
    print (sourcepath)
    print ('===')
    img_to_show_all = np.empty((256,0,3))
    gt_ims = np.empty((256,0,3))
    source_data = load_img_and_audio1(sourcepath)
    source_img = Variable(source_data['image']).unsqueeze(0)
    print ('======')
    audio_feature_source = source_data['audio'].cpu().numpy().reshape(1,-1)
    audio_feature_origin = linearregression.predict(audio_feature_source)
    print ('+++')
    audio_feature_origin = torch.Tensor(audio_feature_origin).unsqueeze(2).unsqueeze(2)
    for imgpath in imgpaths:
        
        # Extract the driving audio features
        fullaudiopath = os.path.join(audio_path, imgpath)
        audio_data = load_img_and_audio(fullaudiopath)
        ggt =  audio_data['image']
        audio_img = Variable(audio_data['image'], volatile=True).unsqueeze(0)
        audio_feature = audio_data['audio'].cpu().numpy().reshape(1,-1)
        if not scalar is None:
            audio_feature = scalar.transform(audio_feature)
            audio_feature_origin = scalar.transform(audio_feature_origin)
        audio_feature = linearregression.predict(audio_feature)
        audio_feature = torch.Tensor(audio_feature).unsqueeze(2).unsqueeze(2)
        
        sourcebn = modelfortargetpose(source_img)
        sourcepose = posemodel(sourcebn.unsqueeze(0))
        sourceposebn = bottleneckmodel(sourcepose)
    
        def update_bottleneck(self, input, output):
            newdrive = sourcebn.unsqueeze(0).unsqueeze(2).unsqueeze(3) + Variable(audio_feature) - Variable(audio_feature_origin)
            audiopose =  posemodel(newdrive.squeeze().unsqueeze(0)) #
            audioposebn = bottleneckmodel(audiopose)
            output[0,:,:,:] = newdrive + sourceposebn.unsqueeze(2).unsqueeze(3) - audioposebn.unsqueeze(2).unsqueeze(3) # if we want to add old pose (of input) and substract pose info that's in the new bottleneck

        # Add a forward hook to update the model's bottleneck
        handle = model.pix2pixSampler.netG.model.submodule.submodule.submodule.submodule.submodule.submodule.submodule.down[1].register_forward_hook(update_bottleneck)
        result = model(source_img, source_img)
        gg = result.squeeze().data.permute(1,2,0).numpy()
        cc += 1
        imsave('results/%d.jpg'%cc,gg )
        ggt = ggt.permute(1,2, 0).numpy()
        imsave('results/gt_%d.jpg'%cc,ggt )
        handle.remove()
        
        img_to_show_all = np.hstack((result.squeeze().data.permute(1,2,0).numpy(), img_to_show_all))
        if img_gt_gen.shape == (0,2560,3):
            gt_ims = np.hstack((audio_img.squeeze().data.permute(1,2,0).numpy(), gt_ims))
    if img_gt_gen.shape == (0,2560,3):
        img_gt_gen = np.vstack((img_gt_gen, gt_ims))
    img_gt_gen = np.vstack((img_gt_gen, img_to_show_all))
plt.rcParams["figure.figsize"] = [14,14]
# plt.imshow(img_gt_gen)
plt.savefig('results/1.jpg')

print('Top row: Frames corresponding to driving audio')
print('Bottom 3 rows: generated frames driven with audio features corresponding to top row')