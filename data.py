import numpy as np
import argparse
from imutils import face_utils
import dlib
import cv2
import os, fnmatch, shutil
from random import shuffle
from copy import deepcopy
import pickle
from skimage import transform as tf
import scipy.io
import librosa
import csv
from shutil import copyfile

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../basics/shape_predictor_68_face_landmarks.dat')
ms_img = np.load('../basics/mean_shape_img.npy')
ms_norm = np.load('../basics/mean_shape_norm.npy')
S = np.load('../basics/S.npy')

MSK = np.reshape(ms_norm, [1, 68*2])
SK = np.reshape(S, [1, S.shape[0], 68*2])
      
def crop_image(image_path):
    
  
    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

 
    rects = detector(gray, 1)
    for (i, rect) in enumerate(rects):

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
      
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        center_x = x + int(0.5 * w)
        center_y = y + int(0.5 * h)

        
        r = int(0.64 * h)
        new_x = center_x - r
        new_y = center_y - r
        roi = image[new_y:new_y + 2 * r, new_x:new_x + 2 * r]
        
        roi = cv2.resize(roi, (163,163), interpolation = cv2.INTER_AREA)
        scale =  163. / (2 * r)
       
        shape = ((shape - np.array([new_x,new_y])) * scale)


        r = int(0.85 * h)
        new_x = center_x - r
        new_y = center_y - r
        roi2 = image[new_y:new_y + 2 * r, new_x:new_x + 2 * r]
        
        roi2 = cv2.resize(roi2, (224,224), interpolation = cv2.INTER_AREA)
       

    
        return roi, shape, roi2



def generating_landmark_lips(lists, name):
    CC =0
    root_path = '/u/lchen63/data'
    image_txt = lists

    if not os.path.exists(os.path.join(root_path,'regions',  name)):
        os.mkdir(os.path.join(root_path,'regions',  name))
    if not os.path.exists(os.path.join(root_path,'faces',  name)):
        os.mkdir(os.path.join(root_path,'faces',  name))
    if not os.path.exists(os.path.join(root_path,'landmark1d') + '/' + name):
            os.mkdir(os.path.join(root_path,'landmark1d') + '/' + name)
    for line in image_txt:
        img_path = os.path.join(root_path, 'images',name, line)
        

       
        landmark_path = os.path.join(root_path,'landmark1d') + '/' + \
            name + '/' + line.replace('jpg','npy')
        lip_path = os.path.join(root_path,'regions') + '/' + \
            name + '/' + line
        face_path = os.path.join(root_path,'faces') + '/' + \
            name + '/' + line
        print (landmark_path)
        print (lip_path)
        print (face_path)
        
        try:
            
            roi, landmark, roi2 = crop_image(img_path)
            cv2.imwrite(face_path, roi2)
            print (face_path)
            if  np.sum(landmark[37:39,1] - landmark[40:42,1]) < -9:

                # pts2 = np.float32(np.array([template[36],template[45],template[30]]))
                template = np.load('../basics/base_68.npy')
            else:
                template = np.load('../basics/base_68_close.npy')
            # pts2 = np.float32(np.vstack((template[27:36,:], template[39,:],template[42,:],template[45,:])))
            pts2 = np.float32(template[27:45,:])
            # pts2 = np.float32(template[17:35,:])
            # pts1 = np.vstack((landmark[27:36,:], landmark[39,:],landmark[42,:],landmark[45,:]))
            pts1 = np.float32(landmark[27:45,:])
            # pts1 = np.float32(landmark[17:35,:])
            tform = tf.SimilarityTransform()
            tform.estimate( pts2, pts1)
            dst = tf.warp(roi, tform, output_shape=(163, 163))

            dst = np.array(dst * 255, dtype=np.uint8)
            dst = dst[1:129,1:129,:]
            cv2.imwrite(lip_path, dst)
        

            gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

            # detect faces in the grayscale image
            rects = detector(gray, 1)
            for (i, rect) in enumerate(rects):

                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                print (shape)
                (x, y, w, h) = cv2.boundingRect(shape[48:,:])

                center_x = x + int(0.5 * w)

                center_y = y + int(0.5 * h)

                if w > h:
                    r = int(0.65 * w)
                else:
                    r = int(0.65 * h)
                new_x = center_x - r
                new_y = center_y - r
                roi = dst[new_y:new_y + 2 * r, new_x:new_x + 2 * r]

                for inx in range(len(shape)):
                    x=int(shape[inx][1])
                    y =int(shape[inx][0])
                    cv2.circle(dst, (y, x), 1, (0, 0, 255), -1)
                cv2.imwrite(landmark_path.replace('.npy','.jpg'), dst)
                # shape, _, _ = normLmarks(shape)
                shape = shape.reshape(68,2)
                np.save(landmark_path, shape)
            CC += 1
                
            
        except:
            print ('====================================')
            continue

def img_extractor(video_name):

    root_path = '/u/lchen63/data'
    if not os.path.exists(os.path.join(root_path, 'images', video_name)):
        os.mkdir(os.path.join(root_path, 'images',video_name))

    v_path = os.path.join(root_path, 'video', video_name + '.mp4')

    # img_path = os.path.join(root_path, 'images' , video_name+ '_%05d.jpg')
    command = ' ffmpeg -i ' + v_path + ' -r 25 -q:v 2   ' + os.path.join(root_path, 'images' , video_name, video_name+ '_%05d.jpg')
    os.system(command)
    # vidcap = cv2.VideoCapture(v_path)
    # success,image = vidcap.read()
    # count = 1
    # success = True
    # while success:
    #   cv2.imwrite(os.path.join(root_path, 'images' , video_name, video_name+ '_%05d.jpg')%count, image)     # save frame as JPEG file
    #   success,image = vidcap.read()
    #   print ('Read a new frame: ', success)
    #   count += 1
# img_extractor('Eddie_Kaye_Thomas')
def audio_extractor(video_name):
    root_path = '/u/lchen63/data'
    v_path = os.path.join(root_path, 'video', video_name + '.mp4')

    # img_path = os.path.join(root_path, 'images' , video_name+ '_%05d.jpg')
    command = ' ffmpeg -i ' + v_path + ' -ar 16000  -ac 1  -y ' + os.path.join(root_path, 'audio' , video_name+ '.wav')
    os.system(command)
def video_cut(video_name): 
    root_path = '/u/lchen63/data'
    v_path = os.path.join(root_path, 'video', video_name + '.mp4')

    new_v_path = os.path.join(root_path, 'video', video_name + '_short.mp4')
    command = 'ffmpeg -i '+ v_path + ' -vcodec copy -acodec copy -ss 00:00:53.000 -t 00:00:28.000 ' + new_v_path

    os.system(command)

# def image_to_video(sample_dir = None, video_name = None):
    
#     command = 'ffmpeg -framerate 25  -i ' + sample_dir +  '/%05d.png -c:v libx264 -y -vf format=yuv420p ' + video_name 
#     #ffmpeg -framerate 25 -i real_%d.png -c:v libx264 -y -vf format=yuv420p real.mp4
#     print (command)
#     os.system(command)

# def add_audio(video_name=None, audio_dir = None):

#     command = 'ffmpeg -i ' + video_name  + ' -i ' + audio_dir + ' -vcodec copy  -acodec copy -y  ' + video_name.replace('.mp4','.mov')
#     #ffmpeg -i /mnt/disk1/dat/lchen63/lrw/demo/new/resutls/results.mp4 -i /mnt/disk1/dat/lchen63/lrw/demo/new/audio/obama.wav -codec copy -c:v libx264 -c:a aac -b:a 192k  -shortest -y /mnt/disk1/dat/lchen63/lrw/demo/new/resutls/results.mov
#     # ffmpeg -i gan_r_high_fake.mp4 -i /mnt/disk1/dat/lchen63/lrw/demo/audio/obama.wav -vcodec copy  -acodec copy -y   gan_r_high_fake.mov

#     print (command)
#     os.system(command)


# video_cut('PeterCapaldi')
def img2video(video_name):
    root_path = '/u/lchen63/data'
    img_path = os.path.join(root_path, 'images' , video_name)

    v_path = os.path.join(root_path, 'video' , video_name + '_compose.mp4')
    audio_path = os.path.join(root_path, 'audio' , video_name  + '.wav')

    command = 'ffmpeg -framerate 25  -i ' + img_path +  '/' + video_name + '_%05d.jpg ' + ' -c:v libx264 -y -vf format=yuv420p  -y ' + v_path
    print (command)
    os.system(command)
    command = 'ffmpeg -i ' + v_path  + ' -i ' + audio_path + ' -vcodec copy  -acodec copy -y  ' + v_path.replace('.mp4','.mov')
    os.system(command)
# img2video("PeterCapaldi_short")
# audio_extractor('Elektra')


def lists(name ):


    imgs = os.listdir(os.path.join( '/u/lchen63/data/images/', name))
    imgs = sorted(imgs)

    return imgs, name


def  read_pickle():
    audio_path_root = '/mnt/disk1/dat/lchen63/lrw/data/lrw_backup/audio'
    dataset_dir = "/mnt/disk1/dat/lchen63/lrw/data/pickle/"
    _file = open(os.path.join(dataset_dir, "new_img_full_gt_test.pkl"), "rb")
    test_data = pickle.load(_file)
    _file.close()
    count = 0
    csv_file = open(os.path.join( '/u/lchen63/data/mat' ,'test.csv' ),'w')
    csv_writer = csv.writer(csv_file, delimiter=',')

    data = []
    for i in range(0,len(test_data)):
        gg = []
        if test_data[i][1] < 7 or test_data[i][1] > 21:
            continue
        tmp  = test_data[i][0].split('/')
        print (tmp)
        
        if not  os.path.exists(os.path.join('/u/lchen63/data/lrw', 'audio') ):
            os.mkdir(os.path.join('/u/lchen63/data/lrw', 'audio'))

        if not os.path.exists(os.path.join('/u/lchen63/data/lrw', 'audio', tmp[0]) ):
            os.mkdir(os.path.join('/u/lchen63/data/lrw', 'audio', tmp[0]))

        if not os.path.exists(os.path.join('/u/lchen63/data/lrw', 'audio', tmp[0], tmp[1]) ):
            os.mkdir(os.path.join('/u/lchen63/data/lrw', 'audio', tmp[0], tmp[1]))

        
        if not os.path.exists(os.path.join('/u/lchen63/data/lrw', 'image') ):
            os.mkdir(os.path.join('/u/lchen63/data/lrw', 'image'))

        if not os.path.exists(os.path.join('/u/lchen63/data/lrw', 'image', tmp[0]) ):
            os.mkdir(os.path.join('/u/lchen63/data/lrw', 'image', tmp[0]))

        if not os.path.exists(os.path.join('/u/lchen63/data/lrw', 'image', tmp[0], tmp[1]) ):
            os.mkdir(os.path.join('/u/lchen63/data/lrw', 'image', tmp[0], tmp[1]))

        if not os.path.exists(os.path.join('/u/lchen63/data/lrw', 'image', tmp[0], tmp[1], tmp[2]) ):
            os.mkdir(os.path.join('/u/lchen63/data/lrw', 'image', tmp[0], tmp[1], tmp[2]))


        img_path = os.path.join('/mnt/disk1/dat/lchen63/lrw/data/image' , test_data[i][0])

        copyfile(img_path, os.path.join('/u/lchen63/data/lrw', 'image', test_data[i][0]))

        audio_path = os.path.join(audio_path_root, tmp[0], tmp[1],tmp[2]+'.wav')
        audio,_ = librosa.load(audio_path, sr = 16000)
        print (audio)

        librosa.output.write_wav(os.path.join('/u/lchen63/data/lrw', 'audio', tmp[0], tmp[1] ,tmp[2]+'.wav'), audio, 16000)
        # print (test_data[i])
        print(img_path)
        print(audio_path)

        gg.append(os.path.join('/u/lchen63/data/lrw', 'image', test_data[i][0]))
        gg.append(os.path.join('/u/lchen63/data/lrw', 'audio', tmp[0], tmp[1] ,tmp[2]+'.wav'))
        gg.append(test_data[i][1])
        data.append(gg)
        csv_writer.writerow(gg)
        count += 1
        if count == 10000:
            break


def  read_pickle_yousaidthat():
    audio_path_root = '/mnt/disk1/dat/lchen63/lrw/data/lrw_backup/audio'
    dataset_dir = "/mnt/disk1/dat/lchen63/lrw/data/pickle/"
    _file = open(os.path.join(dataset_dir, "new_img_full_gt_test.pkl"), "rb")
    test_data = pickle.load(_file)
    _file.close()
    count = 0
    csv_file = open(os.path.join( '/u/lchen63/data/mat' ,'test_yousaidthat.csv' ),'w')
    csv_writer = csv.writer(csv_file, delimiter=',')
    gs = [1,5,10,15,20]
    data = []
    for i in range(0,len(test_data)):
        # if test_data[i][1] < 5:
        #     continue
        tmp  = test_data[i][0].split('/')
        
        if not  os.path.exists(os.path.join('/u/lchen63/data/lrw_yousaidthat', 'audio') ):
            os.mkdir(os.path.join('/u/lchen63/data/lrw_yousaidthat', 'audio'))

        if not os.path.exists(os.path.join('/u/lchen63/data/lrw_yousaidthat', 'audio', tmp[0]) ):
            os.mkdir(os.path.join('/u/lchen63/data/lrw_yousaidthat', 'audio', tmp[0]))

        if not os.path.exists(os.path.join('/u/lchen63/data/lrw_yousaidthat', 'audio', tmp[0], tmp[1]) ):
            os.mkdir(os.path.join('/u/lchen63/data/lrw_yousaidthat', 'audio', tmp[0], tmp[1]))

        
        if not os.path.exists(os.path.join('/u/lchen63/data/lrw_yousaidthat', 'image') ):
            os.mkdir(os.path.join('/u/lchen63/data/lrw_yousaidthat', 'image'))

        if not os.path.exists(os.path.join('/u/lchen63/data/lrw_yousaidthat', 'image', tmp[0]) ):
            os.mkdir(os.path.join('/u/lchen63/data/lrw_yousaidthat', 'image', tmp[0]))

        if not os.path.exists(os.path.join('/u/lchen63/data/lrw_yousaidthat', 'image', tmp[0], tmp[1]) ):
            os.mkdir(os.path.join('/u/lchen63/data/lrw_yousaidthat', 'image', tmp[0], tmp[1]))

        if not os.path.exists(os.path.join('/u/lchen63/data/lrw_yousaidthat', 'image', tmp[0], tmp[1], tmp[2]) ):
            os.mkdir(os.path.join('/u/lchen63/data/lrw_yousaidthat', 'image', tmp[0], tmp[1], tmp[2]))


        img_path = os.path.join('/mnt/disk1/dat/lchen63/lrw/data/regions' , test_data[i][0])
        img = cv2.imread(img_path)
        img = cv2.resize(img, (112,112), interpolation = cv2.INTER_AREA)
        cv2.imwrite(os.path.join('/u/lchen63/data/lrw_yousaidthat', 'image', test_data[i][0]) , img)
        #copyfile(img_path, os.path.join('/u/lchen63/data/lrw_yousaidthat', 'image', test_data[i][0]))

        audio_path = os.path.join(audio_path_root, tmp[0], tmp[1],tmp[2]+'.wav')
        audio,_ = librosa.load(audio_path, sr = 16000)

        librosa.output.write_wav(os.path.join('/u/lchen63/data/lrw_yousaidthat', 'audio', tmp[0], tmp[1] ,tmp[2]+'.wav'), audio, 16000)
        # print (test_data[i])
        if os.path.join('/u/lchen63/data/lrw_yousaidthat', 'audio', tmp[0], tmp[1] ,tmp[2]+'.wav') not in data:
            gg = []
            for g in gs:
                gg.append(os.path.join('/u/lchen63/data/lrw_yousaidthat', 'image', test_data[i][0][:-7] + "%03d.jpg"%g))
            gg.append(os.path.join('/u/lchen63/data/lrw_yousaidthat', 'audio', tmp[0], tmp[1] ,tmp[2]+'.wav'))
            gg.append(os.path.join('/u/lchen63/data/lrw_yousaidthat', 'image', test_data[i][0][:-7]))
            data.append(os.path.join('/u/lchen63/data/lrw_yousaidthat', 'audio', tmp[0], tmp[1] ,tmp[2]+'.wav'))

            csv_writer.writerow(gg)
            count += 1
            print count
        if count == 300:
            break


def  read_pickle_yousaidthat_voxceleb():
    audio_path_root = '/home/cxu-serve/p1/lchen63/voxceleb/audio'
    image_path_root = '/home/cxu-serve/p1/lchen63/voxceleb/regions'
    dataset_dir = "/mnt/disk1/dat/lchen63/lrw/data/pickle/"
    _file = open("/home/cxu-serve/p1/lchen63/voxceleb/vox1_meta.csv", "rb")
    csv_file = open(os.path.join( '/u/lchen63/data/mat' ,'voxceleb_test_yousaidthat.csv' ),'w')
    csv_writer = csv.writer(csv_file, delimiter=',')
    gs = [1,5,10,15,20]
    data = []
    count = 0
    for line in _file:
        tmps = line.split()
        print (tmps)
        if tmps[-1] != 'test':
            continue

        name = tmps[1]

        region_path = os.path.join(image_path_root, name)
        audio_path = os.path.join(audio_path_root, name)
        if not os.path.exists(region_path):
            print ('==================')
            print (region_path)
        elif not os.path.exists(audio_path):
            print ('=========++++====')
            print (audio_path)

        video_names = os.listdir(region_path)

        for video_name in video_names:
            v_path = os.path.join(region_path, video_name)
            clips = os.listdir(v_path)
            gg = []
            for clip in clips:
                if os.path.exists(os.path.join(audio_path, video_name, '%05d.wav'%(int(clip)))):
                    flage = True

                    for i_n in os.listdir(os.path.join(region_path, video_name, clip)):
                        img_name = os.path.join(v_path, clip,  i_n)
                        if i_n[-8:] == '_112.jpg':
                            os.system('rm ' + img_name )
                            continue
                        
                        img = cv2.imread(img_name)
                        img = cv2.resize(img, (112,112), interpolation = cv2.INTER_AREA)
                        # cv2.imwrite(img_name.replace('.jpg','_112.jpg') , img)

                    # for g in gs:
                    #     if os.path.exists(os.path.join(v_path, clip,  "%02d_112.jpg"%g)):
                    #         gg.append(os.path.join(v_path, clip,  "%02d_112.jpg"%g))
                    #     else:
                    #         flage= False

                    #         continue

                    gg.append(os.path.join(audio_path, video_name, '%05d.wav'%(int(clip))))
                    gg.append(os.path.join(region_path, video_name, clip))

                    csv_writer.writerow(gg)
                    count += 1
        #         print count
        #         if count == 2:
        #             break
        #     print count
        #     if count == 2:
        #         break
        # print count
        # if count == 2:
        #     break
                
read_pickle_yousaidthat_voxceleb()
# read_pickle_yousaidthat()
# audio_extractor('EzraMiller')
# imgs, name = lists('Eddie_Kaye_Thomas')
# generating_landmark_lips(imgs, name)

