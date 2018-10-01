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
    
        return roi, shape 



def generating_landmark_lips(lists, name):
    CC =0
    root_path = '/u/lchen63/data'
    image_txt = lists

    
    for line in image_txt:
        img_path = os.path.join(root_path, 'images',name, line)
        print (img_path)

       

        if not os.path.exists(os.path.join(root_path,'regions') + '/' + name):
            os.mkdir(os.path.join(root_path,'regions') + '/' +  name)

        if not os.path.exists(os.path.join(root_path,'landmark1d') + '/' + name):
            os.mkdir(os.path.join(root_path,'landmark1d') + '/' + name)

       
        landmark_path = os.path.join(root_path,'landmark1d') + '/' + \
            name + '/' + line.replace('jpg','npy')
        lip_path = os.path.join(root_path,'regions') + '/' + \
            name + '/' + line
        print (landmark_path)
        print (lip_path)
        
        # try:
        
        roi, landmark= crop_image(img_path)
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
            
            
        # except:
        #     print ('====================================')
        #     continue

def img_extracter(video_name):
    root_path = '/u/lchen63/data'
    v_path = os.path.join(root_path, 'video', video_name + '.mp4')

    # img_path = os.path.join(root_path, 'images' , video_name+ '_%05d.jpg')
    # command = ' ffmpeg -i ' + v_path + ' -r 25  -bsf:v mjpeg2jpeg   ' + os.path.join(root_path, 'images' , video_name, video_name+ '_%05d.jpg')
    # os.system(command)
    vidcap = cv2.VideoCapture(v_path)
    success,image = vidcap.read()
    count = 1
    success = True
    while success:
      cv2.imwrite(os.path.join(root_path, 'images' , video_name+ '_%05d.jpg')%count, image)     # save frame as JPEG file
      success,image = vidcap.read()
      print ('Read a new frame: ', success)
      count += 1


def lists(name ):


    imgs = os.listdir(os.path.join( '/u/lchen63/data/images/', name))
    imgs = sorted(imgs)

    return imgs, name

img_extracter('EzraMiller')
# imgs, name = lists('EzraMiller')
# generating_landmark_lips(imgs, name)

